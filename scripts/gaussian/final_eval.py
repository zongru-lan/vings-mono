import csv
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from skimage.metrics import structural_similarity
from tqdm import tqdm

from gaussian.vis_utils import (
    get_keyframe_render_name,
    prepare_keyframe_render_image,
)


POSE_MATRIX_COLUMNS = [f'm{r}{c}' for r in range(4) for c in range(4)]
REQUIRED_POSE_COLUMNS = [
    'gt_frame_index', 'gt_frame_id', 'gt_image_name', 'gt_image_path',
    'render_name', 'pose_file', 'gt_timestamp', *POSE_MATRIX_COLUMNS,
]


def _log(logger, message):
    if logger is not None:
        logger(message)


def _ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _load_pose_rows(pose_csv):
    if not os.path.exists(pose_csv):
        raise FileNotFoundError(f'Missing pose CSV: {pose_csv}')

    with open(pose_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [col for col in REQUIRED_POSE_COLUMNS if col not in header]
        if missing:
            raise ValueError(f'{pose_csv} is missing required columns: {missing}')
        rows = list(reader)

    if not rows:
        raise ValueError(f'No keyframe pose rows found in {pose_csv}')
    return rows


def _row_c2w(row):
    values = [float(row[col]) for col in POSE_MATRIX_COLUMNS]
    c2w = np.asarray(values, dtype=np.float32).reshape(4, 4)
    if not np.allclose(c2w[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
        raise ValueError(f"Bad homogeneous last row for gt_frame_id={row.get('gt_frame_id')}: {c2w[3]}")
    return c2w


def _render_intrinsic_from_cfg(cfg, device):
    height = int(cfg['frontend']['image_size'][0])
    width = int(cfg['frontend']['image_size'][1])
    raw_h = float(cfg['intrinsic']['H'])
    raw_w = float(cfg['intrinsic']['W'])
    h_scale = height / raw_h
    w_scale = width / raw_w

    return {
        'fu': torch.tensor(float(cfg['intrinsic']['fu']) * h_scale, dtype=torch.float32, device=device),
        'fv': torch.tensor(float(cfg['intrinsic']['fv']) * w_scale, dtype=torch.float32, device=device),
        'cu': torch.tensor(float(cfg['intrinsic']['cu']) * h_scale, dtype=torch.float32, device=device),
        'cv': torch.tensor(float(cfg['intrinsic']['cv']) * w_scale, dtype=torch.float32, device=device),
        'H': height,
        'W': width,
    }


def _tensor_to_uint8_hwc(image_chw):
    array = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _read_gt_rgb(path, expected_hw):
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f'Failed to read GT image: {path}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != expected_hw:
        raise ValueError(
            f'GT image shape mismatch for {path}: got {image_rgb.shape[:2]}, expected {expected_hw}'
        )
    return image_rgb


def _psnr_uint8(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float('inf')
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def _lpips_eval_size(height, width, eval_width):
    eval_width = int(eval_width)
    if eval_width <= 0:
        raise ValueError(f'lpips_eval_width must be positive, got {eval_width}')
    eval_height = max(1, int(round(float(height) * eval_width / float(width))))
    return eval_height, eval_width


def _np_rgb_to_chw_float(image_rgb):
    return torch.from_numpy(image_rgb).permute(2, 0, 1).to(torch.float32) / 255.0


def _compute_lpips(lpips_model, render_image_chw, gt_image_rgb, eval_size, device):
    eval_h, eval_w = eval_size
    render_eval = F.interpolate(
        render_image_chw.unsqueeze(0),
        size=(eval_h, eval_w),
        mode='bilinear',
        align_corners=False,
    )
    gt_eval = F.interpolate(
        _np_rgb_to_chw_float(gt_image_rgb).unsqueeze(0),
        size=(eval_h, eval_w),
        mode='bilinear',
        align_corners=False,
    ).to(device)

    render_eval = render_eval.to(device) * 2.0 - 1.0
    gt_eval = gt_eval * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_model(render_eval, gt_eval).reshape(-1)[0].detach().cpu().item())


def _make_lpips_model(cfg, device):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError('LPIPS metric requires the lpips package. Install project requirements first.') from exc

    net = cfg.get('output', {}).get('lpips_net', 'alex')
    model = lpips.LPIPS(net=net)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _metric_summary(values):
    finite_values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite_values.size == 0:
        return {'count': 0, 'mean': None, 'median': None, 'std': None, 'min': None, 'max': None}
    return {
        'count': int(finite_values.size),
        'mean': float(np.mean(finite_values)),
        'median': float(np.median(finite_values)),
        'std': float(np.std(finite_values)),
        'min': float(np.min(finite_values)),
        'max': float(np.max(finite_values)),
    }


def rerender_final_map_and_metrics(cfg, gaussian_model, pose_rows, metrics_dir, logger=print):
    output_cfg = cfg.get('output', {})
    save_dir = cfg['output']['save_dir']
    render_dir = os.path.join(save_dir, 'renders')
    _ensure_clean_dir(render_dir)

    device = gaussian_model.device
    intrinsic = _render_intrinsic_from_cfg(cfg, device)
    lpips_width = int(output_cfg.get('lpips_eval_width', 1024))
    lpips_model = _make_lpips_model(cfg, device)

    metrics_csv = os.path.join(metrics_dir, 'render_metrics.csv')
    fieldnames = [
        'gt_frame_index', 'gt_frame_id', 'gt_image_name', 'gt_image_path',
        'render_name', 'pose_file', 'gt_timestamp',
        'psnr', 'ssim', 'lpips',
        'render_height', 'render_width', 'lpips_eval_height', 'lpips_eval_width',
    ]
    metric_values = {'psnr': [], 'ssim': [], 'lpips': []}
    rows_written = 0

    _log(logger, f'Final rerendering {len(pose_rows)} keyframes into {render_dir}')
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(pose_rows, desc='final render/eval'):
            c2w = torch.tensor(_row_c2w(row), dtype=torch.float32, device=device)
            w2c = torch.linalg.inv(c2w)
            frame_meta = {
                'render_name': row['render_name'],
                'pose_name': row['pose_file'],
                'frame_id': row['gt_frame_id'],
                'frame_index': row['gt_frame_index'],
                'image_name': row['gt_image_name'],
                'image_path': row['gt_image_path'],
                'matched_timestamp': row['gt_timestamp'],
            }

            with torch.no_grad():
                pred_dict = gaussian_model.render(w2c, intrinsic)
                if cfg.get('use_sky', False) and hasattr(gaussian_model, 'sky_model'):
                    pred_dict_sky = gaussian_model.sky_model.render(w2c, intrinsic)
                    pred_dict['rgb'] = gaussian_model.sky_model.fuse_rgb(pred_dict, pred_dict_sky)
                render_image = prepare_keyframe_render_image(cfg, pred_dict['rgb'])

            render_name = get_keyframe_render_name(cfg, frame_meta, row['gt_frame_id'])
            render_path = os.path.join(render_dir, render_name)
            torchvision.utils.save_image(render_image, render_path)

            render_uint8 = _tensor_to_uint8_hwc(render_image)
            render_h, render_w = render_uint8.shape[:2]
            gt_rgb = _read_gt_rgb(row['gt_image_path'], (render_h, render_w))
            eval_size = _lpips_eval_size(render_h, render_w, lpips_width)

            psnr_value = _psnr_uint8(render_uint8, gt_rgb)
            ssim_value = float(structural_similarity(gt_rgb, render_uint8, channel_axis=2, data_range=255))
            lpips_value = _compute_lpips(lpips_model, render_image, gt_rgb, eval_size, device)

            metric_values['psnr'].append(psnr_value)
            metric_values['ssim'].append(ssim_value)
            metric_values['lpips'].append(lpips_value)

            writer.writerow({
                'gt_frame_index': row['gt_frame_index'],
                'gt_frame_id': row['gt_frame_id'],
                'gt_image_name': row['gt_image_name'],
                'gt_image_path': row['gt_image_path'],
                'render_name': render_name,
                'pose_file': row['pose_file'],
                'gt_timestamp': row['gt_timestamp'],
                'psnr': f'{psnr_value:.10f}' if np.isfinite(psnr_value) else 'inf',
                'ssim': f'{ssim_value:.10f}',
                'lpips': f'{lpips_value:.10f}',
                'render_height': render_h,
                'render_width': render_w,
                'lpips_eval_height': eval_size[0],
                'lpips_eval_width': eval_size[1],
            })
            rows_written += 1

            del render_image, pred_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        'metric_scope': 'final Gaussian map rerendered keyframes only',
        'render_resolution': {
            'height': int(output_cfg.get('render_height', cfg['intrinsic']['H'])),
            'width': int(output_cfg.get('render_width', cfg['intrinsic']['W'])),
        },
        'lpips_eval': {
            'net': output_cfg.get('lpips_net', 'alex'),
            'width': lpips_width,
            'height_policy': 'round(render_height * lpips_eval_width / render_width)',
        },
        'num_keyframes': rows_written,
        'metrics': {name: _metric_summary(values) for name, values in metric_values.items()},
    }
    summary_path = os.path.join(metrics_dir, 'render_metrics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    _log(logger, f'Saved render metrics: {metrics_csv}')


def _infer_gt_pose_path(cfg):
    output_cfg = cfg.get('output', {})
    sequence_name = output_cfg.get('sequence_name') or Path(cfg['dataset']['root']).resolve().name
    default_gt_pose_root = Path(cfg['dataset']['root']).resolve().parent / 'gt_poses'
    gt_pose_root = Path(output_cfg.get('gt_pose_root', default_gt_pose_root))
    return gt_pose_root / f'{sequence_name}.parquet'


def _read_gt_pose_table(gt_pose_path):
    try:
        import pandas as pd
        return pd.read_parquet(gt_pose_path)
    except ImportError as exc:
        raise RuntimeError(
            'ATE evaluation requires pandas plus a parquet engine. Install pyarrow or fastparquet '
            f'to read {gt_pose_path}.'
        ) from exc
    except Exception as exc:
        message = str(exc)
        if 'pyarrow' in message or 'fastparquet' in message or 'parquet' in message.lower():
            raise RuntimeError(
                'ATE evaluation requires a parquet engine. Install pyarrow or fastparquet '
                f'to read {gt_pose_path}.'
            ) from exc
        raise


def _nearest_gt_indices(query_timestamps, gt_timestamps, tolerance_sec):
    order = np.argsort(gt_timestamps)
    sorted_timestamps = gt_timestamps[order]
    indices = []
    errors = []
    unmatched = []

    for row_idx, timestamp in enumerate(query_timestamps):
        pos = int(np.searchsorted(sorted_timestamps, timestamp))
        candidates = []
        if pos < len(sorted_timestamps):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            unmatched.append(row_idx)
            continue
        best_pos = min(candidates, key=lambda i: abs(sorted_timestamps[i] - timestamp))
        dt = float(abs(sorted_timestamps[best_pos] - timestamp))
        if dt > tolerance_sec:
            unmatched.append(row_idx)
            continue
        indices.append(int(order[best_pos]))
        errors.append(dt)

    return np.asarray(indices, dtype=np.int64), np.asarray(errors, dtype=np.float64), unmatched


def _umeyama(src, dst, with_scale):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f'Invalid alignment shapes: src={src.shape}, dst={dst.shape}')
    if src.shape[0] < 3:
        raise ValueError('At least 3 matched poses are required for robust trajectory alignment.')

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    s_matrix = np.diag(sign)
    rotation = u @ s_matrix @ vt

    if with_scale:
        src_var = float(np.mean(np.sum(src_centered ** 2, axis=1)))
        if src_var <= 0.0:
            raise ValueError('Cannot estimate Sim(3) scale from zero-variance estimated trajectory.')
        scale = float(np.sum(singular_values * sign) / src_var)
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    aligned = (scale * (rotation @ src.T)).T + translation
    return aligned, {'scale': scale, 'rotation': rotation, 'translation': translation}


def _ate_stats(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
    }


def compute_ate_metrics(cfg, pose_rows, metrics_dir, logger=print):
    gt_pose_path = _infer_gt_pose_path(cfg)
    if not gt_pose_path.exists():
        raise FileNotFoundError(f'Missing GT pose parquet for ATE: {gt_pose_path}')

    gt_df = _read_gt_pose_table(gt_pose_path)
    required = ['timestamp', 't_x', 't_y', 't_z']
    missing = [col for col in required if col not in gt_df.columns]
    if missing:
        raise ValueError(f'{gt_pose_path} is missing required columns: {missing}')

    sorted_rows = sorted(pose_rows, key=lambda row: float(row['gt_timestamp']))
    est_positions = np.asarray([_row_c2w(row)[:3, 3] for row in sorted_rows], dtype=np.float64)
    query_timestamps = np.asarray([float(row['gt_timestamp']) for row in sorted_rows], dtype=np.float64)

    gt_timestamps = gt_df['timestamp'].to_numpy(dtype=np.float64)
    tolerance = float(cfg.get('output', {}).get('ate_timestamp_tolerance_sec', 0.05))
    gt_indices, timestamp_errors, unmatched = _nearest_gt_indices(query_timestamps, gt_timestamps, tolerance)
    if len(gt_indices) < 3:
        raise ValueError(
            f'Only {len(gt_indices)} poses matched GT timestamps within {tolerance}s; cannot compute ATE.'
        )

    matched_rows = [row for i, row in enumerate(sorted_rows) if i not in set(unmatched)]
    est_positions = np.asarray([_row_c2w(row)[:3, 3] for row in matched_rows], dtype=np.float64)
    query_timestamps = np.asarray([float(row['gt_timestamp']) for row in matched_rows], dtype=np.float64)
    gt_positions = gt_df.iloc[gt_indices][['t_x', 't_y', 't_z']].to_numpy(dtype=np.float64)

    raw_aligned = est_positions.copy()
    se3_aligned, se3_transform = _umeyama(est_positions, gt_positions, with_scale=False)
    sim3_aligned, sim3_transform = _umeyama(est_positions, gt_positions, with_scale=True)

    raw_errors = np.linalg.norm(raw_aligned - gt_positions, axis=1)
    se3_errors = np.linalg.norm(se3_aligned - gt_positions, axis=1)
    sim3_errors = np.linalg.norm(sim3_aligned - gt_positions, axis=1)

    trajectory_csv = os.path.join(metrics_dir, 'ate_aligned_trajectory.csv')
    with open(trajectory_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'gt_frame_id', 'gt_image_name', 'gt_timestamp', 'gt_time_error_sec',
            'est_tx', 'est_ty', 'est_tz',
            'gt_tx', 'gt_ty', 'gt_tz',
            'raw_error',
            'se3_tx', 'se3_ty', 'se3_tz', 'se3_error',
            'sim3_tx', 'sim3_ty', 'sim3_tz', 'sim3_error',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(matched_rows):
            writer.writerow({
                'gt_frame_id': row['gt_frame_id'],
                'gt_image_name': row['gt_image_name'],
                'gt_timestamp': f'{query_timestamps[i]:.9f}',
                'gt_time_error_sec': f'{timestamp_errors[i]:.9f}',
                'est_tx': f'{est_positions[i, 0]:.10f}',
                'est_ty': f'{est_positions[i, 1]:.10f}',
                'est_tz': f'{est_positions[i, 2]:.10f}',
                'gt_tx': f'{gt_positions[i, 0]:.10f}',
                'gt_ty': f'{gt_positions[i, 1]:.10f}',
                'gt_tz': f'{gt_positions[i, 2]:.10f}',
                'raw_error': f'{raw_errors[i]:.10f}',
                'se3_tx': f'{se3_aligned[i, 0]:.10f}',
                'se3_ty': f'{se3_aligned[i, 1]:.10f}',
                'se3_tz': f'{se3_aligned[i, 2]:.10f}',
                'se3_error': f'{se3_errors[i]:.10f}',
                'sim3_tx': f'{sim3_aligned[i, 0]:.10f}',
                'sim3_ty': f'{sim3_aligned[i, 1]:.10f}',
                'sim3_tz': f'{sim3_aligned[i, 2]:.10f}',
                'sim3_error': f'{sim3_errors[i]:.10f}',
            })

    def transform_to_json(transform):
        return {
            'scale': float(transform['scale']),
            'rotation': transform['rotation'].tolist(),
            'translation': transform['translation'].tolist(),
        }

    metrics = {
        'primary_metric': 'sim3_aligned_ate_rmse',
        'pose_source': 'poses/estimated_c2w.csv, T_world_camera / c2w in VINGS internal world/map frame',
        'gt_source': str(gt_pose_path),
        'gt_position_columns': ['t_x', 't_y', 't_z'],
        'matching': {
            'key': 'nearest gt_timestamp to GT parquet timestamp',
            'timestamp_tolerance_sec': tolerance,
            'input_keyframes': len(sorted_rows),
            'matched_keyframes': len(matched_rows),
            'unmatched_keyframes': len(unmatched),
            'mean_abs_time_error_sec': float(np.mean(timestamp_errors)) if len(timestamp_errors) else None,
            'max_abs_time_error_sec': float(np.max(timestamp_errors)) if len(timestamp_errors) else None,
        },
        'raw': _ate_stats(raw_errors),
        'se3_aligned': {
            **_ate_stats(se3_errors),
            'transform_est_to_gt': transform_to_json(se3_transform),
        },
        'sim3_aligned': {
            **_ate_stats(sim3_errors),
            'transform_est_to_gt': transform_to_json(sim3_transform),
        },
        'outputs': {
            'aligned_trajectory_csv': trajectory_csv,
        },
    }

    metrics_path = os.path.join(metrics_dir, 'ate_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    _log(logger, f"Saved ATE metrics: {metrics_path} (Sim3 RMSE={metrics['sim3_aligned']['rmse']:.6f})")


def finalize_sequence_outputs(cfg, gaussian_model, logger=print):
    output_cfg = cfg.get('output', {})
    final_rerender = bool(output_cfg.get('final_rerender', False))
    eval_render_metrics = bool(output_cfg.get('eval_render_metrics', final_rerender))
    eval_ate = bool(output_cfg.get('eval_ate', False))

    if not (final_rerender or eval_render_metrics or eval_ate):
        return
    if eval_render_metrics and not final_rerender:
        raise ValueError('eval_render_metrics=True requires final_rerender=True for the clean railway output.')

    save_dir = cfg['output']['save_dir']
    pose_csv = os.path.join(save_dir, 'poses', 'estimated_c2w.csv')
    pose_rows = _load_pose_rows(pose_csv)
    metrics_dir = os.path.join(save_dir, 'metrics')
    _ensure_clean_dir(metrics_dir)

    if final_rerender:
        rerender_final_map_and_metrics(cfg, gaussian_model, pose_rows, metrics_dir, logger=logger)
    if eval_ate:
        compute_ate_metrics(cfg, pose_rows, metrics_dir, logger=logger)
