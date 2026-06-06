# Railway VO Output Save Conventions

Last updated: 2026-06-06

This document records the agreed output convention for running VINGS-Mono on the seven railway monocular sequences under `/home/leizongru/lzr_ws/railway_data`.

## 1. Output Root

All run outputs are saved under:

```text
/home/leizongru/lzr_ws/VINGS-Mono/output/
```

Each sequence uses one fixed directory named by the sequence name:

```text
/home/leizongru/lzr_ws/VINGS-Mono/output/<sequence_name>/
```

Only this convention document is intended to be tracked under `output/`. Sequence result directories are run artifacts and should remain untracked.

Example:

```text
/home/leizongru/lzr_ws/VINGS-Mono/output/scene_14_train/
```

At the start of a run, if the target sequence directory already exists, it is deleted first. Only the newest result for that sequence is kept.

The sequence name is inferred from `dataset.root` by default. If `--prefix` is passed, the basename of `--prefix` is used as the output sequence name.

## 2. Expected Directory Layout

A normal railway VO run writes:

```text
output/<sequence_name>/
├── config.yaml
├── renders/
│   └── <gt_frame_id>.png
├── poses/
│   ├── <gt_frame_id>.txt
│   ├── estimated_c2w.csv
│   ├── estimated_c2w_tum.txt
│   └── README.md
├── metrics/
│   ├── render_metrics.csv
│   ├── render_metrics_summary.json
│   ├── ate_metrics.json
│   └── ate_aligned_trajectory.csv
└── ply/
    ├── idx=<last_dataset_index>_2dgs.ply
    └── intrinsic.yaml
```

For the current railway config, legacy debug-style outputs are disabled:

```yaml
output:
  save_rgbdnua: False
  save_legacy_outputs: False
  final_rerender: True
  save_online_renders: False
  eval_render_metrics: True
  eval_ate: True
```

So `rgbdnua/`, `droid_c2w/`, and `keyframelist.txt` are not expected for the clean railway VO output.

## 3. Keyframe Render Images

Render images are saved under:

```text
output/<sequence_name>/renders/
```

Only VINGS-selected keyframes are saved, not every input frame.

During online mapping, intermediate keyframe renders are not kept. After the sequence finishes and the final Gaussian map has been built, VINGS-Mono clears `renders/` and rerenders every saved keyframe pose using the final Gaussian map. Therefore `renders/` contains only final-map rerendered RGB images.

Each render is a single RGB image, not the old multi-panel visualization. The final-map rendered RGB is produced at the current VINGS frontend resolution and then upsampled to the original GT image resolution.

Current railway GT/render output resolution:

```text
height = 2504
width  = 4112
```

The render filename is derived from the corresponding GT image filename prefix before the first underscore.

Example:

```text
GT image:
/home/leizongru/lzr_ws/railway_data/scene_05_train/048_1637935055.000000000.png

Saved render:
/home/leizongru/lzr_ws/VINGS-Mono/output/scene_05_train/renders/048.png
```

## 4. Estimated Pose Files

Pose files are saved under:

```text
output/<sequence_name>/poses/
```

For each saved keyframe, a per-frame pose matrix is saved as:

```text
poses/<gt_frame_id>.txt
```

Example:

```text
poses/048.txt
```

Each `.txt` file stores one `4 x 4` homogeneous matrix.

The pose convention is:

```text
T_world_camera / c2w
```

Meaning: the matrix transforms homogeneous points from the camera frame to the VINGS internal world/map frame.

Coordinate frames:

```text
camera frame: +x right, +y down, +z forward
world/map frame: initialized by VINGS-Mono during VO; not GPS, ENU, latitude/longitude, or the OSDaR global pose frame
```

## 5. Pose CSV For ATE Evaluation

The main trajectory file is:

```text
poses/estimated_c2w.csv
```

Each row corresponds to one saved keyframe pose and includes explicit GT association fields to avoid ATE matching mistakes.

Important columns:

```text
gt_frame_index   zero-based index in the sorted image list loaded by the dataset
gt_frame_id      prefix before the first underscore in the GT image filename, e.g. 048
gt_image_name    original GT image filename, e.g. 048_1637935055.000000000.png
gt_image_path    full path to the corresponding GT image
gt_timestamp     timestamp parsed from the GT image filename and used for matching gt_poses/*.parquet
render_name      saved render filename, e.g. 048.png
pose_file        saved 4x4 pose matrix filename, e.g. 048.txt
```

Pose values in the CSV:

```text
tx ty tz          translation from T_world_camera
qx qy qz qw       quaternion from T_world_camera rotation
m00 ... m33       flattened 4x4 T_world_camera matrix, row-major
```

Recommended ATE matching key:

```text
gt_timestamp
```

Use `gt_timestamp` to match the corresponding row in:

```text
/home/leizongru/lzr_ws/railway_data/gt_poses/<sequence_name>.parquet
```

`gt_frame_index`, `gt_frame_id`, and `gt_image_name` are also saved for manual checking and debugging.

## 6. TUM-Style Trajectory

A TUM-style trajectory is also saved:

```text
poses/estimated_c2w_tum.txt
```

Format:

```text
gt_timestamp tx ty tz qx qy qz qw
```

This file uses the same `T_world_camera / c2w` convention as the CSV and per-frame `.txt` pose files.

## 7. Rendering Metrics

Rendering metrics are saved under:

```text
output/<sequence_name>/metrics/
```

Files:

```text
render_metrics.csv
render_metrics_summary.json
```

`render_metrics.csv` contains one row per final-map rerendered keyframe. Important columns:

```text
gt_frame_id      GT frame id used in render naming
gt_image_path    GT RGB image used as reference
render_name      final-map render image filename
psnr             PSNR on the GT-resolution RGB image
ssim             SSIM on the GT-resolution RGB image
lpips            LPIPS on a resized copy of the RGB image
```

Metric convention:

```text
PSNR/SSIM: computed between the saved final render and the GT RGB image at 2504 x 4112.
LPIPS: computed after resizing both images to width 1024 while preserving aspect ratio.
```

## 8. ATE Metrics

ATE results are saved under:

```text
output/<sequence_name>/metrics/
```

Files:

```text
ate_metrics.json
ate_aligned_trajectory.csv
```

ATE convention:

```text
estimated trajectory: translation from poses/estimated_c2w.csv, T_world_camera / c2w in the VINGS internal world/map frame
GT trajectory: t_x, t_y, t_z from /home/leizongru/lzr_ws/railway_data/gt_poses/<sequence_name>.parquet
matching key: nearest gt_timestamp within the configured timestamp tolerance
primary metric: Sim(3) aligned ATE RMSE
```

`ate_metrics.json` also stores raw and SE(3)-aligned ATE for reference. For monocular VO, use the Sim(3)-aligned RMSE as the primary value.

## 9. Map Output

At the end of the sequence, VINGS-Mono saves the Gaussian map under:

```text
output/<sequence_name>/ply/
```

Expected files:

```text
idx=<last_dataset_index>_2dgs.ply
intrinsic.yaml
```

For example, if `scene_14_train` has 298 images, the final PLY is expected to be named approximately:

```text
idx=297_2dgs.ply
```

## 10. Important Notes

- Render images and estimated poses are saved for keyframes selected by VINGS, not for every dataset frame.
- Render images are final-map rerendered outputs, then upsampled to the GT resolution; they are not true full-resolution rasterization outputs.
- The estimated pose world frame is VINGS internal VO world/map frame. ATE is reported after alignment to GT, with Sim(3) aligned RMSE as the primary monocular VO metric.
- The GT association fields in `estimated_c2w.csv` are the authoritative bridge from estimated keyframe outputs back to the railway GT data.
