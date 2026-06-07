# VINGS-Mono 铁路数据集批跑结果报告

- 报告生成时间：`2026-06-07 09:41:32`
- 日志目录：`/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613`
- 输出目录：`/home/leizongru/lzr_ws/VINGS-Mono/output`
- 数据集目录：`/home/leizongru/lzr_ws/railway_data`
- 使用显卡：`GPU 2`
- 开始时间：`2026-06-06 16:36:13`
- 结束时间：`2026-06-06 18:28:24`
- 总耗时：`1小时 52分钟 11秒`
- 批跑汇总：`ok=7 fail=0`
- 最终状态：`成功`

## 总体结论

七个铁路序列全部正常跑完，退出码均为 `0`。每个序列都生成了最终 Gaussian map 重渲图、渲染指标、ATE 指标、位姿 CSV/TUM 文件和 PLY 地图文件。

日志中没有发现 `Traceback`、`RuntimeError`、CUDA OOM、非零退出码等致命错误。出现过的 MMCV 兼容性提示、PyTorch `meshgrid` 提示、ONNX Runtime CUDA 图相关提示、`xFormers not available` 等均为非致命 warning，没有中断运行。

## 汇总指标

- 七个序列 Sim(3) ATE RMSE 均值：`1.571238`
- 最好 Sim(3) ATE RMSE：`scene_11_train` = `0.654872`
- 最差 Sim(3) ATE RMSE：`scene_05_train` = `3.400496`
- 各序列 PSNR 均值的平均值：`16.4700`
- 各序列 SSIM 均值的平均值：`0.4921`
- 各序列 LPIPS 均值的平均值：`0.5113`

## 序列结果汇总

| 序列 | 状态 | 耗时 | 输入帧数 | 关键帧数 | 重渲图数 | 图像分辨率 | PSNR均值 | SSIM均值 | LPIPS均值 | ATE Sim3 RMSE | ATE匹配 | 最大时间误差(s) | 一致性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene_05_train` | 成功 | 31分钟 50秒 | 259 | 120 | 120 | 2504x4112 | 16.6684 | 0.5443 | 0.4877 | 3.400496 | 120/120 | 0.000000238 | 通过 |
| `scene_11_train` | 成功 | 10分钟 24秒 | 288 | 29 | 29 | 2504x4112 | 17.9285 | 0.5539 | 0.5310 | 0.654872 | 29/29 | 0.000000238 | 通过 |
| `scene_13_train` | 成功 | 10分钟 14秒 | 149 | 46 | 46 | 2504x4112 | 15.2229 | 0.4535 | 0.5441 | 1.298138 | 46/46 | 0.000000238 | 通过 |
| `scene_14_train` | 成功 | 14分钟 37秒 | 298 | 63 | 63 | 2504x4112 | 19.5963 | 0.6289 | 0.4189 | 1.249974 | 63/63 | 0.000000238 | 通过 |
| `scene_16_train` | 成功 | 8分钟 55秒 | 179 | 37 | 37 | 2504x4112 | 14.6349 | 0.4204 | 0.5295 | 1.120321 | 37/37 | 0.000000238 | 通过 |
| `scene_17_train` | 成功 | 21分钟 24秒 | 235 | 109 | 109 | 2504x4112 | 14.1492 | 0.4794 | 0.4771 | 1.621948 | 109/109 | 0.000000238 | 通过 |
| `scene_19_train` | 成功 | 14分钟 46秒 | 289 | 57 | 57 | 2504x4112 | 17.0898 | 0.3645 | 0.5907 | 1.652920 | 57/57 | 0.000000238 | 通过 |

## ATE 详细结果

轨迹主指标采用 Sim(3) 对齐后的 ATE RMSE，适合单目 VO。Raw ATE 和 SE(3) 对齐 ATE 只作为参考。

| 序列 | Raw RMSE | SE(3) RMSE | Sim(3) RMSE | 未匹配关键帧 |
|---|---:|---:|---:|---:|
| `scene_05_train` | 357.703391 | 113.510470 | 3.400496 | 0 |
| `scene_11_train` | 67.287344 | 16.308517 | 0.654872 | 0 |
| `scene_13_train` | 150.868761 | 41.852625 | 1.298138 | 0 |
| `scene_14_train` | 167.890727 | 46.774335 | 1.249974 | 0 |
| `scene_16_train` | 102.081820 | 29.916905 | 1.120321 | 0 |
| `scene_17_train` | 311.869177 | 93.778639 | 1.621948 | 0 |
| `scene_19_train` | 141.355965 | 42.297945 | 1.652920 | 0 |

## 输出文件检查

本次检查口径：只保留最终 Gaussian map 对关键帧 pose 的统一重渲图；渲染图为 `2504 x 4112`；每张重渲图在 pose CSV 中有一条对应位姿；ATE 通过 `gt_timestamp` 与 GT parquet 最近邻匹配。

| 序列 | Render PNG | 渲染指标行数 | Pose CSV行数 | 单帧Pose文件 | 指标文件数 | PLY文件 | 输出目录 | 日志文件 |
|---|---:|---:|---:|---:|---:|---|---|---|
| `scene_05_train` | 120 | 120 | 120 | 120 | 4 | `idx=258_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_05_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_05_train_gpu2.log` |
| `scene_11_train` | 29 | 29 | 29 | 29 | 4 | `idx=287_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_11_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_11_train_gpu2.log` |
| `scene_13_train` | 46 | 46 | 46 | 46 | 4 | `idx=148_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_13_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_13_train_gpu2.log` |
| `scene_14_train` | 63 | 63 | 63 | 63 | 4 | `idx=297_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_14_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_14_train_gpu2.log` |
| `scene_16_train` | 37 | 37 | 37 | 37 | 4 | `idx=178_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_16_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_16_train_gpu2.log` |
| `scene_17_train` | 109 | 109 | 109 | 109 | 4 | `idx=234_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_17_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_17_train_gpu2.log` |
| `scene_19_train` | 57 | 57 | 57 | 57 | 4 | `idx=288_2dgs.ply` | `/home/leizongru/lzr_ws/VINGS-Mono/output/scene_19_train` | `/home/leizongru/lzr_ws/VINGS-Mono/logs/railway_7seq_20260606_163613/scene_19_train_gpu2.log` |

## 错误扫描

未发现致命错误。日志中的 `fail=0` 和 `0 failure(s)` 是成功汇总信息，不代表失败。

## 说明

- 渲染指标只针对最终 Gaussian map 重渲结果计算。
- PSNR 和 SSIM 在 GT/输出分辨率 `2504 x 4112` 上计算。
- LPIPS 先将重渲图和 GT 图都缩放到宽度 `1024`，保持宽高比后计算。
- 估计位姿为 `T_world_camera / c2w`，坐标系是 VINGS 内部 VO 地图坐标系。
- `poses/estimated_c2w.csv` 中保存了 `gt_frame_id`、`gt_image_name`、`gt_image_path`、`gt_timestamp`，便于后续复核 ATE 对齐。
