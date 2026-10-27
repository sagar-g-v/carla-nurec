## Reconstruction

| | |
|---|---|
| training resolution | 640 x 360 |
| iterations | 30000 |
| wall time | 54.0 min |
| peak VRAM (torch allocated) | 1949 MB |
| Gaussians (background / actors) | 1200000 / 210000 |

## Held-out on-path views (every 10-th tick, all cameras; 120 images)

| camera | PSNR dB | SSIM | LPIPS (alex) |
|---|---|---|---|
| CAM_FRONT | 20.76 | 0.866 | 0.210 |
| CAM_FRONT_LEFT | 22.68 | 0.866 | 0.238 |
| CAM_FRONT_RIGHT | 21.25 | 0.844 | 0.227 |
| CAM_BACK_LEFT | 22.76 | 0.863 | 0.242 |
| CAM_BACK_RIGHT | 20.95 | 0.839 | 0.248 |
| CAM_BACK | 20.72 | 0.834 | 0.214 |
| **all, raw** | **21.52** | **0.852** | **0.230** |
| all, exposure-aligned | 22.41 | 0.857 | 0.228 |

Raw: the rendering as is. Exposure-aligned: a per-image affine colour fit between rendering and ground truth first (CARLA's automatic exposure differs per image; the model never saw these images' exposure).

Depth vs CARLA depth (non-sky, < 80 m, alpha > 0.5): AbsRel 0.065, RMSE 4.25 m, median |error| 0.324 m, delta<1.25 0.962.

## Novel views off the driven path (front camera shifted sideways, CARLA re-simulation)

| lateral offset (+ = right) | views | PSNR dB | SSIM | LPIPS | PSNR / SSIM / LPIPS exposure-aligned | depth AbsRel | depth median abs m |
|---|---|---|---|---|---|---|---|
| -3 m | 49 | 18.55 | 0.808 | 0.265 | 19.50 / 0.811 / 0.265 | 0.102 | 0.542 |
| -2 m | 51 | 19.66 | 0.829 | 0.244 | 20.80 / 0.839 / 0.240 | 0.094 | 0.461 |
| -1 m | 51 | 20.82 | 0.847 | 0.223 | 22.50 / 0.864 / 0.216 | 0.080 | 0.413 |
| +0 m | 51 | 21.53 | 0.858 | 0.212 | 23.64 / 0.877 / 0.205 | 0.075 | 0.402 |
| +1 m | 51 | 20.69 | 0.846 | 0.220 | 22.59 / 0.864 / 0.213 | 0.074 | 0.419 |
| +2 m | 51 | 19.58 | 0.829 | 0.233 | 21.06 / 0.845 / 0.228 | 0.076 | 0.454 |
| +3 m | 51 | 18.60 | 0.812 | 0.249 | 19.78 / 0.824 / 0.246 | 0.081 | 0.499 |

Re-simulation noise floor (re-simulated front camera at offset 0, ego parked out of view, vs the live recording, ego pixels excluded; differences come from walker animation, the missing ego shadow and exposure): PSNR 31.80 dB, SSIM 0.974, LPIPS 0.014 over 51 frames. Re-simulated actor positions vs the live run (mean / max m): lead 0.000/0.000, cyclist 0.000/0.000, oncoming_lt 0.000/0.000, oncoming_rt 0.000/0.000, oncoming_stop 0.000/0.000, ped_far_0 0.000/0.000, ped_far_1 0.000/0.000, ped_left_0 0.000/0.000, ped_left_1 0.000/0.000, ped_right_0 0.000/0.000, wait_0 0.000/0.000, wait_1 0.000/0.000, cyclist_rider 0.000/0.000.

Skipped 2 novel views whose camera was inside an actor box: frame 172 -3 m (oncoming_stop), frame 176 -3 m (oncoming_stop).

## Geometry vs lidar (held-out ticks, within 50 m)

| | |
|---|---|
| rendered depth at lidar points: median abs error | 0.259 m |
| ... within 0.5 m / 1 m | 0.724 / 0.884 |
| chamfer: accuracy (recon -> lidar) mean / median | 0.814 / 0.229 m |
| chamfer: completeness (lidar -> recon) mean / median | 0.063 / 0.041 m |
| F-score @ 0.2 m / 0.5 m | 0.621 / 0.835 |

## Dynamic actors

| actor | mask IoU (views) | Gaussian extent l x w x h (m) | ground-truth box (m) | centre offset (m) |
|---|---|---|---|---|
| cyclist | 0.784 (7) | 0.92 x 0.64 x 1.42 | 1.47 x 0.84 x 1.48 | 0.204 |
| lead | 0.689 (9) | 1.40 x 1.84 x 1.15 | 4.18 x 1.99 x 1.39 | 1.518 |
| oncoming_lt | 0.791 (14) | 5.31 x 2.16 x 1.96 | 5.59 x 2.15 x 2.06 | 0.042 |
| oncoming_rt | 0.764 (13) | 5.09 x 1.79 x 1.29 | 5.36 x 2.03 x 1.41 | 0.082 |
| oncoming_stop | 0.596 (11) | 5.15 x 1.71 x 1.61 | 5.35 x 1.79 x 1.57 | 0.209 |
| ped_far_0 | 0.561 (8) | 1.46 x 1.53 x 2.77 | 0.33 x 0.87 x 1.80 | 0.748 |
| ped_far_1 | 0.529 (5) | 1.79 x 1.95 x 2.84 | 0.40 x 0.70 x 1.93 | 0.672 |
| ped_left_0 | 0.000 (2) | 37.97 x 16.01 x 6.53 | 0.37 x 0.72 x 1.83 | 5.122 |
| ped_left_1 | 0.012 (1) | 8.86 x 26.06 x 23.30 | 0.42 x 0.78 x 1.95 | 14.004 |
| ped_right_0 | - | 0.88 x 0.68 x 2.14 | 0.43 x 0.70 x 1.84 | 0.530 |

actor positions come from the ground-truth boxes (as NuRec's cuboid tracks); IoU measures how well the reconstructed actor shape covers CARLA's instance mask, with occlusion

CARLA recorder replay (not used for ground truth): best alignment replay tick = frame + 32 (lead max error 1.7949 m); replayed actor position error vs the live run (mean / max m): ego 0.73/1.80, lead 0.90/1.79, oncoming_lt 0.67/1.48, oncoming_rt 0.62/1.38, oncoming_stop 0.47/1.61, ped_far_0 0.01/0.01, ped_far_1 0.01/0.01, ped_left_0 0.01/0.01, ped_left_1 0.01/0.01, ped_right_0 0.01/0.21, wait_0 0.00/0.02, wait_1 0.00/0.04, cyclist_rider 0.50/1.00.

## Exposure check

Median brightness ratio of the same static surfaces in two images (1.00 = same exposure): neighbour same tick 1.121 (5-95 %: 0.990-1.253, 66 pairs); same cam 5 ticks 1.001 (5-95 %: 0.870-1.263, 60 pairs); same cam 20 ticks 1.046 (5-95 %: 0.729-1.580, 60 pairs).

## Pictures

**novel_0040_off+1** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0040_off+1.jpg](img/novel_0040_off+1.jpg)

**novel_0040_off+3** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0040_off+3.jpg](img/novel_0040_off+3.jpg)

**novel_0040_off-1** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0040_off-1.jpg](img/novel_0040_off-1.jpg)

**novel_0040_off-3** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0040_off-3.jpg](img/novel_0040_off-3.jpg)

**novel_0100_off+1** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0100_off+1.jpg](img/novel_0100_off+1.jpg)

**novel_0100_off+3** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0100_off+3.jpg](img/novel_0100_off+3.jpg)

**novel_0100_off-1** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0100_off-1.jpg](img/novel_0100_off-1.jpg)

**novel_0100_off-3** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0100_off-3.jpg](img/novel_0100_off-3.jpg)

**novel_0160_off+1** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0160_off+1.jpg](img/novel_0160_off+1.jpg)

**novel_0160_off+3** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0160_off+3.jpg](img/novel_0160_off+3.jpg)

**novel_0160_off-1** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0160_off-1.jpg](img/novel_0160_off-1.jpg)

**novel_0160_off-3** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![novel_0160_off-3.jpg](img/novel_0160_off-3.jpg)

**test_0005_CAM_FRONT** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![test_0005_CAM_FRONT.jpg](img/test_0005_CAM_FRONT.jpg)

**test_0015_CAM_FRONT** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![test_0015_CAM_FRONT.jpg](img/test_0015_CAM_FRONT.jpg)

**test_0025_CAM_FRONT** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![test_0025_CAM_FRONT.jpg](img/test_0025_CAM_FRONT.jpg)

**test_0035_CAM_FRONT** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![test_0035_CAM_FRONT.jpg](img/test_0035_CAM_FRONT.jpg)

**test_0045_CAM_FRONT** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![test_0045_CAM_FRONT.jpg](img/test_0045_CAM_FRONT.jpg)

**test_0055_CAM_FRONT** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)

![test_0055_CAM_FRONT.jpg](img/test_0055_CAM_FRONT.jpg)
