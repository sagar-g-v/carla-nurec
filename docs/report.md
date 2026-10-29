# Report: CARLA scenario -> NuRec-style reconstruction -> comparison with CARLA ground truth

Date: 2026-10-28, updated 2026-10-30 (retrained after fixing the MCMC noise scale, see
[Before and after the fix](#before-and-after-the-mcmc-noise-fix)). Machine: Windows 11, RTX 4050 Laptop GPU (6 GB),
64 GB RAM. CARLA 0.10 (UE5 `ue5-dev` source build), map `LA_DTLA_ODR` (MirrorMile Downtown LA). All numbers below are
measured and belong to the retrained model `train05b` unless marked otherwise; the full tables are in
[results.md](results.md) (generated from `metrics.json` by `evaluate/report.py`); the raw metric files are in
[../results/](../results/).

## Summary

1. **Scenario** ([scenario.md](scenario.md)): 20 s, 10 Hz, at junction 10295 near the audit spot (x 397.65,
   y 528.80). Ego straight through from lane -3, a lead car, three oncoming cars (one turns left across the ego's path,
   one turns right, one stops for pedestrians), a cyclist running the cross street's red light across the ego's path,
   five pedestrians (two cross in front of the ego), two cars waiting at the red light. Deterministic: a second run
   reproduces every actor to 0.1 mm.
2. **Recording**: 6 cameras (RGB, depth, instance segmentation, 1280 x 720, 90 deg), 64-beam lidar + semantic lidar,
   every actor's 3D box / pose / class / id per frame, the OpenDRIVE, CARLA's recorder file; novel-view ground truth at
   -3 .. +3 m beside the driven path. 9.5 min of CARLA time for both.
3. **Formats**: NCore V4 (NuRec's input, checked by reading it back with NCore's loader and drawing the cuboids into
   the image) and a training dataset ([formats.md](formats.md)).
4. **NuRec itself cannot run here** (Linux, >= 24 GB VRAM). Reconstruction on the laptop: a composite 3DGUT model on
   gsplat (static background + one rigid Gaussian set per dynamic actor placed by its 3D box, as NuRec separates them
   from cuboid tracks; learned sky; per-image exposure). **30 000 iterations at 640 x 360: 47.6 min, 1.96 GB peak
   VRAM.** The NuRec run is packaged for a Linux GPU machine (`cloud/`).
5. **Result** (raw / exposure-aligned; novel views: -m / +m):

| | PSNR dB | SSIM | LPIPS |
|---|---|---|---|
| held-out on-path views (120 images, 6 cameras) | 22.45 / 23.38 | 0.883 / 0.888 | 0.169 / 0.166 |
| novel views +-1 m (front camera) | 21.51, 21.53 / 23.11, 23.27 | 0.876, 0.876 | 0.165, 0.161 |
| novel views +-2 m | 20.21, 20.17 / 21.29, 21.46 | 0.855, 0.857 | 0.189, 0.179 |
| novel views +-3 m | 19.11, 19.11 / 20.00, 20.09 | 0.834, 0.837 | 0.210, 0.198 |

   Depth vs CARLA depth on held-out views: AbsRel 0.059, median error 0.24 m, delta<1.25 0.971. Geometry vs lidar:
   median depth error at lidar points 0.20 m (80 % within 0.5 m); F-score 0.70 @ 0.2 m, 0.89 @ 0.5 m. Dynamic actors
   (positions from the boxes, shape learned): mask IoU vs CARLA instance masks 0.56-0.80 for the cars, the cyclist and
   the crossing pedestrians. Noisy tracker boxes (0.3 m / 3 deg, run with the trainer before the fix): learned
   per-frame corrections reduce the position error from 0.374 to 0.330 m (mean) but increase the yaw error from 2.4
   to 4.4 deg (below).

## Dynamic actor positions vs the ground-truth boxes

With ground-truth boxes as input (as NuRec takes tracked cuboids) the actors sit exactly where the boxes say, so the
meaningful test is a noisy tracker: every actor's box is perturbed independently per frame (sigma 0.3 m in x / y and
3 deg in yaw), the actor shapes are accumulated from lidar with those noisy boxes, and the trainer learns a correction
(translation + yaw) per actor and frame (`train.py --box-noise 0.3,3 --refine-boxes`, 15 000 iterations, 26.5 min,
2.05 GB VRAM; `eval04n/metrics.json`, `eval04n/box_per_actor.json`). This experiment ran with the trainer before
the MCMC noise fix (the same flaw as `train04b`) and has not been repeated; its image numbers compare with
`train04b`, not with the retrained model.

| boxes vs CARLA ground truth (training ticks, 1810 box-frames) | position error mean / median | yaw error mean |
|---|---|---|
| given (noisy) | 0.374 m / 0.353 m | 2.42 deg |
| refined | 0.330 m / 0.292 m | 4.38 deg |

Per actor the position error drops for every actor the model reconstructs (best: the left-turning SUV
0.385 -> 0.285 m; the other cars and the cyclist 0.35-0.39 -> 0.30-0.32 m; pedestrians 0.38-0.39 -> 0.33-0.35 m) and does not
for the two left-crosswalk pedestrians (not reconstructed, see below). The yaw gets **worse** for every actor, least
for the SUV (2.57 -> 2.66 deg): the learned yaw corrections average 3.8 deg, larger than the noise. So this simple per-frame
refinement (no temporal smoothness, Adam momentum on rarely sampled frames, shape learned from the blurred
accumulation) recovers 8-26 % of the position noise (per actor) and none of the heading noise. With the noisy boxes the images
also suffer: held-out 21.01 dB raw / 21.80 dB aligned after 15 000 iterations, actor mask IoU 0.42-0.68 for the cars, 0.27 for the
cyclist, 0.06-0.19 for the crossing pedestrians, 0 for the left-crosswalk pair. Held-out ticks keep their noisy boxes (no image constrains them).

## Training: do we need it?

Yes. NuRec / 3DGUT reconstruct a scene by fitting a new set of 3D Gaussians to that scene's images, lidar and poses;
that per-scene optimisation is the "training". The pretrained NuRec scenes NVIDIA publishes are reconstructions of
NVIDIA's own drives and cannot render our CARLA scene. Feed-forward models exist (NVIDIA's Instant NuRec, DGGT,
STORM, ...): one network pass, no per-scene fitting, but low resolution (240 x 160 to 504 x 280), short windows,
~26-28 dB on real driving data vs ~29-33 dB for dense per-scene fitting at 640 x 960, none evaluated on CARLA or off
the driven path, and none runnable on a 6 GB laptop (Instant NuRec states >= 30 GB). Details: [notes.md](notes.md).

## How the reconstruction works

- Background: Gaussians initialised from a random 300 k subset of the 2.1 M static lidar points (10 cm voxels,
  coloured from unoccluded images) plus 20 k points on a far shell; MCMC densification capped at 1.2 M, its position
  noise scaled as for a scene of unit size (below); 3DGUT rasterisation (gsplat `with_ut`, `with_eval3d`).
- Dynamic actors (NuRec's rule: pedestrians / cyclists always, others if they move > 1 m; 10 actors): each a rigid
  Gaussian set in its box frame, initialised from its lidar points accumulated over all frames, placed every frame by
  its 3D box (cap 30 k per car, 15 k per pedestrian / cyclist). The two cars waiting at the red light never move and
  belong to the background. Pedestrians are treated as rigid (NuRec uses a deformable layer for them).
- Sky: equirectangular map behind the Gaussians, trained with the semantic sky mask. Ego pixels (0.45 %) masked.
- Loss: 0.8 L1 + 0.2 D-SSIM, 0.1 relative lidar depth error, 0.05 sky opacity term, MCMC opacity / scale
  regularisers on the background. Per-image affine exposure (CARLA's auto exposure varies, see below).
- Split: every 10th tick (5, 15, ..., 195) held out, all six cameras (120 images); 1086 training images.

## Before and after the MCMC noise fix

The first model (`train04b`, 2026-10-28) had a flaw in my training setup, found on 2026-10-30 while exporting it for
the 3D viewer: 1.10 M of its 1.41 M Gaussians were background Gaussians more than 300 m from the scene centre with
opacity below 0.05 (median distance of all background Gaussians 3.9 km, the farthest 2.3e8 m). Removing them left
that model's own render unchanged (86-107 dB against the full model at the front camera, ticks 0 / 95 / 150), so it
was fitted with about 0.3 M working Gaussians. Cause: gsplat's MCMC strategy uses the position learning rate only to
size its position noise (covariance x lr x 5e5); `train.py` scales that learning rate by the scene size in metres
(~100 m for the background, 5 m for the actors) where gsplat expects scenes normalised to about unit size, so the
noise was size^2 (~1e4 for the background) too large. Faint Gaussians random-walked away and stayed above the
relocation threshold (opacity 0.005), so they were never recycled; the "alive" count in the training log hid it.

Fix: MCMC now gets the learning rate divided by the group size squared (`train.py --noise-unnormalised` restores
the old behaviour), and the log records the background's median distance and how many Gaussians are beyond 300 m.
Retrained with otherwise identical settings (`train05b`): median background distance 61 m (was 3925 m), 0.90 M
background Gaussians within 100 m (was 0.06 M), 67 k faint ones beyond 300 m (was 1.10 M), 0.28 M with opacity > 0.1
(was 0.06 M). Evaluated exactly as before (`evaluate/compare.py eval04b eval05b`):

| | train04b (before) | train05b (after) |
|---|---|---|
| training wall time / peak VRAM | 54.0 min / 1949 MB | 47.6 min / 1959 MB |
| held-out PSNR raw / exposure-aligned | 21.52 / 22.41 dB | 22.45 / 23.38 dB |
| held-out SSIM / LPIPS (raw) | 0.852 / 0.230 | 0.883 / 0.169 |
| novel PSNR raw at -3 / -2 / -1 / +1 / +2 / +3 m | 18.55 / 19.66 / 20.82 / 20.69 / 19.58 / 18.60 dB | 19.11 / 20.21 / 21.51 / 21.53 / 20.17 / 19.11 dB |
| depth AbsRel / median error / RMSE | 0.065 / 0.324 m / 4.25 m | 0.059 / 0.241 m / 5.13 m |
| lidar depth error median / F-score @ 0.2 m / @ 0.5 m | 0.259 m / 0.621 / 0.835 | 0.196 m / 0.696 / 0.886 |
| actor mask IoU: lead, oncoming_stop, ped_far_1 | 0.689, 0.596, 0.529 | 0.774, 0.668, 0.569 |
| actor mask IoU: oncoming_rt, oncoming_lt, cyclist, ped_far_0 | 0.764, 0.791, 0.784, 0.561 | 0.768, 0.795, 0.782, 0.564 |
| actor mask IoU: ped_left_0, ped_left_1 | 0.000, 0.012 | 0.000, 0.010 |

Every image and lidar measure improves; the depth RMSE gets worse (a few large errors, the median improves); the
cyclist and the two left-crosswalk pedestrians do not change. One training run each, so run-to-run variation is not
measured.

## Findings and limitations

- **Near-field road**: lane dashes and texture within ~10 m of the camera are blurred or missing in held-out and
  novel views (each dash is seen briefly at a grazing angle); this is still the largest error region after the fix.
  Likely remedies: full-resolution training, more iterations, a road-surface prior / ground initialisation.
- **Exposure**: the RGB camera of this CARLA build has no exposure attributes; Unreal's automatic exposure changes
  brightness between images (same static surface: median ratio 1.12 between neighbouring cameras, 0.73-1.58 over 2 s).
  Hence the two metric columns; the model learns an exposure per training image but cannot know a test image's.
- **Off-path views** lose ~1.1 dB per metre of lateral offset (22.6 dB at 0 m to 19.1 dB at +-3 m). The re-simulation
  ground truth itself is consistent to 31.8 dB (offset 0 vs the live recording, ego excluded), so the drop is the
  reconstruction's.
- **Actors**: cars and the cyclist reconstruct well (IoU 0.67-0.80); the crossing pedestrians less so (0.56-0.57, rigid
  model of an animated body); the two left-crosswalk pedestrians fail (IoU ~0 in the 1-2 views where they are big
  enough): they stand still against a shaded facade for 7.6 s (the background absorbs them) and the ego's lidar saw
  only ~800 of their points. The "Gaussian extent" column is only a rough check (the lead car is only ever seen from
  behind; pedestrians carry low-opacity floaters).
- **Offset 0 novel views** are training camera poses (every 4th tick never hits a held-out tick) rendered against an
  ego-free re-simulation; the held-out row is the on-path generalisation number.

## Mistakes made on the way (and fixed)

- First recording (`rec01`): the bike mesh never moved (`static.prop.mesh` ignores `set_transform` unless spawned with
  mass), pedestrians walked at 5 % speed (CARLA ue5-dev `WalkerController` bug, compensated), and ground heights fell
  back to the OpenDRIVE height 1604 times (projections hit the vehicles' own roofs). Found while checking the data,
  fixed, re-recorded (`rec04` is the one evaluated).
- First training runs: plain Adam turned the MCMC regularisers into full-size steps for Gaussians outside the current
  view, so actors faded and the background plateaued at 17.7 dB train PSNR; then the regulariser alone still faded
  far / occluded actors. Fixed with visibility-masked Adam (gsplat `SelectiveAdam`) and regularisers on the background
  only.
- First evaluation dropped all +-1 m / 0 m novel views (the ego's box counted as an obstacle although the ego is
  parked away in the re-simulation); fixed and re-run.
- The CARLA recorder replay was planned as the novel-view source; it replays kinematic actors 1-2 ticks late (up to
  1.8 m, measured on the first recording), so a re-simulation is used instead.
- The MCMC position noise was scaled for metres instead of a unit-size scene, so most of the first model's Gaussians
  drifted away unused (found 2026-10-30); fixed and retrained, see
  [Before and after the fix](#before-and-after-the-mcmc-noise-fix).
