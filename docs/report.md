# Report: CARLA scenario -> NuRec-style reconstruction -> comparison with CARLA ground truth

Date: 2026-10-28. Machine: Windows 11, RTX 4050 Laptop GPU (6 GB), 64 GB RAM. CARLA 0.10 (UE5 `ue5-dev` source
build), map `LA_DTLA_ODR` (MirrorMile Downtown LA). All numbers below are measured; the full tables are in
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
   from cuboid tracks; learned sky; per-image exposure). **30 000 iterations at 640 x 360: 54 min, 1.95 GB peak VRAM.**
   The NuRec run is packaged for a Linux GPU machine (`cloud/`).
5. **Result** (raw / exposure-aligned):

| | PSNR dB | SSIM | LPIPS |
|---|---|---|---|
| held-out on-path views (120 images, 6 cameras) | 21.52 / 22.41 | 0.852 / 0.857 | 0.230 / 0.228 |
| novel views +-1 m (front camera) | 20.82, 20.69 / 22.50, 22.59 | 0.847, 0.846 | 0.223, 0.220 |
| novel views +-2 m | 19.66, 19.58 / 20.80, 21.06 | 0.829, 0.829 | 0.244, 0.233 |
| novel views +-3 m | 18.55, 18.60 / 19.50, 19.78 | 0.808, 0.812 | 0.265, 0.249 |

   Depth vs CARLA depth on held-out views: AbsRel 0.065, median error 0.32 m, delta<1.25 0.962. Geometry vs lidar:
   median depth error at lidar points 0.26 m (72 % within 0.5 m); F-score 0.62 @ 0.2 m, 0.83 @ 0.5 m. Dynamic actors
   (positions from the boxes, shape learned): mask IoU vs CARLA instance masks 0.53-0.79 for the cars, the cyclist and
   the crossing pedestrians. Noisy tracker boxes (0.3 m / 3 deg): learned per-frame corrections reduce the position
   error from 0.374 to 0.330 m (mean) but increase the yaw error from 2.4 to 4.4 deg (below).

## Dynamic actor positions vs the ground-truth boxes

With ground-truth boxes as input (as NuRec takes tracked cuboids) the actors sit exactly where the boxes say, so the
meaningful test is a noisy tracker: every actor's box is perturbed independently per frame (sigma 0.3 m in x / y and
3 deg in yaw), the actor shapes are accumulated from lidar with those noisy boxes, and the trainer learns a correction
(translation + yaw) per actor and frame (`train.py --box-noise 0.3,3 --refine-boxes`, 15 000 iterations, 26.5 min,
2.05 GB VRAM; `eval04n/metrics.json`, `eval04n/box_per_actor.json`).

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
  coloured from unoccluded images) plus 20 k points on a far shell; MCMC densification capped at 1.2 M; 3DGUT rasterisation (gsplat `with_ut`, `with_eval3d`).
- Dynamic actors (NuRec's rule: pedestrians / cyclists always, others if they move > 1 m; 10 actors): each a rigid
  Gaussian set in its box frame, initialised from its lidar points accumulated over all frames, placed every frame by
  its 3D box (cap 30 k per car, 15 k per pedestrian / cyclist). The two cars waiting at the red light never move and
  belong to the background. Pedestrians are treated as rigid (NuRec uses a deformable layer for them).
- Sky: equirectangular map behind the Gaussians, trained with the semantic sky mask. Ego pixels (0.45 %) masked.
- Loss: 0.8 L1 + 0.2 D-SSIM, 0.1 relative lidar depth error, 0.05 sky opacity term, MCMC opacity / scale
  regularisers on the background. Per-image affine exposure (CARLA's auto exposure varies, see below).
- Split: every 10th tick (5, 15, ..., 195) held out, all six cameras (120 images); 1086 training images.

## Findings and limitations

- **Near-field road**: lane dashes and texture within ~10 m of the camera are blurred or missing in held-out and
  novel views (each dash is seen briefly at a grazing angle); this is the largest error region. Likely remedies:
  full-resolution training, more iterations, a road-surface prior / ground initialisation.
- **Exposure**: the RGB camera of this CARLA build has no exposure attributes; Unreal's automatic exposure changes
  brightness between images (same static surface: median ratio 1.12 between neighbouring cameras, 0.73-1.58 over 2 s).
  Hence the two metric columns; the model learns an exposure per training image but cannot know a test image's.
- **Off-path views** lose ~1 dB per metre of lateral offset (21.5 dB at 0 m to 18.6 dB at +-3 m). The re-simulation
  ground truth itself is consistent to 31.8 dB (offset 0 vs the live recording, ego excluded), so the drop is the
  reconstruction's.
- **Actors**: cars and the cyclist reconstruct well (IoU 0.69-0.79); the crossing pedestrians less so (0.53-0.56, rigid
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
