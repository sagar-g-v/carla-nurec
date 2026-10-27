# Running NVIDIA NuRec on this recording (not possible on the laptop)

NuRec reconstruction needs Linux x86_64, an Ampere-or-newer NVIDIA GPU with **>= 24 GB VRAM (48 GB recommended)**,
driver R550+ (R570 recommended), Docker >= 23.0.1 with the NVIDIA Container Toolkit >= 1.13.5, and an NGC API key
(free NGC account; licence: NVIDIA Software License Agreement + Omniverse product-specific terms). The laptop this
repository was made on has 6 GB and Windows, so this step was prepared but **not run**.

## What to copy to the GPU machine

`build/nurec/ncore04/` (NCore V4, written by `convert/to_ncore.py`):
- `carla_la_dtla_j10295.json` - sequence manifest
- `carla_la_dtla_j10295.ncore4.zarr.itar` - 6 cameras (201 PNG frames each), lidar `lidar_top` (201 sweeps),
  rig poses, sensor extrinsics, intrinsics, 2412 `GT_SYNTHETIC` cuboid observations (12 actors)

Then `NGC_API_KEY=... ./run_nurec.sh /path/to/ncore04 /path/to/out` pulls `nre-tools-ga:26.04` and `nre-ga:26.04`
(13.3 GB compressed), generates the auxiliary data, trains (`trainer.max_epochs`, default 30) and leaves
`out/train/<run>/usd-out/last.usdz` plus `checkpoints/last.ckpt`.

## Open points to expect

- The production config `configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml` is written for NVIDIA's Hyperion rig
  (f-theta cameras). Our rig is 6 ideal pinhole cameras (1280 x 720, 90 deg) and a 64-beam lidar; NuRec reads the
  camera model from the NCore intrinsics, but camera-id / resolution assumptions in that config may need overrides.
- Classes: our cuboids use NuRec's vocabulary (`automobile` -> `dynamic_rigids`; `pedestrian`, `cyclist` ->
  `dynamic_deformables`); the two cars waiting at the red light never move and are classified static.
- Comparing NuRec with the local reconstruction: render NuRec at the held-out ticks (5, 15, ..., 195) and at the
  novel-view poses in `build/nurec/novel04/replay.json`, then run `evaluate/evaluate.py`-style metrics against
  the same CARLA images (the novel-view ground truth comes from the re-simulation, not from CARLA's recorder).

## Alternative on the same machine

nv-tlabs/3dgrut (Apache-2.0) reads NCore V4 directly and exports NuRec-format USDZ, but reconstructs static scenes
only (no actor layers).
