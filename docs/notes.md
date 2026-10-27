# Research notes and decisions (2026-10-28)

Web research only (docs, READMEs, papers); nothing in this section was installed or run unless stated.
Claims marked *unverified* could not be confirmed from a primary source.

## NVIDIA Omniverse NuRec

- A suite of 3D Gaussian reconstruction / rendering tools: NCore data format and converters, `nre-tools`
  (auxiliary data: Mask2Former segmentation, lidar segmentation / visibility, optional depth / DINOv2 features,
  ego mask), `nre` (training, gRPC rendering, USDZ export), optional Instant NuRec seeding, Harmonizer, Asset
  Harvester; integrations with AlpaSim, CARLA (0.9.16, UE4) and Isaac Sim. General availability announced at GTC 2026.
  <https://docs.nvidia.com/nurec/av/index.html>, <https://docs.nvidia.com/nurec/release-notes/index.html>
- Containers on NGC: `nvcr.io/nvidia/nre/nre-ga:26.04` (training, export, gRPC render server; 13.3 GB compressed)
  and `nvcr.io/nvidia/nre/nre-tools-ga:26.04` (auxiliary data). Needs a (free) NGC account and API key.
  Licence: NVIDIA Software License Agreement + Product-Specific Terms for Omniverse.
- Hardware (NuRec docs): **Linux x86_64 only**, Ampere / Ada / Hopper / Blackwell, driver R550+, Docker 23.0.1+,
  NVIDIA Container Toolkit 1.13.5+, **>= 24 GB VRAM (48 GB recommended)**, `--shm-size=64g` for training.
  <https://docs.nvidia.com/nurec/basics/hardware.html>
  -> **Cannot run on this laptop** (Windows 11, RTX 4050 Laptop 6 GB). Windows / WSL2 are not documented.
- Input format: **NCore V4** (zarr component stores + a sequence `.json` manifest). Library `nvidia-ncore`
  (Apache-2.0). Rig frame x forward / y left / z up; camera frame +z forward, x right, y down; `T_a_b` maps a to b;
  static `T_sensor_rig`, dynamic rig->world poses with `timestamps_us`; camera models `ideal-pinhole`,
  `opencv-pinhole`, `opencv-fisheye`, `ftheta`, global or rolling shutter; lidar `row-offset-spinning`; dynamic actors
  as `cuboid_track_observations` (track id, class id, timestamp, reference frame, bbox3, source - `GT_SYNTHETIC`
  exists for simulator ground truth). <https://nvidia.github.io/ncore/>
- Static / dynamic split: NuRec classifies tracks (pedestrians, bicycles always dynamic; others if they move > 1 m),
  routes dynamic rigid classes (car, vehicle, bus ...) to a `dynamic_rigids` Gaussian layer and pedestrians /
  cyclists to `dynamic_deformables`; dynamic tracks with no matching layer are cut out of the background.
  <https://docs.nvidia.com/nurec/nurec/dynamic-track-classification.html>
- Output: `usd-out/last.usdz` (Gaussians, the OpenDRIVE map, cuboid tracks, rig trajectories) + checkpoint.
- CARLA integration: CARLA 0.9.16 (UE4) `PythonAPI/examples/nvidia/nurec/` replays a NuRec USDZ through the gRPC
  render container `carlasimulator/nvidia-nurec-grpc:0.2.0`; not present on `ue5-dev` (our CARLA 0.10 build).
- Pretrained NuRec scenes (Hugging Face `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`): ~20 s clips of NVIDIA's own
  drives (Hyperion rig), gated licence (internal AV development, no redistribution, 12-month term). They are
  reconstructions of *those* drives - they cannot render our CARLA scene.

## Do we need to train? (per-scene fitting vs a pretrained model)

NuRec / 3DGUT reconstruct a scene by **fitting** a new set of 3D Gaussians to that scene's images, lidar and poses
(minutes to an hour of GPU optimisation per scene). That fitting *is* the "training"; there is no pretrained network
that renders an arbitrary new scene. A "trained model" exists only for scenes already reconstructed (NVIDIA's drives).

Feed-forward reconstruction (one network pass, no per-scene optimisation) exists. Reported numbers, mostly at low
resolution and on real driving data:

| method | input | moving actors | weights / licence | reported quality | 6 GB laptop |
|---|---|---|---|---|---|
| Instant NuRec (NVIDIA, Jul 2026) | 1-5 cams x <= 18 frames, 504 x 280, posed | yes | NVIDIA Open Model Licence, 838 MB | 28.26 dB / SSIM 0.859 (vs DGGT 26.25) | no: >= 30 GB, Linux |
| DGGT (CVPR 2026) | 3 cams x 20 frames, no poses needed | yes | CC-BY-NC, 5.41 GB | 27.41 dB Waymo; 25.31 zero-shot on nuScenes | unlikely |
| STORM (NVIDIA, ICLR 2025) | 3 cams x 4 timesteps, 160 x 240 | yes | no official checkpoint | 26.38 dB | unknown |
| DrivingForward / Omni-Scene / EVolSplat | 6 cams, 1-2 frames | no | MIT | 24-26 dB nuScenes | no (7.6-10.4 GB) |
| MVSplat / DepthSplat (generic) | 2-6 posed views | no | MIT, 48 MB / 154 MB | trained on indoor / real-estate video | plausible, but not for driving scenes |

Dense per-scene fitting on full sequences reaches about 29-33 dB at 640 x 960 (OmniRe's Waymo table: OmniRe 32.6,
StreetGS 28.5, 3DGS 25.6). Feed-forward models "win" only against optimisation restricted to the same few frames.
None has been evaluated on CARLA renders or on views 1-3 m off the driven path; NVIDIA positions Instant NuRec as a
fast initialiser for NuRec optimisation. Sources: <https://arxiv.org/html/2607.14203v1>,
<https://arxiv.org/html/2512.03004v1>, <https://arxiv.org/html/2501.00602>, <https://arxiv.org/html/2408.16760>.

**Conclusion:** per-scene fitting is needed for a usable reconstruction of our CARLA scene; a feed-forward pass would
at best be a low-resolution preview or an initialiser on a much bigger GPU.

## Open-source code NuRec builds on: 3DGRUT / 3DGUT

- `nv-tlabs/3dgrut` (Apache-2.0): 3DGRT (ray tracing, needs OptiX) and 3DGUT (unscented-transform rasterisation, no
  OptiX; distorted / rolling-shutter cameras). Windows supported since 2025-07 (`install_env_uv.ps1`), but it needs
  an installed CUDA toolkit + MSVC and JIT-builds its kernels (reported Windows build failures in its issues).
  Reads COLMAP, NeRF-synthetic, ScanNet++ and **NCore V4**; exports USDZ (`nurec` or `standard` format) for Isaac Sim.
  Static scenes only (its NCore loader drops dynamic lidar points but does not mask actors in images).
- `nerfstudio-project/gsplat` (Apache-2.0) has 3DGUT since v1.5.0 (`rasterization(..., with_ut=True,
  with_eval3d=True)`) and a **prebuilt Windows wheel** (`gsplat-1.5.3+pt24cu124-cp310`), so it needs no CUDA toolkit.

**Used here:** gsplat 3DGUT with a composite model written for this repo (background Gaussians + one rigid Gaussian
set per dynamic actor placed by its 3D box each frame + learnable sky), the same static / dynamic separation NuRec
makes from cuboid tracks. The recording is also exported as NCore V4 so NuRec (or 3DGRUT) can run on it unchanged
on a suitable machine (`cloud/`).

## Findings about this CARLA build (ue5-dev / 0.10) that matter for ground truth

Measured, see `convert/conventions.json`:
- The server renders with Unreal's rotation convention (positive pitch = nose up). The carla wheel's client-side
  `Transform.get_matrix()` / `get_forward_vector()` use the **opposite pitch / roll sign** (matrix = D M D,
  D = diag(1, 1, -1)); using them for geometry would tilt every pitched pose. Sky / horizon test: IoU 0.95-0.98 with
  Unreal's convention vs 0.35-0.62 with the client's.
- Lidar points are in the sensor frame with CARLA axes (median 5.6 cm from depth-camera points; y-flip: 4.3 m).
- Instance segmentation: R = semantic tag (identical to the semantic camera), G/B = an instance id that is **not**
  the actor id; ids are mapped to actors geometrically (`convert/instance_map.py`).
- The RGB camera blueprint has no exposure / motion-blur attributes; exposure is Unreal's automatic exposure
  (`evaluate/exposure_check.py` measures what that does).
- No two-wheeled vehicle is registered in the blueprint library (17 vehicles, all four-wheeled), although the bike
  blueprints exist in Content. The cyclist is CARLA's CrossBike mesh (`static.prop.mesh`) with a pedestrian posed
  as rider through the walker bone API (pose solved offline by inverse kinematics, `scenario/rider_ik.py`).
- The audit spot's lane (road 933, lane -1) is a left-turn-only lane; the ego drives straight from lane -3.
