# Data formats

## 1. Raw recording (`scenario/record.py` output)

| file | content |
|---|---|
| `meta.json` | rig (camera / lidar mounts in the ego frame, CARLA axes), resolution, FOV, dt, weather, frozen traffic lights, blueprint + CARLA id of every actor, recorder file |
| `frames.jsonl` | one line per tick: `tick`, CARLA `frame`, sim time `t`, `t_rec` (time since the recorder started), `ego` pose, every sensor's world transform as reported by CARLA, and every actor: `id`, `name`, `type_id`, `cls`, `kind`, `static`, `tf` (actor pose), `bbox` (box centre offset + half extents + rotation in the actor frame), `vel`, `angvel` |
| `cam/<CAM>/rgb/NNNN.png` | 1280 x 720 RGB |
| `cam/<CAM>/depth/NNNN.png` | CARLA depth encoding: depth_m = 1000 (R + 256 G + 65536 B) / (256^3 - 1), planar (along the optical axis) |
| `cam/<CAM>/inst/NNNN.png` | R = semantic tag (CARLA 0.9.14+ table, see `convert/carla_geom.py`), G + 256 B = instance id (mapped to actors by `convert/instance_map.py`) |
| `lidar/NNNN.bin` | float32 x, y, z, intensity; sensor frame, CARLA axes; one full 360 deg sweep per tick |
| `semlidar/NNNN.bin` | CARLA SemanticLidarDetection: float32 x, y, z, cos_inc_angle; uint32 object_idx, object_tag |
| `recording.log` | CARLA recorder file (`client.replay_file`) |
| `instance_map.json` | instance id -> actor (votes, purity), lidar object id -> actor |

Rig: 6 pinhole cameras, 90 deg horizontal FOV, 1280 x 720, fx = fy = 640, cx = 640, cy = 360 (pixel-corner origin:
the centre of pixel (0, 0) is (0.5, 0.5)), global shutter, mounted 1.7 m above the ground at yaw 0, +-60, +-120, 180;
64-beam lidar 2.0 m above the ground, +15 / -25 deg, 100 m, 10 Hz (one sweep per 0.1 s tick), no noise, no dropoff.

## 2. Reconstruction dataset (`convert/to_dataset.py` -> `dataset.json`)

All poses converted once to right-handed metric frames: **world** = CARLA world with y negated (z up);
**rig / actor boxes** = FLU (x forward, y left, z up); **camera** = OpenCV (x right, y down, z forward).
Rotations use the convention the server renders with (measured, `convert/conventions.json`).

| key | content |
|---|---|
| `cameras.<CAM>` | `K` (3 x 3, full resolution), `width`, `height`, `T_rig_cam` (camera -> rig) |
| `frames[i]` | `tick`, `t`, `T_world_rig`, `cams.<CAM>` {`T_world_cam`, `rgb`, `depth`, `inst`, `split` train / test}, `lidar` {`path`, `semlidar`, `T_world_lidar`} |
| `actors.<id>` | `name`, `cls`, `kind`, `size` [l, w, h], `dynamic` (NuRec's rule: pedestrians / cyclists always, others if they move > 1 m), `track` {tick: `T_world_box` 4 x 4}, cyclist: `box_offset`, `parts` |
| `merged` | rider walker id -> cyclist (bike) id |
| `ego` | id, size (the ego is masked out, not reconstructed) |
| `points_static.npy` | static-scene lidar points (world, 10 cm voxels) with RGB from the nearest unoccluded train image |
| `actor_points/<id>.npy` | each dynamic actor's lidar points accumulated over all frames, in its box frame |

Split: every 10th tick (ticks 5, 15, ..., 195; all six cameras) is held out from training.

## 3. NCore V4 (`convert/to_ncore.py`) - the input of NVIDIA NuRec

Written with `nvidia-ncore` 19.8.4 (Apache-2.0): one store `<name>.ncore4.zarr.itar` plus the manifest
`<name>.json` (`SequenceComponentGroupsReader(...).get_sequence_meta()`), store type `itar`.

| component (instance) | content |
|---|---|
| `poses` (`default`) | dynamic `("rig", "world")` for every tick (`timestamps_us` = CARLA sim time in us, 100 ms apart, covering the sequence interval); static `(<sensor>, "rig")` per camera and the lidar. NCore `T_a_b` maps a into b. |
| `intrinsics` (`default`) | cameras: `opencv-pinhole`, resolution 1280 x 720, focal 640 / 640, principal point 640 / 360 (pixel-corner origin), zero distortion, `GLOBAL` shutter; lidar `lidar_top`: `row-offset-spinning`, 64 rows (+15 .. -25 deg, uniform, top row first), 1800 columns, 10 Hz, `cw` |
| `masks` (`default`) | empty per camera (the cameras see no ego body; NuRec's `nre-tools` can add an ego mask) |
| `cameras` (`camera_front`, `camera_front_left`, `camera_front_right`, `camera_back_left`, `camera_back_right`, `camera_back`) | the recorded PNGs, `frame_timestamps_us = [t, t]` (global shutter) |
| `lidars` (`lidar_top`) | per sweep: unit ray directions in the lidar frame (FLU), `distance_m` [1, N], `intensity` [1, N] in 0..1, per-ray timestamp = sweep time (CARLA computes a sweep at one instant, so no rolling motion) |
| `cuboids` (`default`) | every actor except the ego at every tick: `track_id` `<name>_<carla id>`, `class_id` in NuRec's vocabulary (`automobile`, `pedestrian`, `cyclist`), reference frame `world`, `bbox3` = centre, full size (l, w, h), rotation as scipy `xyz` Euler angles (rad), `source` `GT_SYNTHETIC` |

`convert/to_ncore.py` reads the sequence back with NCore's `SequenceLoaderV4` and projects the cuboids of the middle
frame into `camera_front` through the loaded pose and intrinsics (`check_cuboids_camera_front_0100.png`).

What NuRec does with it (`cloud/run_nurec.sh`): `nre-tools` adds segmentation / lidar visibility (`.aux` stores next to
the manifest); `nre` classifies the cuboid tracks (pedestrians / cyclists always dynamic, others if they move > 1 m),
fits the background, `dynamic_rigids` (automobile) and `dynamic_deformables` (pedestrian, cyclist) layers, and writes
`usd-out/last.usdz`.
