# carla-nurec

Synthetic driving data from CARLA on a MirrorMile world (Downtown LA, `LA_DTLA_ODR`), recorded from the ego in a
multi-actor scenario, converted to NVIDIA NuRec's input format (NCore V4), reconstructed as 3D Gaussians with the
static background and the dynamic actors separated the way NuRec does it, and compared with CARLA's ground truth.

![scenario](docs/img/scenario_front3.gif)

**Read first:** [docs/report.md](docs/report.md) (summary, numbers, limitations) - [docs/results.md](docs/results.md)
(all tables and pictures) - [docs/scenario.md](docs/scenario.md) - [docs/formats.md](docs/formats.md) -
[docs/notes.md](docs/notes.md) (research: NuRec, 3DGRUT / 3DGUT, feed-forward models; CARLA build findings) -
[cloud/README.md](cloud/README.md) (running real NuRec on a Linux GPU machine).

| held-out on-path views | novel views +-1 / +-2 / +-3 m | depth AbsRel | lidar F-score @ 0.5 m | fit on RTX 4050 6 GB |
|---|---|---|---|---|
| 21.5 dB / SSIM 0.852 / LPIPS 0.230 | ~20.8 / ~19.6 / ~18.6 dB | 0.065 | 0.83 | 54 min, 1.95 GB VRAM |

## Layout

```
scenario/plan.py          scenario plan from the OpenDRIVE (paths, speed profiles, clearance check) -> plan.json
scenario/plot_plan.py     top-down picture of the plan
scenario/record.py        CARLA run: 6 cameras (RGB / depth / instance), lidar, actor boxes, recorder;
                          --novel: re-simulation with offset cameras for novel-view ground truth
scenario/rider_ik.py      inverse kinematics of the cyclist's rider pose (rider_pose.json); rider_pose.py renders it
scenario/rig.json         camera / lidar rig
convert/                  measured CARLA conventions, instance-id map, raw -> PNG, dataset, NCore V4 export
recon/model.py, train.py  composite 3DGUT model (gsplat): background + per-actor Gaussians + sky, trainer
evaluate/                 evaluation, exposure check, report generator, actor debug renders
cloud/                    NuRec run script + notes for a Linux GPU machine (not run here)
run_carla.ps1             CARLA server (-game, off-screen) -> client jobs -> server stopped (also on failure)
run_pipeline.ps1          everything after planning
```

## Running

Requirements: CARLA 0.10 UE5 source build with the `LA_DTLA_ODR` map; the CARLA Python client in a Python 3.8 venv
(`CARLA_PY38`, default `C:\mywork\osm2carla\venv38`); `.venv` (Python 3.10) with torch 2.4.0+cu124, gsplat
1.5.3+pt24cu124 (Windows wheel), numpy 1.26.4, scipy, pillow, lpips, nvidia-ncore.

```powershell
& C:\mywork\osm2carla\venv38\Scripts\python.exe scenario\plan.py C:\mywork\osm2carla\build\nurec\plan.json
.\run_pipeline.ps1 -Build C:\mywork\osm2carla\build\nurec -Tag 04
```

Outputs (recordings, datasets, NCore, checkpoints, evaluation) go to `-Build`, outside the repository.
