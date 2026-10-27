# carla-nurec

Synthetic driving data from CARLA on a MirrorMile world (Downtown LA, `LA_DTLA_ODR`), turned into a neural
reconstruction of the scene (3D Gaussians, static background + dynamic actors, the way NVIDIA NuRec separates them)
and checked against CARLA's own ground truth.

```
scenario/plan.py         scripted scenario: every actor's path + speed profile (OpenDRIVE lanes), clearance check
scenario/record.py       runs it in CARLA (sync, 0.1 s), records 6 cameras (RGB / depth / instance), lidar,
                         every actor's 3D box per frame, CARLA recorder file
scenario/replay_views.py replays the recorder file, renders novel views off the driven path
convert/                 measured conventions, instance-id map, dataset for training, NCore V4 for NuRec
recon/                   composite 3DGUT model (gsplat) + trainer
evaluate/                PSNR / SSIM / LPIPS, depth, lidar geometry, actor masks; exposure check; report
cloud/                   ready-to-run NuRec package for a Linux GPU machine (not run here)
docs/                    results, formats, research notes
```

See [docs/results.md](docs/results.md) for the measured results and [docs/notes.md](docs/notes.md) for the research
(NuRec, 3DGRUT / 3DGUT, feed-forward reconstruction) and the decisions behind this setup.
