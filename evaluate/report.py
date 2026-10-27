"""metrics.json (+ panels) -> docs/report.md: every number in the report is read from the evaluation output.
usage: report.py EVAL_DIR DOCS_DIR [--extra JSON ...]   (extra: exposure_check.json, instance_map.json, replay.json)"""
import json, os, sys
from PIL import Image

E, DOCS = sys.argv[1], sys.argv[2]
M = json.load(open(os.path.join(E, "metrics.json")))
extra = {os.path.basename(p): json.load(open(p)) for p in sys.argv[sys.argv.index("--extra") + 1:]} if "--extra" in sys.argv else {}
img_out = os.path.join(DOCS, "img"); os.makedirs(img_out, exist_ok=True)
f2 = lambda v, d=2: "-" if v is None else ("%.*f" % (d, v))
L = []
tr = M["train"]
L.append("## Reconstruction\n")
L.append("| | |\n|---|---|")
L.append("| training resolution | %d x %d |" % tuple(tr["res"]))
L.append("| iterations | %d |" % tr["args"]["iters"])
L.append("| wall time | %.1f min |" % (tr["elapsed_s"] / 60))
L.append("| peak VRAM (torch allocated) | %.0f MB |" % tr["vram_peak_mb"])
L.append("| Gaussians (background / actors) | %d / %d |" % (tr["gaussians"]["bg"], sum(v for k, v in tr["gaussians"].items() if k != "bg")))
L.append("")
h = M["heldout"]
L.append("## Held-out on-path views (every %s-th tick, all cameras; %d images)\n" % (10, h["n"]))
L.append("| camera | PSNR dB | SSIM | LPIPS (alex) |\n|---|---|---|---|")
for c, v in h["per_camera"].items():
    L.append("| %s | %s | %s | %s |" % (c, f2(v["psnr"]), f2(v["ssim"], 3), f2(v["lpips"], 3)))
L.append("| **all, raw** | **%s** | **%s** | **%s** |" % (f2(h["psnr"]), f2(h["ssim"], 3), f2(h["lpips"], 3)))
L.append("| all, exposure-aligned | %s | %s | %s |" % (f2(h["aligned"]["psnr"]), f2(h["aligned"]["ssim"], 3), f2(h["aligned"]["lpips"], 3)))
L.append("\nRaw: the rendering as is. Exposure-aligned: a per-image affine colour fit between rendering and ground truth "
         "first (CARLA's automatic exposure differs per image; the model never saw these images' exposure).")
d = h["depth"]
L.append("\nDepth vs CARLA depth (non-sky, < 80 m, alpha > 0.5): AbsRel %s, RMSE %s m, median |error| %s m, delta<1.25 %s.\n"
         % (f2(d["absrel"], 3), f2(d["rmse"]), f2(d["median_abs"], 3), f2(d["d125"], 3)))
L.append("## Novel views off the driven path (front camera shifted sideways, CARLA re-simulation)\n")
L.append("| lateral offset (+ = right) | views | PSNR dB | SSIM | LPIPS | PSNR / SSIM / LPIPS exposure-aligned | depth AbsRel | depth median abs m |\n|---|---|---|---|---|---|---|---|")
for k, v in M["novel"].items():
    al = v["aligned"]
    L.append("| %s m | %d | %s | %s | %s | %s / %s / %s | %s | %s |" % (k, v["n"], f2(v["psnr"]), f2(v["ssim"], 3), f2(v["lpips"], 3),
             f2(al["psnr"]), f2(al["ssim"], 3), f2(al["lpips"], 3), f2(v["depth"]["absrel"], 3), f2(v["depth"]["median_abs"], 3)))
nf = M["replay_noise_floor"]
L.append("\nRe-simulation noise floor (re-simulated front camera at offset 0, ego parked out of view, vs the live "
         "recording, ego pixels excluded; differences come from walker animation, the missing ego shadow and exposure): "
         "PSNR %s dB, SSIM %s, LPIPS %s over %d frames. Re-simulated actor positions vs the live run (mean / max m): %s.\n"
         % (f2(nf["psnr"]), f2(nf["ssim"], 3), f2(nf["lpips"], 3), nf["n"],
            ", ".join("%s %.3f/%.3f" % (n, v["mean_m"], v["max_m"]) for n, v in nf["replay_pose_error"].items())))
if M.get("novel_skipped"):
    L.append("Skipped %d novel views whose camera was inside an actor box: %s.\n" % (len(M["novel_skipped"]),
             ", ".join("frame %d %+.0f m (%s)" % (s["frame"], s["offset"], s["inside"]) for s in M["novel_skipped"][:12])))
g = M["geometry"]
L.append("## Geometry vs lidar (held-out ticks, within 50 m)\n")
L.append("| | |\n|---|---|")
L.append("| rendered depth at lidar points: median abs error | %s m |" % f2(g["lidar_depth_abs_err"]["median"], 3))
L.append("| ... within 0.5 m / 1 m | %s / %s |" % (f2(g["lidar_depth_abs_err"]["within_0_5m"], 3), f2(g["lidar_depth_abs_err"]["within_1m"], 3)))
c = g["chamfer"]
L.append("| chamfer: accuracy (recon -> lidar) mean / median | %s / %s m |" % (f2(c["accuracy_mean"], 3), f2(c["accuracy_median"], 3)))
L.append("| chamfer: completeness (lidar -> recon) mean / median | %s / %s m |" % (f2(c["completeness_mean"], 3), f2(c["completeness_median"], 3)))
L.append("| F-score @ 0.2 m / 0.5 m | %s / %s |" % (f2(c["fscore_0_2m"], 3), f2(c["fscore_0_5m"], 3)))
L.append("")
A = M["actors"]
L.append("## Dynamic actors\n")
L.append("| actor | mask IoU (views) | Gaussian extent l x w x h (m) | ground-truth box (m) | centre offset (m) |\n|---|---|---|---|---|")
for n in sorted(set(A["mask_iou"]) | set(A["geometry"])):
    iou = A["mask_iou"].get(n); geo = A["geometry"].get(n, {})
    L.append("| %s | %s | %s | %s | %s |" % (n, "%s (%d)" % (f2(iou["mean"], 3), iou["n"]) if iou else "-",
             " x ".join(f2(v) for v in geo.get("extent_rec", [])) or "-", " x ".join(f2(v) for v in geo.get("box_size", [])) or "-",
             f2(geo.get("centre_offset_m"), 3)))
L.append("\n%s\n" % A["note"])
if "recorder_replay_fidelity.json" in extra:
    r = extra["recorder_replay_fidelity.json"]
    L.append("CARLA recorder replay (not used for ground truth): best alignment %s; replayed actor position error vs the live "
             "run (mean / max m): %s.\n" % (r["alignment"].split(": ", 1)[-1], ", ".join("%s %.2f/%.2f" % (n, v["mean_m"], v["max_m"])
             for n, v in r["pose_error_vs_live_m"].items() if v["mean_m"] < 100)))
if M.get("box_refinement"):
    b = M["box_refinement"]
    L.append("## Actor boxes from a noisy tracker (noise %s m, deg per frame)\n" % b["noise"])
    L.append("| boxes | position error mean / median (m) | yaw error mean (deg) | box-frames |\n|---|---|---|---|")
    for k in ("given", "refined"):
        L.append("| %s | %s / %s | %s | %d |" % (k, f2(b[k]["pos_mean_m"], 3), f2(b[k]["pos_median_m"], 3), f2(b[k]["yaw_mean_deg"], 2), b[k]["n"]))
    L.append("")
if "exposure_check.json" in extra:
    s = extra["exposure_check.json"]["summary"]
    L.append("## Exposure check\n")
    L.append("Median brightness ratio of the same static surfaces in two images (1.00 = same exposure): " +
             "; ".join("%s %.3f (5-95 %%: %.3f-%.3f, %d pairs)" % (k.replace("_", " "), v["median"], v["p5"], v["p95"], v["n"]) for k, v in s.items()) + ".\n")
L.append("## Pictures\n")
for f in sorted(os.listdir(os.path.join(E, "img"))):
    j = f[:-4] + ".jpg"                                            # JPEG keeps the repository small
    Image.open(os.path.join(E, "img", f)).convert("RGB").save(os.path.join(img_out, j), quality=88)
    L.append("**%s** (top: CARLA | reconstruction; bottom: |error| x4 | rendered depth)\n\n![%s](img/%s)\n" % (f[:-4], j, j))
open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8").write("\n".join(L))
print("wrote", os.path.join(DOCS, "results.md"))
