"""Side-by-side table of two evaluations (evaluate.py metrics.json), read from the files, for the report.
usage: compare.py A_DIR B_DIR [--names A,B]"""
import json, os, sys

A, B = (json.load(open(os.path.join(d, "metrics.json"))) for d in sys.argv[1:3])
na, nb = sys.argv[sys.argv.index("--names") + 1].split(",") if "--names" in sys.argv else ("A", "B")
rows = [("training wall time, min", lambda m: m["train"]["elapsed_s"] / 60, 1),
        ("peak VRAM, MB", lambda m: m["train"]["vram_peak_mb"], 0),
        ("held-out PSNR raw, dB", lambda m: m["heldout"]["psnr"], 2),
        ("held-out PSNR exposure-aligned, dB", lambda m: m["heldout"]["aligned"]["psnr"], 2),
        ("held-out SSIM raw", lambda m: m["heldout"]["ssim"], 3),
        ("held-out LPIPS raw", lambda m: m["heldout"]["lpips"], 3)]
for o in ("-3", "-2", "-1", "+0", "+1", "+2", "+3"):
    rows.append(("novel %s m PSNR raw, dB" % o, lambda m, o=o: m["novel"][o]["psnr"], 2))
rows += [("depth AbsRel (held-out)", lambda m: m["heldout"]["depth"]["absrel"], 3),
         ("depth median error, m", lambda m: m["heldout"]["depth"]["median_abs"], 3),
         ("lidar depth error median, m", lambda m: m["geometry"]["lidar_depth_abs_err"]["median"], 3),
         ("lidar F-score @ 0.2 m", lambda m: m["geometry"]["chamfer"]["fscore_0_2m"], 3),
         ("lidar F-score @ 0.5 m", lambda m: m["geometry"]["chamfer"]["fscore_0_5m"], 3)]
for n in A["actors"]["mask_iou"]:
    rows.append(("actor mask IoU %s" % n, lambda m, n=n: m["actors"]["mask_iou"][n]["mean"], 3))
print("| | %s | %s |\n|---|---|---|" % (na, nb))
for label, f, d in rows:
    print("| %s | %.*f | %.*f |" % (label, d, f(A), d, f(B)))
