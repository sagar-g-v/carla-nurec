"""Per-actor box errors of a noisy-box run (train.py --box-noise --refine-boxes): given (noisy) vs refined boxes against
the ground-truth tracks, on training ticks (held-out ticks get no image gradient).
usage: box_per_actor.py DATASET_DIR CKPT OUT_JSON"""
import json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recon"))
from model import BoxDelta

D = json.load(open(os.path.join(sys.argv[1], "dataset.json")))
CK = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
test = {f["tick"] for f in D["frames"] if f["cams"]["CAM_FRONT"]["split"] == "test"}
keys = sorted(CK["box_delta"]["index"].items(), key=lambda kv: kv[1])
bd = BoxDelta([(k.split("|")[0], int(k.split("|")[1])) for k, _ in keys]); bd.d.data = CK["box_delta"]["d"]
yaw = lambda T: math.degrees(math.atan2(T[1, 0], T[0, 0]))
res = {}
for (n, t), i in bd.index.items():
    if t in test:
        continue
    Tg = np.array(CK["gt_tracks"][n][str(t)])
    Tn = torch.tensor(np.array(CK["tracks"][n][str(t)]), dtype=torch.float32)
    Tr = bd.apply(n, t, Tn).detach().numpy(); Tn = Tn.numpy()
    r = res.setdefault(D["actors"][n]["name"], dict(pos_given=[], pos_refined=[], yaw_given=[], yaw_refined=[]))
    r["pos_given"].append(float(np.linalg.norm(Tn[:2, 3] - Tg[:2, 3]))); r["pos_refined"].append(float(np.linalg.norm(Tr[:2, 3] - Tg[:2, 3])))
    r["yaw_given"].append(abs((yaw(Tn) - yaw(Tg) + 180) % 360 - 180)); r["yaw_refined"].append(abs((yaw(Tr) - yaw(Tg) + 180) % 360 - 180))
out = {n: {k: round(float(np.mean(v)), 4) for k, v in r.items()} | {"n": len(r["pos_given"])} for n, r in res.items()}
d = CK["box_delta"]["d"].numpy()
out["_delta_magnitude"] = dict(xy_mean_m=float(np.linalg.norm(d[:, :2], axis=1).mean()), yaw_mean_deg=float(np.degrees(np.abs(d[:, 3])).mean()))
json.dump(out, open(sys.argv[3], "w"), indent=1)
for n, v in out.items():
    print(n, v)
