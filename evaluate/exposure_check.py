"""Does CARLA's automatic exposure change between images? (numpy, .venv)
The same static surface seen by two images: pixels of image A are moved to the world with A's depth and projected
into image B; where B's depth agrees (no occlusion), not sky, not a dynamic actor, the brightness ratio B / A is
collected. A median ratio of 1.00 means the same exposure. Pairs: neighbouring cameras at the same tick (overlap),
and the same camera 5 and 20 ticks apart.
usage: exposure_check.py DATASET_DIR [--every 20]"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import carla_geom as G

D = json.load(open(os.path.join(sys.argv[1], "dataset.json")))
EVERY = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 20
W, H = D["res"]
K = np.array(D["cameras"]["CAM_FRONT"]["K"])
S = 4                                                                     # pixel stride


def load(cam):
    rgb = np.array(Image.open(cam["rgb"]).convert("RGB")).astype(np.float64)
    dep = G.decode_depth(np.array(Image.open(cam["depth"]).convert("RGB")))
    tag, _ = G.decode_inst(np.array(Image.open(cam["inst"]).convert("RGB")))
    return rgb.mean(-1), dep, tag


def ratio(a, b):
    la, da, ta = load(a); lb, db, tb = load(b)
    v, u = np.mgrid[S // 2:H:S, S // 2:W:S]
    z = da[v, u]
    ok = (z < 60) & ~np.isin(ta[v, u], (11,) + G.DYNAMIC_TAGS) & (la[v, u] > 20) & (la[v, u] < 235)
    P = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)[ok]
    T = np.linalg.inv(np.array(b["T_world_cam"])) @ np.array(a["T_world_cam"])
    Q = P @ T[:3, :3].T + T[:3, 3]
    zb = Q[:, 2]; good = zb > 0.5
    ub = K[0, 0] * Q[:, 0] / np.maximum(zb, 1e-6) + K[0, 2] - 0.5; vb = K[1, 1] * Q[:, 1] / np.maximum(zb, 1e-6) + K[1, 2] - 0.5
    good &= (ub >= 0) & (ub < W - 1) & (vb >= 0) & (vb < H - 1)
    ui, vi = np.round(ub[good]).astype(int), np.round(vb[good]).astype(int)
    vis = np.abs(db[vi, ui] - zb[good]) < 0.02 * zb[good] + 0.05
    vis &= ~np.isin(tb[vi, ui], (11,) + G.DYNAMIC_TAGS) & (lb[vi, ui] > 20) & (lb[vi, ui] < 235)
    if vis.sum() < 200:
        return None
    r = lb[vi[vis], ui[vis]] / la[v[ok][good][vis], u[ok][good][vis]]
    return float(np.median(r)), int(vis.sum())


fr = {f["tick"]: f for f in D["frames"]}
order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
out = dict(neighbour_same_tick=[], same_cam_5_ticks=[], same_cam_20_ticks=[])
for k in sorted(fr)[::EVERY]:
    for i, c in enumerate(order):
        r = ratio(fr[k]["cams"][c], fr[k]["cams"][order[(i + 1) % 6]])
        if r:
            out["neighbour_same_tick"].append(dict(tick=k, a=c, b=order[(i + 1) % 6], ratio=r[0], n=r[1]))
        for dk, key in ((5, "same_cam_5_ticks"), (20, "same_cam_20_ticks")):
            if k + dk in fr:
                r = ratio(fr[k]["cams"][c], fr[k + dk]["cams"][c])
                if r:
                    out[key].append(dict(tick=k, cam=c, ratio=r[0], n=r[1]))
summ = {k: dict(n=len(v), median=float(np.median([x["ratio"] for x in v])), p5=float(np.percentile([x["ratio"] for x in v], 5)),
                p95=float(np.percentile([x["ratio"] for x in v], 95))) for k, v in out.items() if v}
json.dump(dict(summary=summ, pairs=out), open(os.path.join(sys.argv[1], "exposure_check.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
