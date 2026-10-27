"""Map CARLA instance-segmentation ids and semantic-lidar object ids to actors (numpy, .venv).
In this CARLA build the instance ids in the camera images are not actor ids, so they are identified geometrically:
depth pixels and lidar points are moved to the world and tested against every actor's ground-truth box (padded);
each id goes to the actor whose box holds most of its points.
Output REC/instance_map.json: {"camera": {inst_id: {actor, name, votes, purity}}, "lidar": {...}, "summary"}
usage: instance_map.py REC_DIR [--every 10] [--pad 0.15]"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carla_geom as G

REC = sys.argv[1]
EVERY = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 10
PAD = float(sys.argv[sys.argv.index("--pad") + 1]) if "--pad" in sys.argv else 0.15
CONV = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conventions.json")))
meta = json.load(open(os.path.join(REC, "meta.json")))
frames = [json.loads(l) for l in open(os.path.join(REC, "frames.jsonl"))]
W, H = meta["res"]
K = G.intrinsics(W, H, meta["rig"]["fov"])
SEML = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"), ("idx", "<u4"), ("tag", "<u4")])
u, v = np.meshgrid(np.arange(0, W, 2) + 0.5, np.arange(0, H, 2) + 0.5)
rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)


def boxes(fr):
    out = []
    for a in fr["actors"]:
        T = G.pose_rh(a["tf"], CONV["rotation"]) @ G.pose_rh(a["bbox"][:3] + a["bbox"][6:9], CONV["rotation"])
        pad = PAD + (0.25 if a["kind"] in ("walker", "rider") else 0.0)
        out.append((a["id"], a["name"], np.linalg.inv(T), np.array(a["bbox"][3:6]) + pad))
    return out


def owner(Pw, bxs):
    """index of the box that contains each world point (-1: none)."""
    own = np.full(len(Pw), -1)
    for i, (_, _, Ti, ext) in enumerate(bxs):
        L = Pw @ Ti[:3, :3].T + Ti[:3, 3]
        own[np.all(np.abs(L) <= ext, axis=1)] = i
    return own


cam_votes, lid_votes = defaultdict(Counter), defaultdict(Counter)
for fr in frames[::EVERY]:
    bxs = boxes(fr)
    names = {i: (b[0], b[1]) for i, b in enumerate(bxs)}
    for cam in meta["rig"]["cameras"]:
        c = cam["name"]
        dep = G.decode_depth(np.array(Image.open(os.path.join(REC, "cam", c, "depth", "%04d.png" % fr["tick"])).convert("RGB")))[::2, ::2]
        tag, iid = G.decode_inst(np.array(Image.open(os.path.join(REC, "cam", c, "inst", "%04d.png" % fr["tick"])).convert("RGB"))[::2, ::2])
        Pc = rays * dep.reshape(-1, 1)
        ok = dep.reshape(-1) < 80
        T = G.cam_to_world_cv(fr["sensors"]["%s/depth" % c], CONV["rotation"])
        Pw = Pc[ok] @ T[:3, :3].T + T[:3, 3]
        own = owner(Pw, bxs)
        for i_, o in zip(iid.reshape(-1)[ok], own):
            if o >= 0 or i_ != 0:
                cam_votes[int(i_)][names[o] if o >= 0 else (None, None)] += 1
    pts = np.frombuffer(open(os.path.join(REC, "semlidar", "%04d.bin" % fr["tick"]), "rb").read(), SEML)
    P = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64) @ np.diag(CONV["lidar_axes"]).T * [1, -1, 1]
    T = G.pose_rh(fr["sensors"]["lidar/sem"], CONV["rotation"])
    own = owner(P @ T[:3, :3].T + T[:3, 3], bxs)
    for i_, o in zip(pts["idx"], own):
        if o >= 0:
            lid_votes[int(i_)][names[o]] += 1


def resolve(votes, min_votes=20):
    out = {}
    for i_, cnt in votes.items():
        if i_ == 0:                                                     # 0 = not an actor (static world)
            continue
        (aid, name), n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if aid is not None and n >= min_votes:
            out[str(i_)] = dict(actor=aid, name=name, votes=n, purity=round(n / tot, 3))
    return out


cam_map, lid_map = resolve(cam_votes), resolve(lid_votes)
actor_ids = {a["id"]: a["name"] for a in frames[0]["actors"]}
summary = dict(camera_ids_per_actor={n: sorted(int(k) for k, m in cam_map.items() if m["actor"] == a) for a, n in actor_ids.items()},
               lidar_idx_equals_actor_id=all(int(k) == m["actor"] for k, m in lid_map.items()),
               min_camera_purity=min((m["purity"] for m in cam_map.values()), default=None),
               min_lidar_purity=min((m["purity"] for m in lid_map.values()), default=None))
json.dump(dict(camera=cam_map, lidar=lid_map, summary=summary, every=EVERY, pad=PAD), open(os.path.join(REC, "instance_map.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
