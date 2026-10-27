"""CARLA recording -> the reconstruction dataset used by recon/train.py and evaluate/ (numpy, .venv).

Everything is converted once into right-handed metric frames (carla_geom.py): world = CARLA world with y negated,
rig / actor boxes = FLU, cameras = OpenCV. Conventions measured on this CARLA build are read from
conventions.json (rotation matrix, lidar axes, instance-id byte order) - see convert/convention_check.py.

OUT/dataset.json
  cameras  {CAM: K (3x3, at full resolution), width, height, T_rig_cam (OpenCV camera -> rig, from the first frame)}
  frames   [{tick, t, T_world_rig, cams: {CAM: {T_world_cam, rgb, depth, inst, split}}, lidar: {path, T_world_lidar}}]
  actors   {actor id: {name, cls, kind, dynamic, size [l, w, h], track: {tick: T_world_box}}}  (ego excluded)
  ego      {id, size}
  split    every TEST_EVERY-th tick (all cameras of it) is held out
OUT/points_static.npy   N x 6 (x y z r g b), lidar points of the static scene in the world, voxel-filtered
OUT/actor_points/<id>.npy   lidar points of each dynamic actor in its box frame (all frames, voxel-filtered)
usage: to_dataset.py REC_DIR OUT_DIR [--test-every 10] [--voxel 0.10]"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carla_geom as G

REC, OUT = sys.argv[1], sys.argv[2]
TEST_EVERY = int(sys.argv[sys.argv.index("--test-every") + 1]) if "--test-every" in sys.argv else 10
VOXEL = float(sys.argv[sys.argv.index("--voxel") + 1]) if "--voxel" in sys.argv else 0.10
CONV = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conventions.json")))
os.makedirs(os.path.join(OUT, "actor_points"), exist_ok=True)
meta = json.load(open(os.path.join(REC, "meta.json")))
frames = [json.loads(l) for l in open(os.path.join(REC, "frames.jsonl"))]
W, H = meta["res"]
K = G.intrinsics(W, H, meta["rig"]["fov"])
LIDAR_AXES = np.diag(CONV["lidar_axes"])                       # lidar point (CARLA build) -> sensor frame, CARLA axes
SEMLIDAR = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"), ("idx", "<u4"), ("tag", "<u4")])


def pose(tf):
    return G.pose_rh(tf, CONV["rotation"])


def cam_pose(tf):
    return pose(tf) @ G.FLU_FROM_CV


ego_id = meta["actors"]["ego"]["id"]
cams = [c["name"] for c in meta["rig"]["cameras"]]
IMAP = json.load(open(os.path.join(REC, "instance_map.json")))       # convert/instance_map.py
# the cyclist is CARLA's bike mesh + a walker posed as rider, rigidly together: one actor in the bike's box frame
rider_id = meta["actors"].get("cyclist_rider", {}).get("id")
bike_id = meta["actors"].get("cyclist", {}).get("id")
MERGE = {rider_id: bike_id} if rider_id is not None and bike_id is not None else {}
LIDAR2ACTOR = {int(k): v["actor"] for k, v in IMAP["lidar"].items()}

# ---------------------------------------------------------------- actors: tracks, dynamic flag (NuRec's rule)
actors = {}
for fr in frames:
    for a in fr["actors"]:
        if a["id"] == ego_id or a["id"] in MERGE:
            continue
        d = actors.setdefault(a["id"], dict(name=a["name"], cls=a["cls"], kind=a["kind"], type_id=a["type_id"],
                                            size=[2 * a["bbox"][3], 2 * a["bbox"][4], 2 * a["bbox"][5]], track={}))
        T_box = pose(a["bbox"][:3] + a["bbox"][6:9])                 # box centre in the actor frame (FLU)
        d["track"][fr["tick"]] = (pose(a["tf"]) @ T_box).tolist()
for aid, d in actors.items():
    P = np.array([np.array(T)[:3, 3] for T in d["track"].values()])
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
    disp = float(np.linalg.norm(P[-1] - P[0]))
    d["path_m"], d["displacement_m"] = round(path, 3), round(disp, 3)
    d["dynamic"] = d["cls"] in ("pedestrian", "bicycle", "cyclist", "motorcycle") or path > 1.0 or disp > 1.0
dyn_ids = sorted(a for a, d in actors.items() if d["dynamic"])
print("actors", len(actors), "dynamic", [actors[a]["name"] for a in dyn_ids])

# ---------------------------------------------------------------- frames
out_frames = []
for fr in frames:
    k = fr["tick"]
    row = dict(tick=k, t=fr["t"], T_world_rig=pose(fr["ego"]).tolist(), cams={},
               lidar=dict(path=os.path.join(REC, "lidar", "%04d.bin" % k),
                          semlidar=os.path.join(REC, "semlidar", "%04d.bin" % k),
                          T_world_lidar=pose(fr["sensors"]["lidar/lidar"]).tolist()))
    for c in cams:
        row["cams"][c] = dict(T_world_cam=cam_pose(fr["sensors"]["%s/rgb" % c]).tolist(),
                              rgb=os.path.join(REC, "cam", c, "rgb", "%04d.png" % k),
                              depth=os.path.join(REC, "cam", c, "depth", "%04d.png" % k),
                              inst=os.path.join(REC, "cam", c, "inst", "%04d.png" % k),
                              split="test" if k % TEST_EVERY == TEST_EVERY // 2 else "train")
        # the three cameras of one position must share the pose (they are separate CARLA sensors)
        for m in ("depth", "inst"):
            dq = np.abs(np.array(fr["sensors"]["%s/%s" % (c, m)]) - np.array(fr["sensors"]["%s/rgb" % c])).max()
            assert dq < 1e-3, (k, c, m, dq)
    out_frames.append(row)
T0 = np.array(out_frames[0]["T_world_rig"])
cam_info = {c: dict(K=K.tolist(), width=W, height=H,
                    T_rig_cam=(np.linalg.inv(T0) @ np.array(out_frames[0]["cams"][c]["T_world_cam"])).tolist()) for c in cams}

# ---------------------------------------------------------------- lidar: static scene points + per-actor points
def vox(P, size):
    if len(P) == 0:
        return P
    key = np.floor(P[:, :3] / size).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


static, per_actor = [], {a: [] for a in dyn_ids}
inv_track = {a: {k: np.linalg.inv(np.array(T)) for k, T in actors[a]["track"].items()} for a in dyn_ids}
for row in out_frames:
    k = row["tick"]
    pts = np.frombuffer(open(row["lidar"]["semlidar"], "rb").read(), SEMLIDAR)
    P = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64) @ LIDAR_AXES.T     # CARLA sensor axes
    P = P * np.array([1.0, -1.0, 1.0])                                                        # -> FLU
    T = np.array(row["lidar"]["T_world_lidar"])
    Pw = P @ T[:3, :3].T + T[:3, 3]
    idx, tag = pts["idx"].astype(np.int64), pts["tag"].astype(np.int64)
    if len(idx):                                                    # lidar object ids -> actor ids, rider -> cyclist
        uq, inv = np.unique(idx, return_inverse=True)
        to = [LIDAR2ACTOR.get(int(i), int(i)) for i in uq]
        idx = np.array([MERGE.get(t, t) for t in to], np.int64)[inv]
    keep = (idx != ego_id) & ~np.isin(tag, G.DYNAMIC_TAGS) & ~np.isin(idx, dyn_ids)
    static.append(np.concatenate([Pw[keep], tag[keep, None]], 1))
    for a in dyn_ids:
        m = idx == a
        if m.any() and k in inv_track[a]:
            Ti = inv_track[a][k]
            per_actor[a].append(Pw[m] @ Ti[:3, :3].T + Ti[:3, 3])
static = vox(np.concatenate(static), VOXEL)
# colour of each static point: the RGB pixel of the nearest-in-time train camera that sees it unoccluded
rgb = np.full((len(static), 3), 128, np.uint8); have = np.zeros(len(static), bool)
for row in out_frames[::5]:
    for c in cams:
        cam = row["cams"][c]
        if cam["split"] != "train":
            continue
        Tcw = np.linalg.inv(np.array(cam["T_world_cam"]))
        Pc = static[:, :3] @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = Pc[:, 2]
        ok = (~have) & (z > 0.5) & (z < 60)
        u = K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        dep = G.decode_depth(np.array(Image.open(cam["depth"]).convert("RGB")))
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        vis = np.abs(dep[vi, ui] - z[ok]) < 0.05 * z[ok] + 0.1                                 # not occluded
        img = np.array(Image.open(cam["rgb"]).convert("RGB"))
        sel = np.flatnonzero(ok)[vis]
        rgb[sel] = img[vi[vis], ui[vis]]; have[sel] = True
static_out = np.concatenate([static[:, :3], rgb.astype(np.float64)], 1).astype(np.float32)
np.save(os.path.join(OUT, "points_static.npy"), static_out)
np.save(os.path.join(OUT, "points_static_tag.npy"), static[:, 3].astype(np.uint8))
act_counts = {}
for a in dyn_ids:
    P = vox(np.concatenate(per_actor[a]), 0.03) if per_actor[a] else np.zeros((0, 3))
    np.save(os.path.join(OUT, "actor_points", "%d.npy" % a), P.astype(np.float32))
    act_counts[actors[a]["name"]] = len(P)
    if a == bike_id and len(P):                                       # cyclist box = bike box U rider points
        half = np.array(actors[a]["size"]) / 2
        lo = np.minimum(np.percentile(P, 1, axis=0), -half); hi = np.maximum(np.percentile(P, 99, axis=0), half)
        actors[a].update(size=(hi - lo).round(3).tolist(), box_offset=((hi + lo) / 2).round(3).tolist(), cls="cyclist",
                         parts=["bike mesh (static.prop.mesh)", "rider walker %d" % rider_id])
print("static points", len(static_out), "coloured", int(have.sum()), "actor points", act_counts)

ego = [a for a in frames[0]["actors"] if a["id"] == ego_id][0]
json.dump(dict(source=os.path.abspath(REC), conventions=CONV, merged={str(k): v for k, v in MERGE.items()}, res=[W, H], fov=meta["rig"]["fov"], dt=meta["dt"],
               test_every=TEST_EVERY, cameras=cam_info, frames=out_frames,
               actors={str(a): d for a, d in actors.items()}, dynamic_ids=dyn_ids,
               ego=dict(id=ego_id, size=[2 * ego["bbox"][3], 2 * ego["bbox"][4], 2 * ego["bbox"][5]]),
               axes="world: CARLA world with y negated (right-handed, z up, m); rig/box: FLU; camera: OpenCV"),
          open(os.path.join(OUT, "dataset.json"), "w"))
n_test = sum(1 for f in out_frames for c in f["cams"].values() if c["split"] == "test")
print("frames", len(out_frames), "images", len(out_frames) * len(cams), "test images", n_test)
