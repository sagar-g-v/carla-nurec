"""Assets for the 3D replay viewer (replay/index.html) from the recording, its dataset and a fitted scene, all in the
checkpoint's local frame (right-handed world = CARLA with y negated, minus the checkpoint centre; z up, metres):
  scene.json        ego and actor box poses per tick, ego rig poses, camera rig, map layers, metrics, per-image PSNR
  splats/*.splat    the reconstruction as Gaussian splats (.splat: position, scale, RGBA, rotation; colour from the
                    degree-0 spherical harmonics, so no view-dependent colour): the background in the local frame and
                    one file per dynamic actor in its box frame (the viewer places it by the actor's box every frame)
  scene_points.bin  static lidar points of the whole drive (10 cm voxels) coloured from the cameras: int16 x y z (cm)
                    for all points, then uint8 r g b
  img/CAM/{carla,recon}/NNNN.jpg   CARLA image and reconstruction render of every camera and tick (training size)
  novel/OFF/{carla,recon}/NNNN.jpg front camera at the re-simulated side offsets
  lidar.bin         semantic lidar per tick, subsampled: int16 x y z (cm) for all points, then uint8 tags
usage: build_replay.py DATASET_DIR CKPT EVAL_DIR MAP_LAYERS_JSON OUT_DIR [--novel NOVEL_DIR] [--lidar-points 20000]"""
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recon")); sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from model import Scene
import carla_geom as G

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("ckpt"); ap.add_argument("eval"); ap.add_argument("map"); ap.add_argument("out")
ap.add_argument("--novel"); ap.add_argument("--lidar-points", type=int, default=20000)
args = ap.parse_args()
t_start = time.time()
dev = "cuda"
D = json.load(open(os.path.join(args.data, "dataset.json")))
CK = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ROT = D["conventions"]["rotation"]
W, H, FACT = CK["W"], CK["H"], CK["factor"]
K = torch.tensor(CK["K"], dtype=torch.float32)
C = np.array(CK["center"])
SHIFT = np.eye(4); SHIFT[:3, 3] = -C
for sub in ("splats", "img", "novel"):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)
FRAMES = {f["tick"]: f for f in D["frames"]}
TICKS = sorted(FRAMES)
CAMS = list(D["frames"][0]["cams"])
LIVE = {json.loads(l)["tick"]: json.loads(l) for l in open(os.path.join(D["source"], "frames.jsonl"))}
imap = json.load(open(os.path.join(D["source"], "instance_map.json")))
EGO_INST = [int(k) for k, v in imap["camera"].items() if v["actor"] == D["ego"]["id"]]


def pose7(T):
    """4x4 -> [x, y, z, qx, qy, qz, qw] (three.js quaternion order)."""
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [round(float(v), 3) for v in T[:3, 3]] + [round(float(v), 5) for v in q]


# ---------------------------------------------------------------- splats
def write_splat(g, path, keep=None):
    """antimatter15 .splat, 32 bytes per Gaussian, most opaque / largest first."""
    keep = np.ones(len(g["means"]), bool) if keep is None else keep
    m = g["means"].numpy()[keep].astype(np.float32)
    s = np.exp(g["scales"].numpy()[keep]).astype(np.float32)
    q = g["quats"].numpy()[keep]; q = q / np.linalg.norm(q, axis=1, keepdims=True)               # w x y z
    a = 1 / (1 + np.exp(-g["opacities"].numpy()[keep]))
    rgb = np.clip(0.5 + 0.28209479177387814 * g["sh0"].numpy()[keep][:, 0], 0, 1)
    order = np.argsort(-(a * s.prod(1)))
    buf = np.zeros(len(m), np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)]))
    buf["p"], buf["s"] = m[order], s[order]
    buf["c"] = np.c_[rgb, a][order] * 255 + 0.5
    buf["r"] = np.clip(q[order] * 128 + 128, 0, 255)
    buf.tofile(path)
    return len(m), m


# background Gaussians farther than FAR_KEEP from the centre lie beyond the model's 2000 m far plane from every camera
# (cameras stay within ~100 m of the centre), so the trained model never draws them; they are left out of the export
FAR_KEEP = 2200.0
splats = []
for name, g in CK["groups"].items():
    f = "splats/%s.splat" % ("background" if name == "bg" else "actor_" + name)
    keep = None
    if name == "bg":
        r = np.linalg.norm(g["means"].numpy(), axis=1)
        a = 1 / (1 + np.exp(-g["opacities"].numpy()))
        keep = r <= FAR_KEEP
        print("background: %d Gaussians; distance from the centre p50 %.0f m, p99 %.0f m, max %.3g m; %d (%.1f %%) beyond "
              "%.0f m left out (their opacity p50 %.3f); within 300 m: %d" % (len(r), *np.percentile(r, [50, 99]), r.max(),
              (~keep).sum(), 100 * (~keep).mean(), FAR_KEEP, np.median(a[~keep]), (r <= 300).sum()), flush=True)
    n, _ = write_splat(g, os.path.join(args.out, f), keep)
    splats.append(dict(group=name, file=f, n=n, left_out_far=int((~keep).sum()) if keep is not None else 0))

# ---------------------------------------------------------------- static scene points (the viewer's backdrop)
SEM_RGB = np.array([[0,0,0],[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],
                    [220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],
                    [0,80,100],[0,0,230],[119,11,32],[110,190,160],[170,120,50],[55,90,80],[45,60,150],[157,234,50],[81,0,81],
                    [150,100,100],[230,150,140],[180,165,180]], np.uint8)
P = np.load(os.path.join(args.data, "points_static.npy")); TAG = np.load(os.path.join(args.data, "points_static_tag.npy"))
loc = np.round((P[:, :3] - C) * 100)
inside = (np.abs(loc) <= 32767).all(1)
rgb = P[:, 3:6].astype(np.uint8)
uncoloured = (rgb == 128).all(1)                                       # seen by no camera: semantic colour, dimmed
rgb[uncoloured] = (SEM_RGB[np.minimum(TAG[uncoloured], len(SEM_RGB) - 1)] * 0.6).astype(np.uint8)
with open(os.path.join(args.out, "scene_points.bin"), "wb") as f:
    loc[inside].astype("<i2").tofile(f); rgb[inside].tofile(f)
n_points = int(inside.sum())
print("scene points: %d (%d without a camera colour, %d outside the int16 range left out)" %
      (n_points, (uncoloured & inside).sum(), (~inside).sum()), flush=True)

# ---------------------------------------------------------------- tracks, rig, map
ego_row = lambda k: next(a for a in LIVE[k]["actors"] if a["id"] == D["ego"]["id"])
e0 = ego_row(TICKS[0])
ego = dict(id=D["ego"]["id"], name="ego", size=[round(2 * v, 3) for v in e0["bbox"][3:6]],
           box=[pose7(SHIFT @ G.pose_rh(ego_row(k)["tf"], ROT) @ G.pose_rh(ego_row(k)["bbox"][:3] + ego_row(k)["bbox"][6:9], ROT))
                for k in TICKS],
           rig=[pose7(SHIFT @ np.array(FRAMES[k]["T_world_rig"])) for k in TICKS])
actors = []
for aid, a in D["actors"].items():
    actors.append(dict(id=int(aid), name=a["name"], cls=a["cls"], kind=a["kind"], size=[round(v, 3) for v in a["size"]],
                       dynamic=a["dynamic"], splat=next((s["file"] for s in splats if s["group"] == aid), None),
                       box=[pose7(SHIFT @ np.array(a["track"][str(k)])) if str(k) in a["track"] else None for k in TICKS]))
to_local = lambda p: [round(p[0] - C[0], 2), round(-p[1] - C[1], 2), round(p[2] - C[2], 2)]
ML = json.load(open(args.map))["layers"]
layers = {k: [dict(v, p=[to_local(p) for p in v["p"]]) if isinstance(v, dict) else [to_local(p) for p in v] for v in vs]
          for k, vs in ML.items()}
cams = {c: dict(T_rig_cam=np.round(np.array(D["cameras"][c]["T_rig_cam"]), 6).tolist(), fov=D["fov"],
                width=D["cameras"][c]["width"], height=D["cameras"][c]["height"]) for c in CAMS}

# ---------------------------------------------------------------- images: CARLA vs reconstruction
groups = {n: torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in g.items()}) for n, g in CK["groups"].items()}
scene = Scene(groups, CK["tracks"]).to(dev)
scene.sky.load_state_dict(CK["sky"])
scene.eval()


def load_carla(rgb_p, inst_p):
    rgb = np.array(Image.open(rgb_p).convert("RGB").resize((W, H), Image.BOX))
    ego = None
    if inst_p:
        _, iid = G.decode_inst(np.array(Image.open(inst_p).convert("RGB")))
        ego = np.isin(iid, EGO_INST).reshape(H, FACT, W, FACT).mean((1, 3)) > 0.5
    return rgb, ego


def save_jpg(a, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path, "JPEG", quality=88)


@torch.no_grad()
def render(T, tick):
    rgb, _, _, _, _ = scene.render(torch.tensor(T, dtype=torch.float32), K, W, H, tick)
    return (rgb.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def psnr(a, b, ok):
    d = ((a.astype(np.float32) - b.astype(np.float32)) / 255.0) ** 2
    return round(-10 * math.log10(max(float(d[ok].mean()), 1e-10)), 2)


pool = ThreadPoolExecutor(8)
PSNR = {c: [] for c in CAMS}
jobs = [(c, k) for c in CAMS for k in TICKS]
pending = {j: pool.submit(load_carla, FRAMES[j[1]]["cams"][j[0]]["rgb"], FRAMES[j[1]]["cams"][j[0]]["inst"]) for j in jobs[:16]}
for i, (c, k) in enumerate(jobs):
    if i + 16 < len(jobs):
        j = jobs[i + 16]
        pending[j] = pool.submit(load_carla, FRAMES[j[1]]["cams"][j[0]]["rgb"], FRAMES[j[1]]["cams"][j[0]]["inst"])
    rec = render(SHIFT @ np.array(FRAMES[k]["cams"][c]["T_world_cam"]), k)
    gt, ego_m = pending.pop((c, k)).result()
    PSNR[c].append(psnr(rec, gt, ~ego_m))
    pool.submit(save_jpg, gt, os.path.join(args.out, "img", c, "carla", "%04d.jpg" % k))
    pool.submit(save_jpg, rec, os.path.join(args.out, "img", c, "recon", "%04d.jpg" % k))
print("rig images: %d views, %.0f s" % (len(jobs), time.time() - t_start), flush=True)

novel = None
if args.novel:
    RP = json.load(open(os.path.join(args.novel, "replay.json")))
    novel = dict(offsets=RP["offsets"], ticks=[fr["frame"] for fr in RP["frames"]], psnr={})
    for fr in RP["frames"]:
        k = fr["frame"]
        for off in RP["offsets"]:
            key = "%+.0f" % off
            T = SHIFT @ G.cam_to_world_cv(fr["cams"][key + "/rgb"], ROT)
            rec = render(T, k)
            gt, _ = load_carla(os.path.join(args.novel, "novel", "off_" + key, "rgb", "%04d.png" % k), None)
            novel["psnr"].setdefault(key, []).append(psnr(rec, gt, np.ones((H, W), bool)))
            pool.submit(save_jpg, gt, os.path.join(args.out, "novel", key, "carla", "%04d.jpg" % k))
            pool.submit(save_jpg, rec, os.path.join(args.out, "novel", key, "recon", "%04d.jpg" % k))
    novel["pose"] = {"%+.0f" % off: [pose7(SHIFT @ G.cam_to_world_cv(fr["cams"]["%+.0f/rgb" % off], ROT)) for fr in RP["frames"]]
                     for off in RP["offsets"]}
    print("novel images: %d views, %.0f s" % (len(RP["frames"]) * len(RP["offsets"]), time.time() - t_start), flush=True)
pool.shutdown(wait=True)

# ---------------------------------------------------------------- lidar
SEML = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"), ("idx", "<u4"), ("tag", "<u4")])
rng = np.random.default_rng(0)
xyz, tags, counts = [], [], []
for k in TICKS:
    pts = np.frombuffer(open(FRAMES[k]["lidar"]["semlidar"], "rb").read(), SEML)
    if len(pts) > args.lidar_points:
        pts = pts[np.sort(rng.choice(len(pts), args.lidar_points, replace=False))]
    P = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64) @ np.diag(D["conventions"]["lidar_axes"]).T
    P = P * np.array([1.0, -1.0, 1.0])                                                            # -> FLU
    T = SHIFT @ np.array(FRAMES[k]["lidar"]["T_world_lidar"])
    xyz.append(np.clip(np.round((P @ T[:3, :3].T + T[:3, 3]) * 100), -32767, 32767).astype("<i2"))
    tags.append(pts["tag"].astype(np.uint8)); counts.append(len(pts))
with open(os.path.join(args.out, "lidar.bin"), "wb") as f:
    np.concatenate(xyz).tofile(f); np.concatenate(tags).tofile(f)

# ---------------------------------------------------------------- metrics, scene.json
M = json.load(open(os.path.join(args.eval, "metrics.json")))
pick = lambda d: {k: round(d[k], 4) for k in ("psnr", "ssim", "lpips") if k in d}
metrics = dict(heldout=dict(n=M["heldout"]["n"], raw=pick(M["heldout"]), aligned=pick(M["heldout"].get("aligned", {}))),
               novel={k: dict(n=v["n"], raw=pick(v)) for k, v in M["novel"].items()},
               depth=M["heldout"].get("depth"), geometry=M.get("geometry"),
               actor_iou={n: round(v["mean"], 3) for n, v in M["actors"]["mask_iou"].items()})
test = sorted(t for t, f in FRAMES.items() if f["cams"]["CAM_FRONT"]["split"] == "test")
scene_json = dict(title="CARLA scenario replay", map="LA_DTLA_ODR", center=C.round(4).tolist(), dt=D["dt"], ticks=TICKS,
                  duration=round(TICKS[-1] * D["dt"], 3), test_ticks=test, ego=ego, actors=actors, cams=cams,
                  image_size=[W, H], layers=layers, splats=splats, psnr=PSNR, novel=novel,
                  lidar=dict(counts=counts, tags="carla"), scene_points=n_points, metrics=metrics,
                  ckpt=dict(name=os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))), iterations=CK["args"].get("iters"),
                            elapsed_s=CK.get("elapsed_s"), vram_peak_mb=CK.get("vram_peak_mb")))
json.dump(scene_json, open(os.path.join(args.out, "scene.json"), "w"), separators=(",", ":"))
print("done: %s, %.0f s" % (args.out, time.time() - t_start), flush=True)
os._exit(0)
