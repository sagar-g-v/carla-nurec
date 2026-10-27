"""Evaluate a fitted scene (recon/train.py checkpoint) against CARLA ground truth.
  1 held-out on-path views (dataset split 'test', all 6 cameras): PSNR / SSIM (ego pixels excluded), LPIPS-alex
    (ego pixels copied from the ground truth), depth vs CARLA depth
  2 novel views from the recorder replay (front camera shifted -3..+3 m sideways): the same metrics; views whose
    camera sits inside an actor box are skipped; offset 0 vs the live recording = the replay's own noise floor
  3 geometry vs lidar: rendered depth at the lidar points of held-out frames; chamfer / F-score between the
    rendered-depth point cloud and the lidar points (both within 50 m of the ego)
  4 dynamic actors: per-actor masks rendered with occlusion (group-id channels) vs CARLA instance masks (IoU);
    actor Gaussian extent vs the ground-truth box size
usage: evaluate.py DATASET_DIR CKPT REPLAY_DIR OUT_DIR [--panels 6]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recon")); sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from model import Scene, BoxDelta
import carla_geom as G
from gsplat import rasterization

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("ckpt"); ap.add_argument("replay"); ap.add_argument("out")
ap.add_argument("--panels", type=int, default=6)
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
dev = "cuda"
D = json.load(open(os.path.join(args.data, "dataset.json")))
CK = torch.load(args.ckpt, map_location="cpu")
imap = json.load(open(os.path.join(D["source"], "instance_map.json")))
inst2actor = {int(k): v["actor"] for k, v in imap["camera"].items()}
ego_id = D["ego"]["id"]
rider_of = {int(k): v for k, v in D.get("merged", {}).items()}            # rider walker id -> its bike (cyclist)
FACT, W, H = CK["factor"], CK["W"], CK["H"]
K = torch.tensor(CK["K"])
center = np.array(CK["center"]); SHIFT = np.eye(4); SHIFT[:3, 3] = -center
groups = {}
for n, g in CK["groups"].items():
    groups[n] = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in g.items()})
scene = Scene(groups, CK["tracks"]).to(dev)
scene.sky.load_state_dict(CK["sky"])
if CK.get("box_delta"):                                                    # noisy-box run with learned corrections
    keys = sorted(CK["box_delta"]["index"].items(), key=lambda kv: kv[1])
    scene.box_delta = BoxDelta([(k.split("|")[0], int(k.split("|")[1])) for k, _ in keys]).to(dev)
    scene.box_delta.d.data = CK["box_delta"]["d"].to(dev)
scene.eval()
gnames = list(scene.groups.keys())
import lpips
LP = lpips.LPIPS(net="alex", verbose=False).to(dev)


def ssim_map(a, b):
    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x[None], w, padding=5, groups=3)[0]
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean(0)


def load_gt(rgb_p, depth_p, inst_p):
    rgb = np.array(Image.open(rgb_p).convert("RGB").resize((W, H), Image.BOX)).astype(np.float32) / 255.0
    dep = G.decode_depth(np.array(Image.open(depth_p).convert("RGB")))
    tag, iid = G.decode_inst(np.array(Image.open(inst_p).convert("RGB")))
    blk = lambda m: m.reshape(H, FACT, W, FACT).mean((1, 3))
    dep_d = np.median(dep.reshape(H, FACT, W, FACT).transpose(0, 2, 1, 3).reshape(H, W, -1), -1)
    act = np.vectorize(lambda i: inst2actor.get(int(i), -1), otypes=[np.int64])(iid)
    act = np.vectorize(lambda a: rider_of.get(int(a), a), otypes=[np.int64])(act) if rider_of else act
    act_d = act[FACT // 2::FACT, FACT // 2::FACT]
    return rgb, dep_d, blk(tag == 11) > 0.5, blk(act == ego_id) > 0.5, act_d


@torch.no_grad()
def render_all(T_world_cam, tick):
    T = torch.tensor(SHIFT @ T_world_cam, dtype=torch.float32)
    rgb, depth, alpha, info, owner = scene.render(T, K, W, H, tick)
    # occlusion-aware owner map: render one-hot group ids with the same Gaussians
    (means, quats, scales, opac, sh), own = scene.gather(tick)
    ids = torch.cat([torch.full((n,), gnames.index(name), device=dev) for name, n in own])
    onehot = F.one_hot(ids, len(gnames)).float()
    viewmat = torch.linalg.inv(T.to(dev))[None]
    acc, _, _ = rasterization(means, quats, torch.exp(scales), torch.sigmoid(opac), onehot, viewmat, K.to(dev)[None], W, H,
                              sh_degree=None, packed=False, with_ut=True, with_eval3d=True, near_plane=0.2, far_plane=2000.0)
    owner_map = torch.where(acc[0].max(-1).values > 0.5, acc[0].argmax(-1), torch.full_like(acc[0][..., 0], -1, dtype=torch.long))
    return rgb.clamp(0, 1), depth, alpha, owner_map


def metrics(pred, gt, mask_ok):
    """pred, gt (H, W, 3) tensors in [0, 1]; mask_ok (H, W) bool: pixels that count."""
    m = mask_ok.float()[..., None]
    mse = (((pred - gt) ** 2) * m).sum() / (m.sum() * 3)
    psnr = -10 * math.log10(max(mse.item(), 1e-10))
    s = (ssim_map(pred.permute(2, 0, 1), gt.permute(2, 0, 1)) * mask_ok.float()).sum() / mask_ok.float().sum()
    p2 = pred * m + gt * (1 - m)
    lp = LP(p2.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item()
    return psnr, s.item(), lp


def align(pred, gt, mask_ok):
    """per-image affine colour fit (3 x 4, least squares on the counted pixels): removes the exposure / white balance
    difference of an image whose exposure the model never saw (NeRF-W style "appearance-aligned" evaluation)."""
    X = torch.cat([pred[mask_ok], torch.ones_like(pred[mask_ok][:, :1])], 1)
    A = torch.linalg.lstsq(X, gt[mask_ok]).solution
    return (torch.cat([pred, torch.ones_like(pred[..., :1])], -1) @ A).clamp(0, 1)


def depth_metrics(dr, dg, ok):
    dr, dg = dr[ok], dg[ok]
    if len(dg) == 0:
        return None
    rel = np.abs(dr - dg) / dg
    ratio = np.maximum(dr / dg, dg / np.maximum(dr, 1e-6))
    return dict(absrel=float(rel.mean()), rmse=float(np.sqrt(((dr - dg) ** 2).mean())), median_abs=float(np.median(np.abs(dr - dg))),
                d125=float((ratio < 1.25).mean()), n=int(len(dg)))


def iou_table(owner_map, act_gt, rows):
    for gi, name in enumerate(gnames):
        if name == "bg":
            continue
        a = int(name)
        g = act_gt == a
        if g.sum() < 150:
            continue
        p = owner_map == gi
        rows.setdefault(D["actors"][name]["name"], []).append(float((p & g).sum() / max((p | g).sum(), 1)))


def panel(path, gt, pred, dg, dr):
    err = np.abs(pred - gt).mean(-1)
    to8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
    dv = lambda d: to8(1 - np.clip(np.log(np.maximum(d, 1)) / np.log(100), 0, 1))
    top = np.concatenate([to8(gt), to8(pred)], 1)
    bot = np.concatenate([np.repeat(to8(err * 4)[..., None], 3, -1), np.repeat(dv(dr)[..., None], 3, -1)], 1)
    Image.fromarray(np.concatenate([top, bot], 0)).save(path)


R = dict(ckpt=os.path.abspath(args.ckpt), train=dict(elapsed_s=CK["elapsed_s"], vram_peak_mb=CK["vram_peak_mb"],
         gaussians={n: len(g["means"]) for n, g in CK["groups"].items()}, res=[W, H], args=CK["args"]))
t0 = time.time()
# ---------------------------------------------------------------- 1 held-out on-path views
test = [(f["tick"], c, f["cams"][c]) for f in D["frames"] for c in D["cameras"] if f["cams"][c]["split"] == "test"]
rows, drows, iou_rows, panels = [], [], {}, 0
for tick, c, cam in test:
    gt, dg, sky, ego, act = load_gt(cam["rgb"], cam["depth"], cam["inst"])
    T = np.array(cam["T_world_cam"])
    rgb, depth, alpha, owner_map = render_all(T, tick)
    ok = torch.tensor(~ego, device=dev)
    gtt = torch.tensor(gt, device=dev)
    p, s, l = metrics(rgb, gtt, ok)
    pa, sa, la = metrics(align(rgb, gtt, ok), gtt, ok)
    rows.append(dict(tick=tick, cam=c, psnr=p, ssim=s, lpips=l, psnr_al=pa, ssim_al=sa, lpips_al=la))
    dr = depth.cpu().numpy()
    dm = depth_metrics(dr, dg, (~sky) & (~ego) & (dg < 80) & (alpha.cpu().numpy() > 0.5))
    if dm:
        drows.append(dm)
    iou_table(owner_map.cpu().numpy(), act, iou_rows)
    if c == "CAM_FRONT" and panels < args.panels:
        panel(os.path.join(args.out, "img", "test_%04d_%s.png" % (tick, c)), gt, rgb.cpu().numpy(), dg, dr); panels += 1
agg = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else None
R["heldout"] = dict(n=len(rows), psnr=agg(rows, "psnr"), ssim=agg(rows, "ssim"), lpips=agg(rows, "lpips"),
                    aligned=dict(psnr=agg(rows, "psnr_al"), ssim=agg(rows, "ssim_al"), lpips=agg(rows, "lpips_al")),
                    per_camera={c: dict(psnr=agg([r for r in rows if r["cam"] == c], "psnr"), ssim=agg([r for r in rows if r["cam"] == c], "ssim"),
                                        lpips=agg([r for r in rows if r["cam"] == c], "lpips")) for c in D["cameras"]},
                    depth={k: agg(drows, k) for k in ("absrel", "rmse", "median_abs", "d125")})
print("held-out:", {k: R["heldout"][k] for k in ("n", "psnr", "ssim", "lpips")}, R["heldout"]["depth"], flush=True)

# ---------------------------------------------------------------- 2 novel views from the replay
RP = json.load(open(os.path.join(args.replay, "replay.json")))
frames_by_tick = {f["tick"]: f for f in D["frames"]}
src = D["source"]
live_frames = {json.loads(l)["tick"]: json.loads(l) for l in open(os.path.join(src, "frames.jsonl"))}


def inside_actor(cam_tf, tick):
    p = G.pose_rh(cam_tf, D["conventions"]["rotation"])[:3, 3]
    for a in live_frames[tick]["actors"]:
        T = G.pose_rh(a["tf"], D["conventions"]["rotation"]) @ G.pose_rh(a["bbox"][:3] + a["bbox"][6:9], D["conventions"]["rotation"])
        L = np.linalg.inv(T) @ np.append(p, 1.0)
        if np.all(np.abs(L[:3]) <= np.array(a["bbox"][3:6]) + 0.3):
            return a["name"]
    return None


novel, skipped, nd, floor = {}, [], {}, []
for fr in RP["frames"]:
    k = fr["frame"]
    for off in RP["offsets"]:
        key = "%+.0f" % off
        tf = fr["cams"][key + "/rgb"]
        hit = inside_actor(tf, k)
        if hit:
            skipped.append(dict(frame=k, offset=off, inside=hit)); continue
        base = os.path.join(args.replay, "novel", "off_" + key)
        gt, dg, sky, ego, act = load_gt(os.path.join(base, "rgb", "%04d.png" % k), os.path.join(base, "depth", "%04d.png" % k),
                                        os.path.join(base, "inst", "%04d.png" % k))
        T = G.cam_to_world_cv(tf, D["conventions"]["rotation"])
        rgb, depth, alpha, owner_map = render_all(T, k)
        ok = torch.ones(H, W, dtype=torch.bool, device=dev)
        gtt = torch.tensor(gt, device=dev)
        p, s, l = metrics(rgb, gtt, ok)
        pa, sa, la = metrics(align(rgb, gtt, ok), gtt, ok)
        novel.setdefault(key, []).append(dict(frame=k, psnr=p, ssim=s, lpips=l, psnr_al=pa, ssim_al=sa, lpips_al=la))
        dm = depth_metrics(depth.cpu().numpy(), dg, (~sky) & (dg < 80) & (alpha.cpu().numpy() > 0.5))
        if dm:
            nd.setdefault(key, []).append(dm)
        iou_table(owner_map.cpu().numpy(), act, iou_rows)
        if off == 0 and k in frames_by_tick:
            # replay noise floor: replayed render (ego hidden) vs the live front camera, ego pixels excluded
            lc = frames_by_tick[k]["cams"]["CAM_FRONT"]
            lgt, _, _, lego, _ = load_gt(lc["rgb"], lc["depth"], lc["inst"])
            fp, fs, fl = metrics(torch.tensor(gt, device=dev), torch.tensor(lgt, device=dev), torch.tensor(~lego, device=dev))
            floor.append(dict(frame=k, psnr=fp, ssim=fs, lpips=fl))
        if k in (40, 100, 160) and off in (-3, -1, 1, 3):
            panel(os.path.join(args.out, "img", "novel_%04d_off%s.png" % (k, key)), gt, rgb.cpu().numpy(), dg, depth.cpu().numpy())
R["novel"] = {key: dict(n=len(v), psnr=agg(v, "psnr"), ssim=agg(v, "ssim"), lpips=agg(v, "lpips"),
                        aligned=dict(psnr=agg(v, "psnr_al"), ssim=agg(v, "ssim_al"), lpips=agg(v, "lpips_al")),
                        depth={m: agg(nd.get(key, []), m) for m in ("absrel", "rmse", "median_abs", "d125")}) for key, v in sorted(novel.items(), key=lambda kv: float(kv[0]))}
R["novel_skipped"] = skipped
R["replay_noise_floor"] = dict(n=len(floor), psnr=agg(floor, "psnr"), ssim=agg(floor, "ssim"), lpips=agg(floor, "lpips"),
                               replay_pose_error=RP["replay_pose_error"], rider_posed=RP.get("rider_posed"))
print("novel:", {k: (v["n"], round(v["psnr"], 2), round(v["ssim"], 3), round(v["lpips"], 3)) for k, v in R["novel"].items()}, flush=True)
print("replay noise floor:", R["replay_noise_floor"]["psnr"], flush=True)

# ---------------------------------------------------------------- 3 geometry vs lidar
SEML = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"), ("idx", "<u4"), ("tag", "<u4")])
lid_err, rec_pts, gt_pts = [], [], []
test_ticks = sorted({t for t, _, _ in test})
for tick in test_ticks:
    fr = frames_by_tick[tick]
    pts = np.frombuffer(open(fr["lidar"]["semlidar"], "rb").read(), SEML)
    P = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64) * [1, -1, 1]
    Tl = np.array(fr["lidar"]["T_world_lidar"])
    Pw = (P @ Tl[:3, :3].T + Tl[:3, 3])[(pts["idx"] != ego_id) & (np.linalg.norm(P, axis=1) < 50)]
    gt_pts.append(Pw)
    for c in D["cameras"]:
        cam = fr["cams"][c]
        T = np.array(cam["T_world_cam"])
        rgb, depth, alpha, _ = render_all(T, tick)
        d = depth.cpu().numpy(); a = alpha.cpu().numpy()
        Tcw = np.linalg.inv(T)
        Pc = Pw @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = Pc[:, 2]; okz = z > 1.0
        u = K[0, 0].item() * Pc[okz, 0] / z[okz] + K[0, 2].item(); v = K[1, 1].item() * Pc[okz, 1] / z[okz] + K[1, 2].item()
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui, vi, zz = u[inb].astype(int), v[inb].astype(int), z[okz][inb]
        lid_err.append(np.abs(d[vi, ui] - zz))
        # rendered surface points (alpha > 0.5, within 50 m)
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        m = (a > 0.5) & (d < 50) & (d > 0.5)
        rays = np.stack([(uu - K[0, 2].item()) / K[0, 0].item(), (vv - K[1, 2].item()) / K[1, 1].item(), np.ones_like(uu)], -1)
        Pr = (rays * d[..., None])[m]
        rec_pts.append(Pr @ T[:3, :3].T + T[:3, 3])
le = np.concatenate(lid_err)
A = np.concatenate(rec_pts); B = np.concatenate(gt_pts)
vox = lambda P, s: P[np.unique(np.floor(P / s).astype(np.int64), axis=0, return_index=True)[1]]
A, B = vox(A, 0.05), vox(B, 0.05)
da, _ = cKDTree(B).query(A); db, _ = cKDTree(A).query(B)
fs = lambda t: float(2 * (da < t).mean() * (db < t).mean() / max((da < t).mean() + (db < t).mean(), 1e-9))
R["geometry"] = dict(lidar_depth_abs_err=dict(median=float(np.median(le)), mean=float(le.mean()), within_0_5m=float((le < 0.5).mean()),
                                              within_1m=float((le < 1.0).mean()), n=int(len(le))),
                     chamfer=dict(accuracy_mean=float(da.mean()), accuracy_median=float(np.median(da)), completeness_mean=float(db.mean()),
                                  completeness_median=float(np.median(db)), chamfer_mean=float((da.mean() + db.mean()) / 2),
                                  fscore_0_2m=fs(0.2), fscore_0_5m=fs(0.5), n_rec=int(len(A)), n_lidar=int(len(B))))
print("geometry:", R["geometry"], flush=True)

# ---------------------------------------------------------------- 4 dynamic actors
act_geo = {}
for name in gnames:
    if name == "bg":
        continue
    g = CK["groups"][name]
    keep = torch.sigmoid(g["opacities"]) > 0.3
    P = g["means"][keep].numpy()
    a = D["actors"][name]
    if len(P) < 10:
        continue
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    act_geo[a["name"]] = dict(extent_rec=[round(float(v), 3) for v in hi - lo], box_size=[round(v, 3) for v in a["size"]],
                              centre_offset_m=round(float(np.linalg.norm((hi + lo) / 2)), 3), n_gauss=int(keep.sum()))
R["actors"] = dict(mask_iou={n: dict(mean=float(np.mean(v)), n=len(v)) for n, v in iou_rows.items()}, geometry=act_geo,
                   note="actor positions come from the ground-truth boxes (as NuRec's cuboid tracks); IoU measures how well "
                        "the reconstructed actor shape covers CARLA's instance mask, with occlusion")
if CK.get("box_delta"):
    def yaw(T):
        return math.degrees(math.atan2(T[1, 0], T[0, 0]))
    errs = {"given": [], "refined": []}
    for (name, tick), i in scene.box_delta.index.items():
        Tg = np.array(CK["gt_tracks"][name][str(tick)] if str(tick) in CK["gt_tracks"][name] else CK["gt_tracks"][name][tick])
        Tn = CK["tracks"][name][str(tick)] if str(tick) in CK["tracks"][name] else CK["tracks"][name][tick]
        Tn = torch.tensor(np.array(Tn), dtype=torch.float32, device=dev)
        Tr = scene.box_delta.apply(name, tick, Tn).detach().cpu().numpy()
        for key, T in (("given", Tn.cpu().numpy()), ("refined", Tr)):
            errs[key].append((float(np.linalg.norm(T[:2, 3] - Tg[:2, 3])), abs((yaw(T) - yaw(Tg) + 180) % 360 - 180)))
    R["box_refinement"] = {k: dict(pos_mean_m=float(np.mean([e[0] for e in v])), pos_median_m=float(np.median([e[0] for e in v])),
                                   yaw_mean_deg=float(np.mean([e[1] for e in v])), n=len(v)) for k, v in errs.items()}
    R["box_refinement"]["noise"] = CK["args"]["box_noise"]
    print("box refinement:", R["box_refinement"], flush=True)
R["eval_s"] = time.time() - t0
json.dump(R, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
print("actor IoU:", {n: round(v["mean"], 3) for n, v in R["actors"]["mask_iou"].items()}, flush=True)
print("done in %.0f s" % R["eval_s"])
