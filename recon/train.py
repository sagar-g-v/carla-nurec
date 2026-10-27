"""Fit the composite 3DGUT scene (model.py) to a converted CARLA recording (convert/to_dataset.py), on one GPU.
Train views: every camera of every tick except the held-out ticks (dataset split). Images at 1/FACTOR resolution.
Loss: (1 - 0.2) L1 + 0.2 (1 - SSIM) on non-ego pixels, + 0.1 relative depth error at lidar points, + 0.05 BCE of
alpha against the non-sky mask, + MCMC opacity / scale regularisers. Densification: gsplat MCMC per Gaussian group.
usage: train.py DATASET_DIR OUT_DIR [--iters 30000] [--factor 2] [--cap-bg 600000] [--cap-actor 25000]
                [--exposure] [--seed 0]"""
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.strategy import MCMCStrategy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from model import Scene, init_group, BoxDelta
import carla_geom as G

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--iters", type=int, default=30000); ap.add_argument("--factor", type=int, default=2)
ap.add_argument("--cap-bg", type=int, default=600000); ap.add_argument("--cap-actor", type=int, default=25000)
ap.add_argument("--init-bg", type=int, default=300000); ap.add_argument("--exposure", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=200)
ap.add_argument("--box-noise", type=str, default="", help="T_SIGMA_M,YAW_SIGMA_DEG: perturb the actor boxes per frame")
ap.add_argument("--refine-boxes", action="store_true", help="learn per-frame box corrections")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
D = json.load(open(os.path.join(args.data, "dataset.json")))
imap = json.load(open(os.path.join(D["source"], "instance_map.json")))
ego_id = D["ego"]["id"]
inst2actor = {int(k): v["actor"] for k, v in imap["camera"].items()}
ego_inst = [k for k, a in inst2actor.items() if a == ego_id]
W, H = D["res"][0] // args.factor, D["res"][1] // args.factor
cams = list(D["cameras"])
K = torch.tensor(D["cameras"][cams[0]]["K"], dtype=torch.float32)
K[:2] /= args.factor
# a local origin near the cameras keeps float32 precise
center = np.mean([np.array(c["T_world_cam"])[:3, 3] for f in D["frames"] for c in f["cams"].values()], axis=0)
SHIFT = np.eye(4); SHIFT[:3, 3] = -center
cam_pos = np.array([np.array(c["T_world_cam"])[:3, 3] for f in D["frames"] for c in f["cams"].values()]) - center
scene_scale = float(np.linalg.norm(cam_pos - cam_pos.mean(0), axis=1).max() * 1.1)
print("images %dx%d, scene scale %.1f m, ego instance ids %s" % (W, H, scene_scale, ego_inst), flush=True)

# ---------------------------------------------------------------- images, masks, lidar depth
views = [(f["tick"], c, f["cams"][c]) for f in D["frames"] for c in cams if f["cams"][c]["split"] == "train"]


def load(v):
    tick, c, cam = v
    rgb = Image.open(cam["rgb"]).convert("RGB").resize((W, H), Image.BOX)
    tag, iid = G.decode_inst(np.array(Image.open(cam["inst"]).convert("RGB")))
    ego = np.isin(iid, ego_inst)
    # a pixel is ego if most of its full-resolution block is ego (same for sky)
    blk = lambda m: m.reshape(H, args.factor, W, args.factor).mean((1, 3)) > 0.5
    return np.array(rgb), blk(ego), blk(tag == 11)


t0 = time.time()
with ThreadPoolExecutor(12) as ex:
    data = list(ex.map(load, views))
RGB = torch.tensor(np.stack([d[0] for d in data]))                     # uint8 (N, H, W, 3), CPU
EGO = torch.tensor(np.stack([d[1] for d in data]))
SKY = torch.tensor(np.stack([d[2] for d in data]))
print("loaded %d train views in %.0f s; ego pixels %.2f %%, sky %.1f %%" % (len(views), time.time() - t0,
      100 * EGO.float().mean(), 100 * SKY.float().mean()), flush=True)
SEML = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"), ("idx", "<u4"), ("tag", "<u4")])
frame_of = {f["tick"]: f for f in D["frames"]}
lidar_cache = {}


def lidar_world(tick):
    if tick not in lidar_cache:
        fr = frame_of[tick]
        pts = np.frombuffer(open(fr["lidar"]["semlidar"], "rb").read(), SEML)
        P = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64) @ np.diag(D["conventions"]["lidar_axes"]).T * [1, -1, 1]
        T = np.array(fr["lidar"]["T_world_lidar"])
        Pw = P @ T[:3, :3].T + T[:3, 3]
        lidar_cache[tick] = Pw[pts["idx"] != ego_id]
    return lidar_cache[tick]


def sparse_depth(v, n=4000):
    tick, c, cam = v
    Tcw = np.linalg.inv(np.array(cam["T_world_cam"]))
    Pc = lidar_world(tick) @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = Pc[:, 2]
    ok = (z > 1.0) & (z < 80.0)
    u = K[0, 0].item() * Pc[ok, 0] / z[ok] + K[0, 2].item(); vv = K[1, 1].item() * Pc[ok, 1] / z[ok] + K[1, 2].item()
    inb = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    s = np.stack([u[inb], vv[inb], z[ok][inb]], -1)
    if len(s) > n:
        s = s[np.random.choice(len(s), n, replace=False)]
    return torch.tensor(s, dtype=torch.float32)


LID = [sparse_depth(v) for v in views]
print("lidar depth targets: median %d points per view" % int(np.median([len(l) for l in LID])), flush=True)

# ---------------------------------------------------------------- model
Pst = np.load(os.path.join(args.data, "points_static.npy"))
if len(Pst) > args.init_bg:
    Pst = Pst[np.random.choice(len(Pst), args.init_bg, replace=False)]
far_n = 20000
dirs = np.random.randn(far_n, 3); dirs[:, 2] = np.abs(dirs[:, 2]) * 0.6; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
far = dirs * np.random.uniform(250, 500, (far_n, 1))
P_bg = np.concatenate([Pst[:, :3] - center, far])
C_bg = np.concatenate([Pst[:, 3:6] / 255.0, np.full((far_n, 3), 0.6)])
groups = {"bg": init_group(P_bg, C_bg)}
tracks, caps = {}, {"bg": args.cap_bg}
for aid in D["dynamic_ids"]:
    a = D["actors"][str(aid)]
    P = np.load(os.path.join(args.data, "actor_points", "%d.npy" % aid)).astype(np.float64)
    L, Wd, Hh = a["size"]
    if len(P) < 1500:                                                    # pad sparse actors with box samples
        R = np.random.uniform(-0.5, 0.5, (1500 - len(P), 3)) * [L, Wd, Hh]
        P = np.concatenate([P, R]) if len(P) else R
    groups[str(aid)] = init_group(P, np.full((len(P), 3), 0.5), opacity=0.1)
    tracks[str(aid)] = {k: (SHIFT @ np.array(T)).tolist() for k, T in a["track"].items()}
    caps[str(aid)] = args.cap_actor if a["kind"] == "car" else args.cap_actor // 2
gt_tracks = {a: dict(t) for a, t in tracks.items()}
if args.box_noise:                                                     # a noisy tracker: independent per frame
    st, sy = (float(v) for v in args.box_noise.split(","))
    rng = np.random.default_rng(args.seed + 1)
    for a, tr in tracks.items():
        for k in tr:
            T = np.array(tr[k]); e = rng.normal(0, st, 2); y = np.radians(rng.normal(0, sy))
            Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            T[:3, :3] = Rz @ T[:3, :3]; T[:2, 3] += e
            tr[k] = T.tolist()
scene = Scene(groups, tracks).to(dev)
if args.refine_boxes:
    scene.box_delta = BoxDelta([(a, int(k)) for a, tr in tracks.items() for k in tr]).to(dev)
    box_opt = torch.optim.Adam(scene.box_delta.parameters(), lr=1e-3)
print("initial Gaussians:", {n: len(g["means"]) for n, g in scene.groups.items()}, flush=True)

class SafeMCMC(MCMCStrategy):
    """MCMC that skips relocation / sampling for a group without live Gaussians (multinomial of zeros fails)."""

    def _relocate_gs(self, params, optimizers, binoms):
        if (torch.sigmoid(params["opacities"].flatten()) > self.min_opacity).sum() == 0:
            return 0
        return super()._relocate_gs(params, optimizers, binoms)

    def _add_new_gs(self, params, optimizers, binoms):
        if (torch.sigmoid(params["opacities"].flatten()) > self.min_opacity).sum() == 0:
            return 0
        return super()._add_new_gs(params, optimizers, binoms)


LR = dict(scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
opt, strat, state = {}, {}, {}
for n, g in scene.groups.items():
    mlr = 1.6e-4 * (scene_scale if n == "bg" else 5.0)
    opt[n] = {k: torch.optim.Adam([g[k]], lr=(mlr if k == "means" else LR[k]), eps=1e-15) for k in g.keys()}
    strat[n] = SafeMCMC(cap_max=caps[n], refine_start_iter=500, refine_stop_iter=int(args.iters * 0.8),
                            refine_every=100, min_opacity=0.005)
    strat[n].check_sanity(g, opt[n]); state[n] = strat[n].initialize_state()
mlr0 = {n: o["means"].param_groups[0]["lr"] for n, o in opt.items()}
sky_opt = torch.optim.Adam(scene.sky.parameters(), lr=1e-2)
if args.exposure:
    expo = torch.nn.Parameter(torch.zeros(len(views), 3, 4, device=dev))   # affine residual per train image
    expo_opt = torch.optim.Adam([expo], lr=1e-3)


def ssim(a, b):
    """mean SSIM of (3, H, W) images, 11x11 Gaussian window."""
    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x[None], w, padding=5, groups=3)[0]
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()


Tcw_all = [torch.tensor(SHIFT @ np.array(v[2]["T_world_cam"]), dtype=torch.float32) for v in views]
log = open(os.path.join(args.out, "train_log.jsonl"), "w")
t_start, hist = time.time(), []
torch.cuda.reset_peak_memory_stats()
for it in range(1, args.iters + 1):
    i = np.random.randint(len(views))
    tick = views[i][0]
    sh = min(it // 1000, 3)
    gt = RGB[i].to(dev).float() / 255.0
    ego, sky = EGO[i].to(dev), SKY[i].to(dev)
    rgb, depth, alpha, info, owner = scene.render(Tcw_all[i], K, W, H, tick, sh_degree=sh)
    if args.exposure:
        A = torch.eye(3, 4, device=dev) + expo[i]
        rgb = rgb @ A[:, :3].T + A[:, 3]
    valid = (~ego).float()[..., None]
    rgb_m = rgb * valid + gt * (1 - valid)
    l1 = (torch.abs(rgb - gt) * valid).sum() / (valid.sum() * 3 + 1)
    loss = 0.8 * l1 + 0.2 * (1 - ssim(rgb_m.permute(2, 0, 1), gt.permute(2, 0, 1)))
    lid = LID[i].to(dev)
    if len(lid):
        u, v = lid[:, 0].long().clamp(0, W - 1), lid[:, 1].long().clamp(0, H - 1)
        keep = ~ego[v, u]
        dz = depth[v[keep], u[keep]]
        loss = loss + 0.1 * (torch.abs(dz - lid[keep, 2]) / lid[keep, 2]).mean()
    a = alpha.clamp(1e-4, 1 - 1e-4)
    target = (~sky).float()
    loss = loss + 0.05 * F.binary_cross_entropy(a[~ego], target[~ego])
    present = {name for name, _ in owner}                              # MCMC regularisers on the groups in this frame
    for n in present:
        g = scene.groups[n]
        loss = loss + 0.01 * torch.sigmoid(g["opacities"]).mean() + 0.01 * torch.exp(g["scales"]).mean()
    if args.exposure:
        loss = loss + 1e-2 * (expo[i] ** 2).sum()
    loss.backward()
    frac = it / args.iters
    for n, g in scene.groups.items():
        lr_m = mlr0[n] * (0.01 ** frac)
        for k, o in opt[n].items():
            if k == "means":
                o.param_groups[0]["lr"] = lr_m
            o.step(); o.zero_grad(set_to_none=True)
        strat[n].step_post_backward(g, opt[n], state[n], it, info, lr=lr_m)
    sky_opt.step(); sky_opt.zero_grad(set_to_none=True)
    if args.refine_boxes and it > 500:
        box_opt.step()
    if args.refine_boxes:
        box_opt.zero_grad(set_to_none=True)
    if args.exposure:
        expo_opt.step(); expo_opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        mse = (((rgb - gt) ** 2) * valid).sum() / (valid.sum() * 3 + 1)
        hist.append(-10 * math.log10(max(mse.item(), 1e-10)))
    if it % args.log_every == 0 or it == args.iters:
        n_g = {n: len(g["means"]) for n, g in scene.groups.items()}
        row = dict(it=it, loss=round(loss.item(), 5), psnr_train=round(float(np.mean(hist[-args.log_every:])), 3),
                   n_bg=n_g["bg"], n_actors=sum(v for k, v in n_g.items() if k != "bg"),
                   it_s=round(it / (time.time() - t_start), 2), vram_peak_mb=round(torch.cuda.max_memory_allocated() / 1e6),
                   vram_reserved_mb=round(torch.cuda.memory_reserved() / 1e6), elapsed_s=round(time.time() - t_start))
        log.write(json.dumps(row) + "\n"); log.flush()
        print(row, flush=True)
ckpt = dict(groups={n: {k: v.detach().cpu() for k, v in g.items()} for n, g in scene.groups.items()},
            sky=scene.sky.state_dict(), tracks=tracks, center=center.tolist(), factor=args.factor, W=W, H=H, K=K.tolist(),
            args=vars(args), elapsed_s=time.time() - t_start, vram_peak_mb=torch.cuda.max_memory_allocated() / 1e6,
            views=[(t, c) for t, c, _ in views], exposure=(expo.detach().cpu() if args.exposure else None),
            gt_tracks=gt_tracks, box_delta=(dict(index={"%s|%d" % k: i for k, i in scene.box_delta.index.items()},
                                                 d=scene.box_delta.d.detach().cpu()) if args.refine_boxes else None))
torch.save(ckpt, os.path.join(args.out, "ckpt.pt"))
print("saved", os.path.join(args.out, "ckpt.pt"), "in %.0f s" % (time.time() - t_start), flush=True)
