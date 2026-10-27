"""Debug one actor: find held-out views where CARLA's instance mask shows it, render the reconstruction there and save
GT | render | GT mask (green) vs reconstructed owner mask (red) crops. usage: debug_actor.py DATASET CKPT NAME OUT_DIR"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recon")); sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from model import Scene
import carla_geom as G
from gsplat import rasterization

DS, CKPT, NAME, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(DS, "dataset.json")))
CK = torch.load(CKPT, map_location="cpu", weights_only=False)
imap = json.load(open(os.path.join(D["source"], "instance_map.json")))
merged = {int(k): v for k, v in D.get("merged", {}).items()}
inst2actor = {int(k): merged.get(v["actor"], v["actor"]) for k, v in imap["camera"].items()}
aid = [int(k) for k, a in D["actors"].items() if a["name"] == NAME][0]
groups = {n: torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in g.items()}) for n, g in CK["groups"].items()}
scene = Scene(groups, CK["tracks"]).cuda(); scene.sky.load_state_dict(CK["sky"])
gn = list(scene.groups.keys()); gi = gn.index(str(aid))
SHIFT = np.eye(4); SHIFT[:3, 3] = -np.array(CK["center"])
W, H, K, FA = CK["W"], CK["H"], torch.tensor(CK["K"]), CK["factor"]
g = CK["groups"][str(aid)]
op = torch.sigmoid(g["opacities"])
print(NAME, "gaussians", len(op), "opacity mean %.3f, > 0.5: %d" % (op.mean(), (op > 0.5).sum()))
n = 0
for f in D["frames"]:
    for c, cam in f["cams"].items():
        if cam["split"] != "test":
            continue
        tag, iid = G.decode_inst(np.array(Image.open(cam["inst"]).convert("RGB")))
        m = np.isin(iid, [k for k, a in inst2actor.items() if a == aid])
        if m.sum() < 600:
            continue
        T = torch.tensor(SHIFT @ np.array(cam["T_world_cam"]), dtype=torch.float32)
        with torch.no_grad():
            rgb, depth, alpha, info, owner = scene.render(T, K, W, H, f["tick"])
            (means, quats, scales, opac, sh), own = scene.gather(f["tick"])
            ids = torch.cat([torch.full((k,), gn.index(nm), device="cuda") for nm, k in own])
            acc, _, _ = rasterization(means, quats, torch.exp(scales), torch.sigmoid(opac), F.one_hot(ids, len(gn)).float(),
                                      torch.linalg.inv(T.cuda())[None], K.cuda()[None], W, H, sh_degree=None, packed=False,
                                      with_ut=True, with_eval3d=True, near_plane=0.2, far_plane=2000.0)
            mine = acc[0][..., gi].cpu().numpy()
        gt = np.array(Image.open(cam["rgb"]).convert("RGB").resize((W, H), Image.BOX))
        md = m[FA // 2::FA, FA // 2::FA]
        ov = gt.copy(); ov[md] = (0.5 * ov[md] + [0, 127, 0]).astype(np.uint8); ov[mine > 0.3] = (0.5 * ov[mine > 0.3] + [127, 0, 0]).astype(np.uint8)
        ys, xs = np.nonzero(md)
        y0, y1, x0, x1 = max(ys.min() - 30, 0), min(ys.max() + 30, H), max(xs.min() - 30, 0), min(xs.max() + 30, W)
        r8 = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        crop = np.concatenate([gt[y0:y1, x0:x1], r8[y0:y1, x0:x1], ov[y0:y1, x0:x1]], 1)
        Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.NEAREST).save(os.path.join(OUT, "%s_%04d_%s.png" % (NAME, f["tick"], c)))
        print("tick", f["tick"], c, "GT px", int(md.sum()), "recon owner-weight max %.2f, px > 0.3: %d" % (mine.max(), (mine > 0.3).sum()))
        n += 1
        if n >= 3:
            sys.exit(0)
