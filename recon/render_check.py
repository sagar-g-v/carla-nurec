"""Render one dataset view from a checkpoint next to the ground truth (quick visual check).
usage: render_check.py DATASET_DIR CKPT TICK CAM OUT_PNG"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Scene

D = json.load(open(os.path.join(sys.argv[1], "dataset.json")))
CK = torch.load(sys.argv[2], map_location="cpu")
tick, cam = int(sys.argv[3]), sys.argv[4]
groups = {n: torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in g.items()}) for n, g in CK["groups"].items()}
scene = Scene(groups, CK["tracks"]).cuda(); scene.sky.load_state_dict(CK["sky"])
SHIFT = np.eye(4); SHIFT[:3, 3] = -np.array(CK["center"])
fr = [f for f in D["frames"] if f["tick"] == tick][0]
T = torch.tensor(SHIFT @ np.array(fr["cams"][cam]["T_world_cam"]), dtype=torch.float32)
with torch.no_grad():
    rgb, depth, alpha, _, _ = scene.render(T, torch.tensor(CK["K"]), CK["W"], CK["H"], tick)
gt = Image.open(fr["cams"][cam]["rgb"]).convert("RGB").resize((CK["W"], CK["H"]), Image.BOX)
out = Image.new("RGB", (CK["W"] * 2, CK["H"]))
out.paste(gt, (0, 0)); out.paste(Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)), (CK["W"], 0))
out.save(sys.argv[5])
print("split", fr["cams"][cam]["split"], "alpha mean %.3f" % alpha.mean().item())
