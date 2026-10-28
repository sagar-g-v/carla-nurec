"""Interactive viewer of a fitted scene (recon/train.py checkpoint) next to CARLA's ground truth, in a browser.
A local HTTP server renders the reconstruction on the GPU for the pose the page asks for: one of the six rig cameras
at a tick, then moved / turned freely from there. Where CARLA rendered exactly that pose (the rig cameras, or the front
camera at the re-simulated lateral offsets) the ground truth is shown beside it, with the PSNR of the view.
usage: serve.py DATASET_DIR CKPT [--novel NOVEL_DIR] [--port 8765]   then open http://127.0.0.1:8765/"""
import argparse, base64, io, json, math, os, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import torch
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recon")); sys.path.insert(0, os.path.join(HERE, "..", "convert"))
from model import Scene, BoxDelta
import carla_geom as G

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("ckpt")
ap.add_argument("--novel", help="re-simulation dir (replay.json, novel/) for the off-path ground truth")
ap.add_argument("--port", type=int, default=8765)
args = ap.parse_args()
dev = "cuda"
D = json.load(open(os.path.join(args.data, "dataset.json")))
CK = torch.load(args.ckpt, map_location="cpu", weights_only=False)
FACT, W, H = CK["factor"], CK["W"], CK["H"]
K = np.array(CK["K"])
SHIFT = np.eye(4); SHIFT[:3, 3] = -np.array(CK["center"])
groups = {n: torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in g.items()}) for n, g in CK["groups"].items()}
scene = Scene(groups, CK["tracks"]).to(dev)
scene.sky.load_state_dict(CK["sky"])
if CK.get("box_delta"):                                                    # noisy-box run with learned corrections
    keys = sorted(CK["box_delta"]["index"].items(), key=lambda kv: kv[1])
    scene.box_delta = BoxDelta([(k.split("|")[0], int(k.split("|")[1])) for k, _ in keys]).to(dev)
    scene.box_delta.d.data = CK["box_delta"]["d"].to(dev)
scene.eval()
FRAMES = {f["tick"]: f for f in D["frames"]}
TICKS = sorted(FRAMES)
CAMS = list(D["frames"][0]["cams"])
imap = json.load(open(os.path.join(D["source"], "instance_map.json")))
EGO_INST = [int(k) for k, v in imap["camera"].items() if v["actor"] == D["ego"]["id"]]
RP = json.load(open(os.path.join(args.novel, "replay.json"))) if args.novel else None
NOVEL = {fr["frame"]: fr for fr in RP["frames"]} if RP else {}
OFFSETS = RP["offsets"] if RP else []
BOXES = {n: (a["name"], np.array(a["size"]), a["track"]) for n, a in D["actors"].items()}
LOCK = threading.Lock()                                                    # one GPU render at a time


def delta(tx, ty, tz, yaw, pitch):
    """camera motion in the reference camera's frame (OpenCV axes): right / up / forward metres, then yaw (+ = left)
    about the camera's up axis and pitch (+ = up) about its right axis, in degrees."""
    a, b = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    T = np.eye(4); T[:3, :3] = Ry @ Rx; T[:3, 3] = [tx, -ty, tz]
    return T


def turbo(x):
    """Turbo colour map, polynomial approximation (x in [0, 1] -> RGB uint8)."""
    x = np.clip(x, 0, 1)
    P = np.stack([x ** i for i in range(6)], -1)
    c = np.stack([P @ np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943]),
                  P @ np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604]),
                  P @ np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])], -1)
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def depth_img(d, valid):
    """depth (m) -> colour: near red, 150 m blue (log scale); invalid (sky / empty) black."""
    x = 1.0 - np.log(np.clip(d, 1.0, 150.0)) / math.log(150.0)
    out = turbo(x); out[~valid] = 0
    return out


def draw_boxes(img, T_world_cam, Kr, tick):
    """ground-truth 3D boxes of all actors (dataset tracks), clipped at the near plane."""
    dr = ImageDraw.Draw(img)
    V = np.linalg.inv(T_world_cam)
    sx = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(sx[a] - sx[b]).sum() == 2]
    for name, size, track in BOXES.values():
        T = track.get(str(tick))
        if T is None:
            continue
        c = (V @ SHIFT @ np.array(T) @ np.c_[sx * size / 2, np.ones(8)].T).T[:, :3]
        if (c[:, 2] < 0.3).all() or np.linalg.norm(c.mean(0)) > 80:
            continue
        for a, b in edges:
            p, q = c[a], c[b]
            if p[2] < 0.3 and q[2] < 0.3:
                continue
            if p[2] < 0.3 or q[2] < 0.3:                                    # clip the edge at z = 0.3 m
                s = (0.3 - p[2]) / (q[2] - p[2]); m = p + s * (q - p)
                p, q = (m, q) if p[2] < 0.3 else (p, m)
            pu = Kr @ p / p[2]; qu = Kr @ q / q[2]
            dr.line([(pu[0], pu[1]), (qu[0], qu[1])], fill=(255, 220, 0), width=2)
        top = c[np.argmin(c[:, 1])]
        if top[2] > 0.3:
            u = Kr @ top / top[2]
            dr.text((u[0] + 3, u[1] - 12), name, fill=(255, 220, 0))


def jpeg(a):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ground_truth(cam, tick, ref, d, s):
    """(rgb, depth, ego mask, pose, label) of the CARLA image taken exactly at this pose, else None."""
    if ref != tick or any(abs(v) > 1e-6 for v in d[1:]):
        return None
    if abs(d[0]) < 1e-6:
        c = FRAMES[tick]["cams"][cam]
        paths, T = (c["rgb"], c["depth"], c["inst"]), SHIFT @ np.array(c["T_world_cam"])
        label = "CARLA %s, tick %d (%s view)" % (cam, tick, "held-out" if c["split"] == "test" else "training")
    elif cam == "CAM_FRONT" and tick in NOVEL and any(abs(d[0] - o) < 1e-6 for o in OFFSETS):
        key = "%+.0f" % d[0]
        base = os.path.join(args.novel, "novel", "off_" + key)
        paths = [os.path.join(base, m, "%04d.png" % tick) for m in ("rgb", "depth", "inst")]
        T = SHIFT @ G.cam_to_world_cv(NOVEL[tick]["cams"][key + "/rgb"], D["conventions"]["rotation"])
        label = "CARLA re-simulation, front camera %s m beside the driven path, tick %d" % (key, tick)
    else:
        return None
    w, h, f = W * s, H * s, FACT // s
    rgb = np.array(Image.open(paths[0]).convert("RGB").resize((w, h), Image.BOX))
    dep = G.decode_depth(np.array(Image.open(paths[1]).convert("RGB")))[f // 2::f, f // 2::f]
    tag, iid = G.decode_inst(np.array(Image.open(paths[2]).convert("RGB")))
    ego = np.isin(iid, EGO_INST) if abs(d[0]) < 1e-6 else np.zeros_like(iid, bool)   # re-simulation: no ego
    ego = ego.reshape(h, f, w, f).mean((1, 3)) > 0.5
    return rgb, dep, tag[f // 2::f, f // 2::f] == 11, ego, T, label


@torch.no_grad()
def render(q):
    cam, tick, ref = q["cam"], int(q["tick"]), int(q["ref"])
    d = [float(q[k]) for k in ("tx", "ty", "tz", "yaw", "pitch")]
    s = 2 if q.get("hq") == "1" else 1
    mode = q.get("mode", "rgb")
    gt = ground_truth(cam, tick, ref, d, s)
    T = gt[4] if gt else SHIFT @ np.array(FRAMES[ref]["cams"][cam]["T_world_cam"]) @ delta(*d)
    Kr = K.copy(); Kr[:2] *= s
    t0 = time.time()
    with LOCK:
        rgb, depth, alpha, _, _ = scene.render(torch.tensor(T, dtype=torch.float32), torch.tensor(Kr, dtype=torch.float32),
                                               W * s, H * s, tick, actors=None if q.get("actors", "1") == "1" else [])
        rgb, depth, alpha = rgb.clamp(0, 1).cpu().numpy(), depth.cpu().numpy(), alpha.cpu().numpy()
    ms = (time.time() - t0) * 1000
    info = dict(ms=round(ms), gt=gt[5] if gt else None, x=round(float(T[0, 3] + CK["center"][0]), 2),
                y=round(float(-(T[1, 3] + CK["center"][1])), 2), z=round(float(T[2, 3] + CK["center"][2]), 2))
    rec8 = (rgb * 255).astype(np.uint8)
    if gt:
        g = gt[0].astype(np.float32) / 255.0
        ok = ~gt[3]
        mse = float(((rgb - g) ** 2)[ok].mean())
        info["psnr"] = round(-10 * math.log10(max(mse, 1e-10)), 2)
    if mode == "depth":
        a, b = depth_img(depth, alpha > 0.5), (depth_img(gt[1], ~gt[2]) if gt else None)
    elif mode == "error" and gt:
        e = np.abs(rgb - gt[0].astype(np.float32) / 255.0).mean(-1)
        a = turbo(np.clip(e * 4, 0, 1)); a[gt[3]] = 0; b = gt[0]
    else:
        a, b = rec8, (gt[0] if gt else None)
    out = []
    for im in (a, b):
        if im is None:
            out.append(None); continue
        pil = Image.fromarray(im)
        if q.get("boxes") == "1":
            draw_boxes(pil, T, Kr, tick)
        out.append(jpeg(np.array(pil)))
    return dict(recon=out[0], gt=out[1], info=info)


META = dict(cams=CAMS, ticks=[TICKS[0], TICKS[-1]], dt=D["dt"], W=W, H=H, offsets=OFFSETS, novel_ticks=sorted(NOVEL),
            test_ticks=sorted(t for t, f in FRAMES.items() if f["cams"]["CAM_FRONT"]["split"] == "test"),
            ckpt=os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))),
            gaussians={n: len(g["means"]) for n, g in CK["groups"].items()},
            actors={n: BOXES[n][0] for n in scene.tracks})


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self.send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/meta":
                self.send(json.dumps(META).encode(), "application/json")
            elif u.path == "/render":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                self.send(json.dumps(render(q)).encode(), "application/json")
            else:
                self.send_error(404)
        except Exception as e:                                             # report to the page, keep serving
            self.send_response(500); self.end_headers(); self.wfile.write(repr(e).encode())

    def log_message(self, *a):
        pass


print("viewer: http://127.0.0.1:%d/  (%s, %d Gaussians)" % (args.port, META["ckpt"], sum(META["gaussians"].values())), flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
