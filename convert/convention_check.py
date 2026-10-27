"""Score the two candidate rotation conventions against the server renders of convention_probe.py.
  H_ue     : carla_geom.carla_matrix (Unreal FRotator: positive pitch = nose up)
  H_client : the carla wheel's Transform.get_matrix() of this build = D H_ue D, D = diag(1, 1, -1)
Cameras: predicted sky (world ray z > 0, camera 600 m up) vs rendered sky, IoU per case.
Lidar: lidar points vs depth-camera points at the same pose, compared in the sensor frame (axes test, y as is or
negated), and the ground height of the lidar points in the world under each rotation hypothesis.
usage: convention_check.py PROBE_DIR"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carla_geom import carla_matrix, decode_depth

D = np.diag([1.0, 1.0, -1.0])
P = sys.argv[1]
c = json.load(open(os.path.join(P, "convention.json")))
H = {"H_ue": lambda tf: carla_matrix(*tf), "H_client": lambda tf: np.block([[D @ carla_matrix(*tf)[:3, :3] @ D, carla_matrix(*tf)[:3, 3:]], [np.zeros((1, 3)), np.ones((1, 1))]])}
out = {"cameras": []}
for cam in c["cams"]:
    w, h = cam["w"], cam["h"]
    f = w / (2 * np.tan(np.radians(cam["fov"]) / 2))
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = np.stack([np.ones_like(u), (u - w / 2) / f, -(v - h / 2) / f], -1).reshape(-1, 3)      # UE camera frame
    sky = np.frombuffer(open(os.path.join(P, cam["name"] + ".sky"), "rb").read(), np.uint8).astype(bool)
    row = dict(name=cam["name"], set=cam["set"], reported=[round(x, 3) for x in cam["reported"][3:]], sky_frac=round(float(sky.mean()), 3))
    for k, fn in H.items():
        pred = (d @ fn(cam["reported"])[:3, :3].T)[:, 2] > 0
        row[k] = round(float((pred & sky).sum() / max((pred | sky).sum(), 1)), 3)
    out["cameras"].append(row)

pts = np.frombuffer(open(os.path.join(P, "lidar.bin"), "rb").read(), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"), ("idx", "<u4"), ("tag", "<u4")]))
L = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64)
dep = decode_depth(np.array(Image.open(os.path.join(P, "depth.png")).convert("RGB")))
w, h = c["depth_wh"]; f = w / 2.0
u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
ok = dep < 55
Q = np.stack([dep, (u - w / 2) / f * dep, -(v - h / 2) / f * dep], -1)[ok]                     # UE camera frame
tree = cKDTree(Q)
for name, M in (("lidar_as_is", np.eye(3)), ("lidar_y_negated", np.diag([1.0, -1.0, 1.0])), ("lidar_x_negated", np.diag([-1.0, 1.0, 1.0]))):
    L2 = L @ M.T
    # only lidar points inside the depth camera's frustum
    inf = (L2[:, 0] > 0.5) & (np.abs(L2[:, 1] / L2[:, 0]) < 1.0) & (np.abs(L2[:, 2] / L2[:, 0]) < h / w)
    dd, _ = tree.query(L2[inf])
    out[name] = dict(n=int(inf.sum()), median_nn_m=round(float(np.median(dd)), 4), p90_nn_m=round(float(np.percentile(dd, 90)), 4))
road = (pts["tag"] == 1) | (pts["tag"] == 24)
for k, fn in H.items():
    T = fn(c["lidar"])
    best = min(("lidar_as_is", np.eye(3)), ("lidar_y_negated", np.diag([1.0, -1.0, 1.0])), key=lambda m: out[m[0]]["median_nn_m"])
    Wp = (L[road] @ best[1].T) @ T[:3, :3].T + T[:3, 3]
    out["ground_" + k] = dict(z_median=round(float(np.median(Wp[:, 2])), 3), z_std=round(float(np.std(Wp[:, 2])), 3), n=int(road.sum()))
print(json.dumps(out, indent=1))
