"""Offline forward / inverse kinematics of the CARLA walker skeleton (numpy, .venv or the project venv).
  rider_ik.py SKELETON_JSON check              FK of the dumped rest pose vs the dumped component positions
  rider_ik.py SKELETON_JSON solve OUT_POSE     cycling pose (bone deltas for rider_pose.py) for the CrossBike
Bone deltas are rotations (pitch, yaw, roll) applied in each bone's own frame, Unreal convention."""
import json, sys
import numpy as np
from scipy.optimize import minimize

SK = json.load(open(sys.argv[1]))
REL = {b["name"]: np.array(b["rel"], float) for b in SK}
COMP = {b["name"]: np.array(b["comp"], float) for b in SK}
NAMES = [b["name"] for b in SK]


def R(p, y, r):
    cy, sy = np.cos(np.radians(y)), np.sin(np.radians(y))
    cr, sr = np.cos(np.radians(r)), np.sin(np.radians(r))
    cp, sp = np.cos(np.radians(p)), np.sin(np.radians(p))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                     [sp, -cp * sr, cp * cr]])


def euler(M):
    return [float(np.degrees(np.arcsin(np.clip(M[2, 0], -1, 1)))), float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            float(np.degrees(np.arctan2(-M[2, 1], M[2, 2])))]


def parent(n):
    side = n[-3:] if n.endswith(("__L", "__R")) else ""
    chains = [["crl_root", "crl_hips__C", "crl_spine__C", "crl_spine01__C", "crl_neck__C", "crl_Head__C"],
              ["crl_spine01__C", "crl_shoulder" + side, "crl_arm" + side, "crl_foreArm" + side, "crl_hand" + side],
              ["crl_hips__C", "crl_thigh" + side, "crl_leg" + side, "crl_foot" + side, "crl_toe" + side, "crl_toeEnd" + side],
              ["crl_Head__C", "crl_eye" + side]]
    for ch in chains:
        if n in ch[1:]:
            return ch[ch.index(n) - 1]
    if n.startswith("crl_hand") and n != "crl_hand" + side:          # fingers: chain by digit
        base = n[len("crl_hand"):-3]
        digit = base.rstrip("0123456789").replace("End", "")
        order = ["", "01", "02", "End"]
        k = order.index(base[len(digit):]) if base[len(digit):] in order else 0
        return "crl_hand" + side if k == 0 else "crl_hand" + digit + order[k - 1] + side
    return None


BODY = [n for n in NAMES if not any(k in n for k in ("Thumb", "Index", "Middle", "Ring", "Pinky"))]


def fk(deltas=None, offset=(np.zeros(3), np.eye(3)), names=None):
    deltas = deltas or {}
    pos, rot = {}, {}
    for n in names or NAMES:
        loc, rr = REL[n][:3], R(*REL[n][3:])
        if n in deltas:
            rr = rr @ R(*deltas[n])
        p = parent(n)
        if p is None:
            pos[n], rot[n] = offset[0] + offset[1] @ loc, offset[1] @ rr
        else:
            pos[n], rot[n] = pos[p] + rot[p] @ loc, rot[p] @ rr
    return pos, rot


def fit_offset():
    best = None
    for yaw in (0, 90, 180, -90):
        for roll in (0, 90, -90, 180):
            Ro = R(0, yaw, roll)
            pos, _ = fk(offset=(np.zeros(3), Ro))
            t = np.mean([COMP[n] - pos[n] for n in NAMES], axis=0)
            err = max(np.linalg.norm(COMP[n] - pos[n] - t) for n in NAMES)
            if best is None or err < best[0]:
                best = (err, yaw, roll, t, Ro)
    return best


if sys.argv[2] == "check":
    err, yaw, roll, t, Ro = fit_offset()
    print("mesh offset yaw %d roll %d translation %s -> max FK error %.4f m" % (yaw, roll, np.round(t, 3), err))
    sys.exit(0 if err < 0.01 else 1)

# ---------------------------------------------------------------- solve the cycling pose
err, yaw, roll, t, Ro = fit_offset()
assert err < 0.01, err
OFF = (t, Ro)
# CrossBike geometry in the bike frame (x forward, y right, z up, origin = mesh pivot), from rider_pose.py pose_info
B = json.load(open(sys.argv[4]))["bike_bbox"] if len(sys.argv) > 4 else [0.0, 0.0, 0.5, 0.85, 0.3, 0.5]
cx, cz, ex, ey, ez = B[0], B[2], B[3], B[4], B[5]
ground, top = cz - ez, cz + ez
r_wheel = 0.27 * (top - ground)                                           # wheel radius ~ 0.27 x bike height (render)
rear_x, front_x = cx - ex + r_wheel, cx + ex - r_wheel
bb_pt = np.array([cx + 0.02, 0.0, ground + r_wheel - 0.03])               # crank axle near the middle
saddle = np.array([bb_pt[0] - 0.20, 0.0, ground + 0.80 * (top - ground)])
grip = np.array([front_x - 0.08, ey - 0.07, top - 0.04])                  # bar ends (bbox width = bar width)
crank = 0.17
TARGET = {"crl_foot__L": bb_pt + [crank, -0.12, -0.05], "crl_foot__R": bb_pt + [-crank * 0.8, 0.12, crank * 0.6],
          "crl_hand__L": grip * [1, -1, 1], "crl_hand__R": grip}
pos0, _ = fk(offset=OFF)
seat = saddle - pos0["crl_hips__C"] + np.array([0.0, 0.0, 0.10])        # walker origin in the bike frame
JOINTS = ["crl_spine__C", "crl_spine01__C", "crl_neck__C", "crl_thigh__L", "crl_leg__L", "crl_thigh__R", "crl_leg__R",
          "crl_arm__L", "crl_foreArm__L", "crl_arm__R", "crl_foreArm__R", "crl_foot__L", "crl_foot__R"]


def unpack(x):
    return {n: list(x[3 * i:3 * i + 3]) for i, n in enumerate(JOINTS)}


def cost(x):
    pos, rot = fk(unpack(x), OFF, BODY)
    c = 0.0
    for n, tgt in TARGET.items():
        c += 10.0 * np.sum((pos[n] + seat - tgt) ** 2)
    neck = pos["crl_neck__C"] - pos["crl_hips__C"]                          # torso leaning forward ~35 deg
    c += 2.0 * (np.degrees(np.arctan2(neck[0], neck[2])) - 35.0) ** 2 / 100.0
    eye = 0.5 * (pos["crl_eye__L"] + pos["crl_eye__R"]) - pos["crl_Head__C"]  # look ahead, not down
    c += 0.5 * (eye[2] / max(np.linalg.norm(eye), 1e-6)) ** 2
    for s in ("L", "R"):                                                     # knees forward, elbows down / out
        knee = pos["crl_leg__" + s] - pos["crl_thigh__" + s]
        c += 5.0 * max(0.0, -knee[0]) ** 2 + 2.0 * (pos["crl_leg__" + s][1] - pos["crl_thigh__" + s][1]) ** 2
        toe = pos["crl_toeEnd__" + s] - pos["crl_foot__" + s]
        c += 1.0 * (toe[2] / max(np.linalg.norm(toe), 1e-6)) ** 2 + 0.5 * max(0.0, -toe[0]) ** 2
    c += 1e-5 * np.sum(np.asarray(x) ** 2)
    return c


x0 = np.zeros(3 * len(JOINTS))
res = minimize(cost, x0, method="L-BFGS-B", options=dict(maxiter=5000))
res = minimize(cost, res.x, method="Powell", options=dict(maxiter=20000, xtol=1e-3, ftol=1e-10))
pos, _ = fk(unpack(res.x), OFF)
report = {n: round(float(np.linalg.norm(pos[n] + seat - tgt)), 3) for n, tgt in TARGET.items()}
print("cost %.5f  residual (m) %s  seat %s" % (res.fun, report, np.round(seat, 3)))
json.dump(dict(walker="0015", seat=[float(v) for v in seat], bones={n: [round(v, 2) for v in d] for n, d in unpack(res.x).items()},
               targets={n: [float(v) for v in t] for n, t in TARGET.items()}, residual_m=report,
               bike_geometry=dict(saddle=saddle.tolist(), crank=bb_pt.tolist(), grip=grip.tolist())),
          open(sys.argv[3], "w"), indent=1)
