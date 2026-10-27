"""Scenario plan for the downtown LA intersection (junction 10295 of LA_DTLA_ODR): every actor's path (0.5 m
polyline from the OpenDRIVE lanes, Hermite curves through the junction where the map has no connection) and its
speed profile, timed so the conflicts happen in a fixed order. Plain Python + carla (venv38), no server needed:
the map is built client-side from the xodr. Output: plan.json (paths, profiles, per-tick poses, clearance table).

Timeline (dt 0.1 s, 200 ticks):
  ego          lane -3 of road 933 (left-most straight lane), straight through the junction onto road 396
  lead         same lane, 25 m ahead, 9 m/s, straight through
  oncoming_rt  road 396 lane 1 towards the ego, turns right into the cross street (the map's connection 3917)
  oncoming_lt  road 396 lane 1, turns left across the ego's path onto road 787 (scripted curve: the map has no
               connection), crossing the ego's lane at T_LT
  oncoming_stop road 396 lane 1, stops before the far crosswalk for the pedestrians
  cyclist      rides along the right edge of the cross street (one-way, 4 lanes, red light) through the junction,
               across the ego's path at T_BIKE
  ped_far_*    two pedestrians cross the far crosswalk (across road 396) right -> left in front of the ego
  ped_left_*   two pedestrians on the left parallel crosswalk, ped_right_0 on the right one
  wait_*       two cars standing at the red light on the cross street (static)
usage: plan.py OUT_JSON [--xodr PATH]"""
import json, math, os, sys
import carla

XODR = sys.argv[sys.argv.index("--xodr") + 1] if "--xodr" in sys.argv else r"C:\mywork\osm2carla\export\LA_DTLA\LA_DTLA.xodr"
OUT = sys.argv[1]
DT, TICKS = 0.1, 200
STEP = 0.5                                  # polyline spacing, m
SPOT = carla.Location(397.65, 528.80, 0.0)  # the sensor audit's downtown_intersection spot (road 933, lane -1)
T_BIKE, T_LT = 11.0, 13.0                    # cyclist / left-turner in the ego's lane at these times
T_PED_CLEAR = 15.0                          # far-crosswalk pedestrians 2 m past the ego's lane centre by then
SIZES = {"car": (4.8, 2.0), "bike": (1.8, 0.7), "walker": (0.6, 0.6)}   # footprint for the clearance check

m = carla.Map("LA_DTLA_ODR", open(XODR, encoding="utf-8").read())


# ---------------------------------------------------------------- geometry helpers
def wp_xyz(w):
    l = w.transform.location
    return [l.x, l.y, l.z]


def follow(w, length, prefer=()):
    """Polyline along the lane from waypoint w for `length` m; at branches take a road in `prefer`, else the
    branch with the smallest heading change."""
    pts, d = [wp_xyz(w)], 0.0
    while d < length:
        nx = w.next(STEP)
        if not nx:
            break
        pick = [n for n in nx if n.road_id in prefer]
        if not pick:
            yaw = w.transform.rotation.yaw
            pick = sorted(nx, key=lambda n: abs((n.transform.rotation.yaw - yaw + 180) % 360 - 180))
        w = pick[0]; pts.append(wp_xyz(w)); d += STEP
    return pts, w


def back(w, length):
    """Polyline from `length` m upstream of w up to w."""
    p = w.previous(length)
    if not p:
        raise RuntimeError("no lane %.0f m upstream of road %d lane %d" % (length, w.road_id, w.lane_id))
    pts, _ = follow(p[0], length - 1e-6)
    return pts


def hermite(p0, yaw0, p1, yaw1, k=1.0):
    """Cubic Hermite curve p0 -> p1 with the given end headings, sampled every ~STEP m (z linear)."""
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * k
    m0 = (L * math.cos(math.radians(yaw0)), L * math.sin(math.radians(yaw0)))
    m1 = (L * math.cos(math.radians(yaw1)), L * math.sin(math.radians(yaw1)))
    n = max(4, int(L / STEP * 1.6))
    out = []
    for i in range(n + 1):
        u = i / n
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        out.append([h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
                    h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1], p0[2] + (p1[2] - p0[2]) * u])
    return resample(out)


def resample(pts):
    """Uniform STEP spacing along the polyline, drop duplicates."""
    out, carry = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a[:2], b[:2])
        if seg < 1e-9:
            continue
        t = STEP - carry
        while t <= seg:
            f = t / seg
            out.append([a[i] + (b[i] - a[i]) * f for i in range(3)]); t += STEP
        carry = seg - (t - STEP)
    return out


def offset(pts, lateral):
    """Polyline shifted `lateral` m to the right (CARLA frame: right of heading yaw is (-sin yaw, cos yaw))."""
    out = []
    for i, p in enumerate(pts):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        out.append([p[0] - math.sin(yaw) * lateral, p[1] + math.cos(yaw) * lateral, p[2]])
    return resample(out)


def join(*parts):
    pts = []
    for p in parts:
        if pts and math.dist(pts[-1][:2], p[0][:2]) < 0.05:
            p = p[1:]
        pts += p
    return resample(pts)


def cum(pts):
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + math.dist(a[:2], b[:2]))
    return s


def at(pts, S, s):
    """(x, y, z, yaw) at arc length s (clamped)."""
    s = min(max(s, 0.0), S[-1])
    i = max(0, min(len(S) - 2, int(s / STEP)))
    while i < len(S) - 2 and S[i + 1] < s:
        i += 1
    while i > 0 and S[i] > s:
        i -= 1
    a, b = pts[i], pts[i + 1]
    f = (s - S[i]) / max(S[i + 1] - S[i], 1e-9)
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f,
            math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))]


def cross(p, q):
    """Arc lengths (s_p, s_q) where polylines p and q intersect (first hit)."""
    Sp, Sq = cum(p), cum(q)
    for i in range(len(p) - 1):
        for j in range(len(q) - 1):
            (x1, y1), (x2, y2) = p[i][:2], p[i + 1][:2]
            (x3, y3), (x4, y4) = q[j][:2], q[j + 1][:2]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-12:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                return Sp[i] + t * (Sp[i + 1] - Sp[i]), Sq[j] + u * (Sq[j + 1] - Sq[j])
    return None


def dist_s(profile, t):
    """Distance covered by time t for a piecewise-linear speed profile [[t, v], ...] (v constant outside)."""
    d, tt = 0.0, 0.0
    while tt < t - 1e-12:
        h = min(0.01, t - tt)
        d += 0.5 * (speed(profile, tt) + speed(profile, tt + h)) * h; tt += h
    return d


def speed(profile, t):
    if t <= profile[0][0]:
        return profile[0][1]
    for (t0, v0), (t1, v1) in zip(profile, profile[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return profile[-1][1]


# ---------------------------------------------------------------- crosswalks (quads of 4 consecutive points)
cwp = m.get_crosswalks()
quads = [[[p.x, p.y, p.z] for p in cwp[i:i + 4]] for i in range(0, len(cwp) - len(cwp) % 4, 4)]


def crosswalk(near):
    """Centre line (two ends: mid points of the short edges) of the crosswalk quad nearest to `near`."""
    q = min(quads, key=lambda q: math.dist(near, [sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4]))
    e = [(q[i], q[(i + 1) % 4]) for i in range(4)]
    short = sorted(e, key=lambda ab: math.dist(ab[0][:2], ab[1][:2]))[:2]
    ends = [[(a[k] + b[k]) / 2 for k in range(3)] for a, b in short]
    width = math.dist(short[0][0][:2], short[0][1][:2])
    return ends, width, q


def walk_line(a, b, ext, lateral=0.0, ext_end=None):
    """Straight walking path a -> b, extended `ext` m before a and `ext_end` (default `ext`) m beyond b, shifted
    `lateral` m to the left."""
    ext_end = ext if ext_end is None else ext_end
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
    lx, ly = uy, -ux                                   # left of the walking direction (x fwd, y right frame)
    p0 = [a[0] - ux * ext + lx * lateral, a[1] - uy * ext + ly * lateral, a[2]]
    p1 = [b[0] + ux * ext_end + lx * lateral, b[1] + uy * ext_end + ly * lateral, b[2]]
    return resample([p0, p1])


# ---------------------------------------------------------------- paths
spot = m.get_waypoint(SPOT, project_to_road=True, lane_type=carla.LaneType.Driving)
assert spot.road_id == 933 and spot.lane_id == -1, (spot.road_id, spot.lane_id)
ego_lane = spot.get_right_lane().get_right_lane()                  # lane -3: left-most straight-through lane
assert ego_lane.lane_id == -3
ego_path = join(back(ego_lane, 60.0), follow(ego_lane, 260.0, prefer=(3921,))[0])

w396 = m.get_waypoint(carla.Location(436.4, 487.2, 84.2))          # start of connection 3917 (road 396 lane 1)
assert w396.road_id == 396 and w396.lane_id == 1, (w396.road_id, w396.lane_id)
onc_up = back(w396, 110.0)
rt_path = join(onc_up, follow(w396, 90.0, prefer=(3917,))[0])
w787 = m.get_waypoint(carla.Location(446.7, 535.0, 83.2))          # road 787 lane -1 (exit to the ego's right), s=5
assert w787.road_id == 787 and w787.lane_id == -1, (w787.road_id, w787.lane_id)
lt_turn = hermite(wp_xyz(w396), w396.transform.rotation.yaw, wp_xyz(w787), w787.transform.rotation.yaw, k=1.1)
lt_path = join(onc_up, lt_turn, follow(w787, 60.0)[0])

near_cw, near_w, near_q = crosswalk([409.4, 510.1])               # across road 933 just before the junction
far_cw, far_w, far_q = crosswalk([438.7, 487.0])                  # across road 396 beyond the junction
left_cw, left_w, left_q = crosswalk([408.8, 496.7])               # across the cross street, ego's left
right_cw, right_w, right_q = crosswalk([451.0, 530.2])            # across the right-hand street

# ends ordered right -> left of the ego (ego right vector = (-sin yaw, cos yaw) in CARLA's frame)
eyaw = math.radians(ego_lane.transform.rotation.yaw)
rvec = (-math.sin(eyaw), math.cos(eyaw))
def right_to_left(ends):
    return sorted(ends, key=lambda p: -(p[0] * rvec[0] + p[1] * rvec[1]))
near_r, near_l = right_to_left(near_cw)
far_r, far_l = right_to_left(far_cw)
fwd = (math.cos(eyaw), math.sin(eyaw))
left_n, left_f = sorted(left_cw, key=lambda p: p[0] * fwd[0] + p[1] * fwd[1])      # near/far along the ego heading
right_n, right_f = sorted(right_cw, key=lambda p: p[0] * fwd[0] + p[1] * fwd[1])

w_bike = m.get_waypoint(carla.Location(461.2, 522.7, 83.2))        # cross street lane -4, 5 m before the junction
assert w_bike.lane_id == -4 and not w_bike.is_junction, (w_bike.road_id, w_bike.lane_id)
bike_path = offset(join(back(w_bike, 40.0), follow(w_bike, 95.0, prefer=(3920,))[0]), 1.1)   # near the kerb
peds = {
    "ped_far_0": walk_line(far_r, far_l, 2.5, lateral=0.5),
    "ped_far_1": walk_line(far_r, far_l, 2.5, lateral=-0.6),
    "ped_left_0": walk_line(left_f, left_n, 2.5, lateral=0.5),
    "ped_left_1": walk_line(left_f, left_n, 2.5, lateral=-0.5),
    "ped_right_0": walk_line(right_n, right_f, 2.5, lateral=0.0),
}

# cars waiting at the red light on the cross street (road 3920 lanes -1, -2 enter the junction at these points)
waits = []
for x, y in ((450.8, 527.9), (453.0, 525.1)):
    w = m.get_waypoint(carla.Location(x, y, 83.2))
    p = w.previous(9.0)[0]                                           # behind the right-hand crosswalk
    waits.append(p)

# ---------------------------------------------------------------- timing
ego_prof = [[0, 8.0], [6, 8.0], [9, 5.0], [12, 5.0], [16, 9.0], [20, 9.0]]
lead_prof = [[0, 9.0], [20, 9.0]]
s_bike_ego, s_ego_bike = cross(bike_path, ego_path)
s_lt, s_ego_lt = cross(lt_path, ego_path)
bike_prof = [[0, 5.0], [20, 5.0]]
lt_prof = [[0, 7.0], [9, 7.0], [11, 5.5], [20, 7.5]]
rt_prof = [[0, 7.5], [5, 7.5], [8, 4.5], [20, 7.0]]
stop_prof = [[0, 8.0], [9, 8.0], [14.5, 0.0], [20, 0.0]]
s_far_ego = cross(walk_line(far_r, far_l, 2.5), ego_path)[1]
s_far_stop = cross(walk_line(far_r, far_l, 2.5), onc_up + follow(w396, 5.0)[0])[1]


def walker_prof(t0, v):
    return [[0, 0.0], [t0, 0.0], [t0 + 0.6, v], [20, v]]


def ped_start(path, v):
    """Start time so the walker is 2 m past the ego's lane centre at T_PED_CLEAR."""
    s_c = cross(path, ego_path)[0]
    return round(T_PED_CLEAR - 0.3 - (s_c + 2.0) / v, 2)


def actor(name, kind, cls, path, prof, s0, role="", static=False):
    return dict(name=name, kind=kind, cls=cls, role=role or name, path=path, profile=prof, s0=s0, static=static)


plan = [
    actor("ego", "car", "car", ego_path, ego_prof, 0.0, role="hero"),
    actor("lead", "car", "car", ego_path, lead_prof, 25.0),
    actor("cyclist", "bike", "bicycle", bike_path, bike_prof, s_bike_ego - dist_s(bike_prof, T_BIKE)),
    actor("oncoming_lt", "car", "car", lt_path, lt_prof, s_lt - dist_s(lt_prof, T_LT)),
    actor("oncoming_rt", "car", "car", rt_path, rt_prof, 110.0 - 50.0),
    actor("oncoming_stop", "car", "car", onc_up + follow(w396, 5.0)[0], stop_prof,
          (s_far_stop - 6.0) - dist_s(stop_prof, 20.0)),
    actor("ped_far_0", "walker", "pedestrian", peds["ped_far_0"], walker_prof(ped_start(peds["ped_far_0"], 1.4), 1.4), 0.0),
    actor("ped_far_1", "walker", "pedestrian", peds["ped_far_1"], walker_prof(ped_start(peds["ped_far_1"], 1.3) + 0.4, 1.3), 0.0),
    actor("ped_left_0", "walker", "pedestrian", peds["ped_left_0"], walker_prof(0.0, 1.3), 0.0),
    actor("ped_left_1", "walker", "pedestrian", peds["ped_left_1"], walker_prof(1.2, 1.2), 0.0),
    actor("ped_right_0", "walker", "pedestrian", peds["ped_right_0"], walker_prof(0.5, 1.25), 0.0),
]
for i, w in enumerate(waits):
    pts, _ = follow(w, 6.0)
    plan.append(actor("wait_%d" % i, "car", "car", pts, [[0, 0.0], [20, 0.0]], 0.0, static=True))

# per-tick poses (tick k is the state after k * DT s)
violations = []
for a in plan:
    S = cum(a["path"])
    a["length"] = S[-1]
    a["poses"] = []
    for k in range(TICKS + 1):
        s = a["s0"] + dist_s(a["profile"], k * DT)
        if a["kind"] == "walker":
            s = min(s, S[-1])                                    # walkers stop at the end of their path
        if (s < 0 or s > S[-1]) and not any(v.startswith(a["name"] + " ") for v in violations):
            violations.append("%s leaves its path at tick %d (s=%.1f of %.1f)" % (a["name"], k, s, S[-1]))
        a["poses"].append([round(v, 4) for v in at(a["path"], S, s)] + [round(s, 3)])


# ---------------------------------------------------------------- clearance check (3 circles per footprint)
def circles(a, k):
    x, y, z, yaw, s = a["poses"][k]
    L, W = SIZES[a["kind"]]
    c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    off = [0.0] if a["kind"] == "walker" else [-(L - W) / 2, 0.0, (L - W) / 2]
    return [(x + c * o, y + sn * o, W / 2) for o in off]


table = []
for i in range(len(plan)):
    for j in range(i + 1, len(plan)):
        best = (1e9, 0)
        for k in range(TICKS + 1):
            d = min(math.hypot(p[0] - q[0], p[1] - q[1]) - p[2] - q[2] for p in circles(plan[i], k) for q in circles(plan[j], k))
            best = min(best, (d, k))
        table.append(dict(a=plan[i]["name"], b=plan[j]["name"], gap_m=round(best[0], 2), t=round(best[1] * DT, 1)))
kind = {a["name"]: a["kind"] for a in plan}
tight = [r for r in table if r["gap_m"] < (0.3 if kind[r["a"]] == kind[r["b"]] == "walker" else 1.0)]

ego_s = lambda t: dist_s(ego_prof, t)
summary = dict(
    ego_distance_m=round(ego_s(20.0), 1),
    bike_in_ego_lane_t=T_BIKE, ego_gap_to_bike_crossing_m=round(s_ego_bike - ego_s(T_BIKE), 1),
    left_turner_in_ego_lane_t=T_LT, ego_gap_to_lt_crossing_m=round(s_ego_lt - ego_s(T_LT), 1),
    ego_reaches_far_crosswalk_t=round(next(k * DT for k in range(TICKS + 1) if ego_s(k * DT) >= s_far_ego), 1),
    tight_pairs=tight)
meta = dict(map="LA_DTLA_ODR", xodr=os.path.abspath(XODR), dt=DT, ticks=TICKS, step=STEP,
            spot=[SPOT.x, SPOT.y], junction=10295,
            crosswalks=dict(near=near_q, far=far_q, left=left_q, right=right_q))
# lane centre lines around the junction for plots
lanes = []
for w in m.generate_waypoints(2.0):
    l = w.transform.location
    if math.hypot(l.x - 430, l.y - 510) < 140:
        lanes.append([round(l.x, 2), round(l.y, 2), w.road_id, w.lane_id, int(w.is_junction)])
json.dump(dict(meta=meta, summary=summary, clearance=table, actors=plan, lanes=lanes), open(OUT, "w"))
print(json.dumps(summary, indent=1))
for v in violations:
    print("VIOLATION", v)
os._exit(1 if violations and "--loose" not in sys.argv else 0)
