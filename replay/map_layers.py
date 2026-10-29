"""Map layers for the 3D replay viewer from the OpenDRIVE (offline carla.Map, no server): lane lines with their
marking type and colour, road boundaries (lane edges with no driving lane beside them), lane centres (junction
connections included) and crosswalk outlines, within RADIUS m of a point. CARLA coordinates. venv38 (carla).
usage: map_layers.py XODR OUT_JSON --center X,Y [--radius 250]"""
import json, math, sys
import carla

XODR, OUT = sys.argv[1], sys.argv[2]
CX, CY = (float(v) for v in sys.argv[sys.argv.index("--center") + 1].split(","))
RADIUS = float(sys.argv[sys.argv.index("--radius") + 1]) if "--radius" in sys.argv else 250.0
STEP = 1.0
m = carla.Map("map", open(XODR, encoding="utf-8").read())


def near(l):
    return math.hypot(l.x - CX, l.y - CY) <= RADIUS


def edge(w, side):
    """point on the lane's left (-1) or right (+1) edge; CARLA yaw, y to the right."""
    l, a = w.transform.location, math.radians(w.transform.rotation.yaw)
    h = side * w.lane_width / 2
    return [round(l.x - math.sin(a) * h, 2), round(l.y + math.cos(a) * h, 2), round(l.z, 2)]


def driving(w):
    return w is not None and w.lane_type == carla.LaneType.Driving


lanes = {}
for w in m.generate_waypoints(STEP):
    if near(w.transform.location):
        lanes.setdefault((w.road_id, w.section_id, w.lane_id), []).append(w)

layers = {"lane_line": [], "road_boundary": [], "lane": [], "junction_lane": [], "crosswalk": []}
for key, ws in lanes.items():
    ws.sort(key=lambda w: w.s)
    centre = [[round(w.transform.location.x, 2), round(w.transform.location.y, 2), round(w.transform.location.z, 2)] for w in ws]
    if len(centre) > 1:
        layers["junction_lane" if ws[0].is_junction else "lane"].append(centre)
    if ws[0].is_junction:                                              # overlapping connections: centres only
        continue
    for side, nb, mk in ((-1, lambda w: w.get_left_lane(), lambda w: w.left_lane_marking),
                         (1, lambda w: w.get_right_lane(), lambda w: w.right_lane_marking)):
        run, kind = [], None
        for w in ws + [None]:                                          # split where the edge type changes
            if w is not None:
                mark = mk(w)
                k = ("road_boundary", "solid", "white") if not driving(nb(w)) else \
                    ("lane_line", str(mark.type).lower(), str(mark.color).lower())
                if k[0] == "lane_line" and k[1] == "none":
                    k = None
            if w is None or k != kind:
                if kind is not None and len(run) > 1:
                    layers[kind[0]].append(dict(p=run, type=kind[1], color=kind[2]) if kind[0] == "lane_line" else run)
                run, kind = [], (k if w is not None else None)
            if w is not None and k is not None:
                run.append(edge(w, side))

cw = [[round(l.x, 2), round(l.y, 2), round(l.z, 2)] for l in m.get_crosswalks()]
for i in range(0, len(cw) - 3, 4):                                     # this map: 4 corners per crosswalk
    poly = cw[i:i + 4] + [cw[i]]
    if any(near(carla.Location(x, y, z)) for x, y, z in poly):
        layers["crosswalk"].append(poly)

json.dump(dict(center=[CX, CY], radius=RADIUS, frame="carla", layers=layers), open(OUT, "w"), separators=(",", ":"))
print({k: len(v) for k, v in layers.items()}, flush=True)
