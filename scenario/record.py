"""Run plan.json in CARLA (synchronous, fixed delta) and record it from the ego (venv38, server running).

Actors: vehicles and the bicycle are kinematic (physics off, placed every tick on the plan pose, height and pitch
from the road mesh under the wheels); walkers walk on CARLA's character physics, steered every tick towards their
planned position. Traffic lights are frozen (ego / oncoming green, cross street red), weather is fixed.
Sensors on the ego (rig.json): 6 cameras (RGB, depth, instance segmentation; pinhole, global shutter), a spinning
lidar and a semantic lidar at the same pose. CARLA's recorder runs for the whole scenario (replay_views.py).

Output (OUT/):
  meta.json                 rig, intrinsics, settings, weather, blueprints, recorder file, timing
  frames.jsonl              per frame: tick, CARLA frame, sim time, ego / sensor poses, every actor's box + pose
  cam/<CAM>/{rgb,depth,inst}/NNNN.png   raw CARLA encodings (depth: 24-bit in RGB, inst: tag in R, id in G/B)
  lidar/NNNN.bin            float32 x, y, z, intensity (sensor frame, CARLA axes)
  semlidar/NNNN.bin         CARLA SemanticLidarDetection records (x, y, z, cos_inc: f32; object_idx, tag: u32)
  recording.log             CARLA recorder file
Re-simulation for novel views (--novel LIVE_REC): the same scenario again without recorder, rig or lidar; every
kinematic actor (cars, bike, rider) is placed at its transform recorded in LIVE_REC, walkers walk with the same
controller (their deviation from the live run is measured), the ego is parked out of sight, and free cameras with
the front camera's intrinsics follow the recorded ego pose at lateral offsets (ego frame, + = right); RGB / depth /
instance images every EVERY-th tick -> OUT/novel/off_<dy>/..., poses -> OUT/replay.json. (CARLA's recorder replays
the kinematic actors 1-2 ticks late in this build, so its replay is not used for ground truth.)
usage: record.py PLAN_JSON OUT [--probe] [--ticks N] [--res 1280x720] [--raw]
       record.py PLAN_JSON OUT --novel LIVE_REC [--offsets -3,-2,-1,0,1,2,3] [--every 4]"""
import json, math, os, queue, subprocess, sys, time, traceback
import carla


def _die(*exc):                                    # an uncaught error must not leave the client hanging on exit
    traceback.print_exception(*exc); sys.stderr.flush(); sys.stdout.flush(); os._exit(1)


sys.excepthook = _die

PLAN, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
PROBE = "--probe" in sys.argv
NOVEL = os.path.abspath(sys.argv[sys.argv.index("--novel") + 1]) if "--novel" in sys.argv else None
OFFS = [float(v) for v in (sys.argv[sys.argv.index("--offsets") + 1] if "--offsets" in sys.argv else "-3,-2,-1,0,1,2,3").split(",")]
EVERY = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 4
LIVE = [json.loads(l) for l in open(os.path.join(NOVEL, "frames.jsonl"))] if NOVEL else None
RAW = "--raw" in sys.argv          # dump BGRA buffers (fast); convert/raw_to_png.py writes the PNGs afterwards


def save_img(d, path_png):
    if RAW:
        open(path_png[:-4] + ".bgra", "wb").write(bytes(d.raw_data))
    else:
        d.save_to_disk(path_png)
RES = sys.argv[sys.argv.index("--res") + 1] if "--res" in sys.argv else "1280x720"
W, H = (int(v) for v in RES.split("x"))
plan = json.load(open(PLAN))
DT, TICKS = plan["meta"]["dt"], plan["meta"]["ticks"]
TICKS = int(sys.argv[sys.argv.index("--ticks") + 1]) if "--ticks" in sys.argv else TICKS
WARMUP = 30
HERE = os.path.dirname(os.path.abspath(__file__))
RIG = json.load(open(os.path.join(HERE, "rig.json")))
os.makedirs(OUT, exist_ok=True)


def gpu_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return [int(v) for v in r.stdout.strip().split(",")]
    except Exception:
        return None


def tf_list(t):
    return [t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw, t.rotation.roll]


client = carla.Client("localhost", 2000); client.set_timeout(600.0)
world = client.get_world()
if not world.get_map().name.endswith(plan["meta"]["map"]):
    world = client.load_world(plan["meta"]["map"])
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = DT; world.apply_settings(s)
cmap, bl = world.get_map(), world.get_blueprint_library()
for a in world.get_actors():
    if a.type_id.startswith(("vehicle.", "walker.", "sensor.", "controller.")):
        a.destroy()
world.tick()

weather = world.get_weather()
for k, v in dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=0.0,
                 sun_azimuth_angle=200.0, sun_altitude_angle=55.0, fog_density=0.0, wetness=0.0).items():
    setattr(weather, k, v)
world.set_weather(weather)

# traffic lights of the junction: ego road (933) and the oncoming road (396) green, the rest red; all frozen
tl_state = {}
for tl in world.get_actors().filter("traffic.traffic_light"):
    roads = {w.road_id for w in tl.get_affected_lane_waypoints()}
    if tl.get_location().distance(carla.Location(430, 508, 84)) < 60:
        st = carla.TrafficLightState.Green if roads & {933, 396} else carla.TrafficLightState.Red
        tl.set_state(st); tl_state[tl.id] = dict(state=str(st), roads=sorted(roads), loc=tf_list(tl.get_transform())[:3])
world.freeze_all_traffic_lights(True)
world.tick()

# blueprints (deterministic choices; fall back to the first 4-wheel car by id)
def battr(b, k, default=""):
    if not b.has_attribute(k):
        return default
    a = b.get_attribute(k)
    try:
        return a.as_str()
    except RuntimeError:                                     # int attributes refuse as_str
        return str(int(a))


cars = sorted([b for b in bl.filter("vehicle.*") if battr(b, "number_of_wheels", "4") == "4"
               and battr(b, "base_type").lower() in ("car", "")], key=lambda b: b.id)
bikes = sorted([b for b in bl.filter("vehicle.*") if battr(b, "number_of_wheels") == "2"
                and "bike" in (battr(b, "base_type").lower() + b.id)], key=lambda b: b.id)
walkers = sorted(bl.filter("walker.pedestrian.*"), key=lambda b: b.id)
PREF = {"ego": "vehicle.lincoln.mkz", "lead": "vehicle.ue4.audi.tt", "oncoming_lt": "vehicle.nissan.patrol",
        "oncoming_rt": "vehicle.ue4.chevrolet.impala", "oncoming_stop": "vehicle.taxi.ford",
        "wait_0": "vehicle.ue4.mercedes.ccc", "wait_1": "vehicle.ue4.bmw.grantourer"}
WALKER_IDS = {"ped_far_0": "0016", "ped_far_1": "0021", "ped_left_0": "0026", "ped_left_1": "0030", "ped_right_0": "0035"}
# no 2-wheeled vehicle is registered in this build: the cyclist is CARLA's CrossBike mesh + a walker posed as rider
BIKE_MESH = "/Game/Carla/Static/Bicycle/CrossBike/SM_sc_crossbike.SM_sc_crossbike"
RIDER = json.load(open(os.path.join(HERE, "rider_pose.json")))      # rider_ik.py output: walker, seat, bone deltas


def ue_rot(p, y, r):
    """Unreal rotation matrix (the server's convention, convert/conventions.json; NOT the client's get_matrix)."""
    cy, sy = math.cos(math.radians(y)), math.sin(math.radians(y))
    cr, sr = math.cos(math.radians(r)), math.sin(math.radians(r))
    cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr]]


def mat_euler(M):
    return (math.degrees(math.asin(max(-1.0, min(1.0, M[2][0])))), math.degrees(math.atan2(M[1][0], M[0][0])),
            math.degrees(math.atan2(-M[2][1], M[2][2])))


def rider_tf(bike_tf):
    """Rider walker origin = bike transform applied to the seat offset (bike frame)."""
    r = bike_tf.rotation
    M = ue_rot(r.pitch, r.yaw, r.roll)
    s = RIDER["seat"]
    l = bike_tf.location
    return carla.Transform(carla.Location(l.x + sum(M[0][j] * s[j] for j in range(3)), l.y + sum(M[1][j] * s[j] for j in range(3)),
                                          l.z + sum(M[2][j] * s[j] for j in range(3))), r)


def pose_rider(walker):
    rel = {b.name: b.relative for b in walker.get_bones().bone_transforms}
    out = []
    for n, d in RIDER["bones"].items():
        t = rel[n]
        p, y, r = mat_euler([[sum(a * b for a, b in zip(row, col)) for col in zip(*ue_rot(*d))]
                             for row in ue_rot(t.rotation.pitch, t.rotation.yaw, t.rotation.roll)])
        out.append((n, carla.Transform(t.location, carla.Rotation(pitch=p, yaw=y, roll=r))))
    walker.set_bones(carla.WalkerBoneControlIn(out))
    walker.blend_pose(1.0)


def blueprint(a, i):
    if a["kind"] in ("walker", "rider"):
        want = "walker.pedestrian." + (RIDER["walker"] if a["kind"] == "rider" else WALKER_IDS.get(a["name"], "0015"))
        b = next((w for w in walkers if w.id == want), walkers[(3 * i) % len(walkers)])
        if b.has_attribute("is_invincible"):
            b.set_attribute("is_invincible", "false")
    elif a["kind"] == "bike":                              # mass > 0 makes the prop movable (tested: without it
        b = bl.find("static.prop.mesh"); b.set_attribute("mesh_path", BIKE_MESH)   # set_transform does nothing)
        b.set_attribute("mass", "15")
    else:
        pool = cars or sorted(bl.filter("vehicle.*"), key=lambda b: b.id)
        b = next((x for x in bl.filter("vehicle.*") if x.id == PREF.get(a["name"])), pool[i % len(pool)])
        if b.has_attribute("color"):
            rec = b.get_attribute("color").recommended_values
            if rec:
                b.set_attribute("color", rec[i % len(rec)])
    if b.has_attribute("role_name"):
        b.set_attribute("role_name", a["role"])
    return b


GROUND = {carla.CityObjectLabel.Roads, carla.CityObjectLabel.RoadLines, carla.CityObjectLabel.Ground,
          carla.CityObjectLabel.Sidewalks, carla.CityObjectLabel.Terrain}
ground_miss = [0]


def ground_z(x, y, z_hint):
    """road-mesh height under (x, y); started lower when a pole / tree / sign is hit first"""
    for up in (3.0, 1.5, 0.8):
        p = world.ground_projection(carla.Location(x, y, z_hint + up), up + 5.0)
        if p is not None and p.label in GROUND:
            return p.location.z
    ground_miss[0] += 1
    return z_hint


GROUND_CACHE = {}
# the CrossBike mesh's own box (measured at yaw 0, rider_pose.py); CARLA reports an inconsistent box for a rotated prop
BIKE_BB = carla.BoundingBox(carla.Location(0.0, 0.0, 0.539), carla.Vector3D(0.737, 0.418, 0.530))


def kin_pose(a, k, bb):
    """Plan pose k of a kinematic actor on the road mesh: z from the ground under its front and rear axles."""
    x, y, z, yaw, _ = a["poses"][min(k, len(a["poses"]) - 1)]
    c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    half = max(bb.extent.x * 0.6, 0.4)
    key = (a["name"], min(k, len(a["poses"]) - 1))
    if key not in GROUND_CACHE:                     # normally filled before any actor exists (no self hits)
        GROUND_CACHE[key] = (ground_z(x + c * half, y + sn * half, z), ground_z(x - c * half, y - sn * half, z))
    zf, zr = GROUND_CACHE[key]
    pitch = math.degrees(math.atan2(zf - zr, 2 * half))
    base = bb.location.z - bb.extent.z                       # box bottom relative to the actor origin
    return carla.Transform(carla.Location(x, y, (zf + zr) / 2 - base + 0.01), carla.Rotation(pitch=pitch, yaw=yaw))


# ---------------------------------------------------------------- road heights for every kinematic pose, before any actor
# exists (a projection started above a spawned vehicle would hit its own roof); box sizes from a parked copy
t_g = time.time()
_probe = {}
for a in (plan["actors"] if not NOVEL else []):
    if a["kind"] not in ("car", "bike"):
        continue
    if a["kind"] == "bike":
        bb_ = BIKE_BB
    else:
        tmp = world.try_spawn_actor(blueprint(a, 0), carla.Transform(carla.Location(0, 0, 500 + 10 * len(_probe))))
        bb_ = tmp.bounding_box; tmp.destroy(); _probe[a["name"]] = 1
    for k in (range(len(a["poses"])) if not a["static"] else [0]):
        kin_pose(a, k, bb_)
world.tick()
print("road heights: %d poses, %d misses, %.0f s" % (len(GROUND_CACHE), ground_miss[0], time.time() - t_g), flush=True)
ground_miss_precompute = ground_miss[0]

# ---------------------------------------------------------------- recorder (before the actors exist)
rec_file = os.path.join(OUT, "recording.log")
if not PROBE and not NOVEL:
    print("recorder:", client.start_recorder(rec_file, True), flush=True)
t_rec0 = world.get_snapshot().timestamp.elapsed_seconds

# ---------------------------------------------------------------- actors
actors, poses_k = {}, {}
spawn_list = plan["actors"] if not PROBE else [a for a in plan["actors"] if a["name"] in ("ego", "lead")]
if any(a["kind"] == "bike" for a in spawn_list):
    cyc = next(a for a in spawn_list if a["kind"] == "bike")
    spawn_list.append(dict(name="cyclist_rider", kind="rider", cls="cyclist", role="cyclist_rider", static=False,
                           poses=cyc["poses"], rides=cyc["name"]))
for i, a in enumerate(spawn_list):
    bp = blueprint(a, i)
    x, y, z, yaw, _ = a["poses"][0]
    if a["kind"] == "walker":
        zg = ground_z(x, y, z)
        act = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, zg + 1.0), carla.Rotation(yaw=yaw)))
    elif a["kind"] == "rider":
        act = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, z + 3.0), carla.Rotation(yaw=yaw)))
        if act is not None:
            act.set_simulate_physics(False)
    else:
        act = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, z + 2.0), carla.Rotation(yaw=yaw)))
        if act is not None:
            act.set_simulate_physics(False)
    if act is None:
        raise RuntimeError("spawn failed: %s (%s)" % (a["name"], bp.id))
    actors[a["name"]] = (a, act)
    print("spawned %-14s %-32s id %d" % (a["name"], bp.id, act.id), flush=True)
world.tick()
def place_kinematic(k):
    """Kinematic actors to plan pose k (riders follow their bike); returns nothing."""
    for name, (a, act) in actors.items():
        if a["kind"] in ("car", "bike") and (not a["static"] or k == 0):
            tf = kin_pose(a, k, BIKE_BB if a["kind"] == "bike" else act.bounding_box)
            act.set_transform(tf)
            if a["kind"] == "bike" and "cyclist_rider" in actors:
                actors["cyclist_rider"][1].set_transform(rider_tf(tf))


def compose(parent, child):
    """parent ∘ child for [x, y, z, pitch, yaw, roll] lists (Unreal convention) -> carla.Transform"""
    R = ue_rot(*parent[3:6])
    loc = [parent[i] + sum(R[i][j] * child[j] for j in range(3)) for i in range(3)]
    p, y, r = mat_euler([[sum(R[i][m] * ue_rot(*child[3:6])[m][j] for m in range(3)) for j in range(3)] for i in range(3)])
    return carla.Transform(carla.Location(*loc), carla.Rotation(pitch=p, yaw=y, roll=r))


def place_live(k):
    """re-simulation: kinematic actors exactly where the live run had them; the ego out of sight."""
    rows = {r["name"]: r["tf"] for r in LIVE[min(k, len(LIVE) - 1)]["actors"]}
    for name, (a, act) in actors.items():
        if name == "ego":
            act.set_transform(carla.Transform(carla.Location(0.0, 0.0, -1000.0)))
        elif a["kind"] in ("car", "bike", "rider") and name in rows:
            t = rows[name]
            act.set_transform(carla.Transform(carla.Location(*t[:3]), carla.Rotation(pitch=t[3], yaw=t[4], roll=t[5])))


if NOVEL:
    place_live(0)
else:
    place_kinematic(0)
if "cyclist_rider" in actors:
    pose_rider(actors["cyclist_rider"][1])
world.tick()
ego = actors["ego"][1]

# ---------------------------------------------------------------- sensors
def cam_bp(kind):
    b = bl.find(kind)
    b.set_attribute("image_size_x", str(W)); b.set_attribute("image_size_y", str(H)); b.set_attribute("fov", str(RIG["fov"]))
    if kind == "sensor.camera.rgb":
        for k, v in RIG["rgb_attributes"].items():
            if b.has_attribute(k):
                b.set_attribute(k, str(v))
    return b


def lidar_bp(kind):
    b = bl.find(kind)
    for k, v in RIG["lidar"]["attributes"].items():
        if b.has_attribute(k):
            b.set_attribute(k, str(v))
    return b


sensors, qs = [], {}
def attach(key, b, pose):
    loc, rot = pose[:3], pose[3:]
    sen = world.spawn_actor(b, carla.Transform(carla.Location(*loc), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])),
                            attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
    q = queue.Queue(); sen.listen(q.put); sensors.append(sen); qs[key] = q


FRONT = RIG["cameras"][0]
assert FRONT["name"] == "CAM_FRONT"
if NOVEL:                                                   # free cameras at lateral offsets of the front camera
    for dy in OFFS:
        for mod, kind in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth"),
                          ("inst", "sensor.camera.instance_segmentation")):
            sen = world.spawn_actor(cam_bp(kind), carla.Transform(carla.Location(0, 0, -500)))
            q = queue.Queue(); sen.listen(q.put); sensors.append(sen); qs[("%+.0f" % dy, mod)] = q
            os.makedirs(os.path.join(OUT, "novel", "off_%+.0f" % dy, mod), exist_ok=True)
    novel_cams = {key: sen for key, sen in zip(qs.keys(), sensors)}


    def place_novel(k):
        ego_tf = LIVE[min(k, len(LIVE) - 1)]["ego"]
        for (dy, mod), sen in novel_cams.items():
            mount = list(FRONT["pose"]); mount[1] += float(dy)
            sen.set_transform(compose(ego_tf, mount))
else:
    for cam in RIG["cameras"]:
        for mod, kind in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth"),
                          ("inst", "sensor.camera.instance_segmentation")):
            attach((cam["name"], mod), cam_bp(kind), cam["pose"])
            os.makedirs(os.path.join(OUT, "cam", cam["name"], mod), exist_ok=True)
    attach(("lidar", "lidar"), lidar_bp("sensor.lidar.ray_cast"), RIG["lidar"]["pose"])
    attach(("lidar", "sem"), lidar_bp("sensor.lidar.ray_cast_semantic"), RIG["lidar"]["pose"])
    for d in ("lidar", "semlidar"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)


def collect(frame):
    got = {}
    for key, q in qs.items():
        while True:
            d = q.get(timeout=120.0)
            if d.frame >= frame:
                break
        if d.frame != frame:
            raise RuntimeError("sensor %s skipped frame %d (got %d)" % (key, frame, d.frame))
        got[key] = d
    return got


# CARLA ue5-dev WalkerController.cpp: AddMovementInput(direction, Control.Speed / GetMaximumWalkSpeed()) with the
# maximum hard-coded to 4096 cm/s, while the line that would set the character's MaxWalkSpeed to it is disabled for
# UE5 ("#if 0 //@ Carla UE5"); the walker blueprints keep MaxWalkSpeed = 200 cm/s. A commanded speed v therefore walks
# at v * 200 / 4096 (measured: 0.0488 x, three walkers, rec03). The command is scaled back (reachable: <= 2.0 m/s).
WALKER_SPEED_GAIN = 4096.0 / 200.0


def walker_step(a, act, k):
    """Walk towards the planned position of tick k+1 (look-ahead 1 m); speed corrected for lag (+-30 %)."""
    tx, ty = a["poses"][min(k + 1, TICKS)][:2]
    nx, ny = a["poses"][min(k + 11, TICKS)][:2]
    loc = act.get_location()
    v_plan = math.hypot(nx - tx, ny - ty) / (min(k + 11, TICKS) - min(k + 1, TICKS) or 1) / DT
    dx, dy = tx - loc.x, ty - loc.y
    err = math.hypot(dx, dy)
    if v_plan < 0.05 and err < 0.25:
        return carla.WalkerControl(carla.Vector3D(0, 0, 0), 0.0)
    ax, ay = (nx - loc.x, ny - loc.y) if math.hypot(nx - loc.x, ny - loc.y) > 1.0 else (dx, dy)
    n = math.hypot(ax, ay) or 1.0
    sp = min(max(v_plan + 0.8 * err * (1 if (dx * ax + dy * ay) > 0 else -1), 0.0), max(v_plan * 1.3, 0.6))
    return carla.WalkerControl(carla.Vector3D(ax / n, ay / n, 0), sp * WALKER_SPEED_GAIN)


def actor_rows(snap):
    rows = []
    for name, (a, act) in actors.items():
        s_ = snap.find(act.id)
        bb = BIKE_BB if a["kind"] == "bike" else act.bounding_box
        v, w_ = s_.get_velocity(), s_.get_angular_velocity()
        rows.append(dict(name=name, id=act.id, type_id=act.type_id, cls=a["cls"], kind=a["kind"], static=a["static"],
                         tf=tf_list(s_.get_transform()), bbox=[bb.location.x, bb.location.y, bb.location.z,
                         bb.extent.x, bb.extent.y, bb.extent.z, bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll],
                         vel=[v.x, v.y, v.z], angvel=[w_.x, w_.y, w_.z]))
    return rows


meta = dict(plan=os.path.abspath(PLAN), map=cmap.name, dt=DT, ticks=TICKS, warmup=WARMUP, res=[W, H], rig=RIG,
            weather={k: getattr(weather, k) for k in dir(weather) if not k.startswith("_") and
                     isinstance(getattr(weather, k), (int, float))},
            traffic_lights=tl_state, actors={n: dict(id=act.id, type_id=act.type_id, role=a["role"], cls=a["cls"],
                                                      kind=a["kind"], static=a["static"])
                                              for n, (a, act) in actors.items()},
            recorder=None if (PROBE or NOVEL) else rec_file, novel_of=NOVEL, gpu_before=gpu_mem(),
            carla_client=getattr(carla, "__version__", None), axes="CARLA/UE: x forward, y right, z up, left-handed; "
            "degrees; lidar points in the sensor frame")
fj = open(os.path.join(OUT, "frames.jsonl"), "w")
t_start = time.time()
try:
    # warm-up: everybody holds the tick-0 pose (renderer history, sensor queues)
    if NOVEL:
        place_novel(0)
    for i in range(WARMUP):
        for name, (a, act) in actors.items():
            if a["kind"] == "walker":
                act.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 0.0))
        f = world.tick(); collect(f)
    if PROBE:
        # top-down semantic picture of the junction for the plan plot
        td = bl.find("sensor.camera.semantic_segmentation")
        for k_, v_ in (("image_size_x", "1400"), ("image_size_y", "1400"), ("fov", "60")):
            td.set_attribute(k_, v_)
        cam_td = world.spawn_actor(td, carla.Transform(carla.Location(430, 508, 84 + 150), carla.Rotation(pitch=-90, yaw=-90)))
        qtd = queue.Queue(); cam_td.listen(qtd.put)
        for _ in range(3):
            f = world.tick(); collect(f)
        img = qtd.get(timeout=60)
        img.save_to_disk(os.path.join(OUT, "topdown_seg.png"), carla.ColorConverter.CityScapesPalette)
        json.dump(dict(cx=430, cy=508, half=150 * math.tan(math.radians(30)), note="perspective from 150 m"),
                  open(os.path.join(OUT, "topdown_seg.json"), "w"))
        cam_td.stop(); cam_td.destroy()
        tick_times = []
        for i in range(10):
            t0 = time.time(); f = world.tick(); got = collect(f); tick_times.append(time.time() - t0)
        meta["probe"] = dict(tick_s=tick_times, gpu=gpu_mem(),
                             bikes=[b.id for b in bikes], n_walkers=len(walkers), walkers=[b.id for b in walkers][:40],
                             cars=[b.id for b in cars], ground_miss=ground_miss[0],
                             vehicles_all=sorted(b.id for b in bl.filter("vehicle.*")),
                             rgb_attributes={a.id: [str(a), list(a.recommended_values)] for a in bl.find("sensor.camera.rgb")},
                             rig_rgb_attributes_missing=[k for k in RIG["rgb_attributes"] if not bl.find("sensor.camera.rgb").has_attribute(k)],
                             ego_bbox=[ego.bounding_box.location.x, ego.bounding_box.location.y, ego.bounding_box.location.z,
                                       ego.bounding_box.extent.x, ego.bounding_box.extent.y, ego.bounding_box.extent.z],
                             ego_tf=tf_list(ego.get_transform()),
                             lidar_points=len(bytes(got[("lidar", "lidar")].raw_data)) // 16,
                             semlidar_points=len(bytes(got[("lidar", "sem")].raw_data)) // 24)
        for (cam, mod), d in got.items():
            if cam in ("CAM_FRONT", "CAM_BACK"):
                if mod in ("rgb", "inst"):
                    d.save_to_disk(os.path.join(OUT, "probe_%s_%s.png" % (cam, mod)))
                elif mod == "depth":
                    d.save_to_disk(os.path.join(OUT, "probe_%s_depth.png" % cam))
        # instance id encoding: which actor ids appear in the front instance image?
        inst = got[("CAM_FRONT", "inst")]
        raw = bytes(inst.raw_data)                           # BGRA
        vals = {}
        for p in range(0, len(raw), 4 * 7):
            b_, g_, r_ = raw[p], raw[p + 1], raw[p + 2]
            if r_ in (14, 15, 16, 18, 19, 12, 13):           # vehicle-ish / pedestrian tags (CARLA 0.9.14+ numbering)
                vals[(r_, g_, b_)] = vals.get((r_, g_, b_), 0) + 1
        meta["probe"]["inst_values"] = sorted([[k[0], k[1], k[2], n] for k, n in vals.items()], key=lambda r: -r[3])[:20]
        meta["probe"]["actor_ids"] = {n: act.id for n, (a, act) in actors.items()}
        print(json.dumps(meta["probe"], indent=1), flush=True)
    elif NOVEL:
        views, dev = [], {}
        for k in range(TICKS + 1):
            for name, (a, act) in actors.items():
                if a["kind"] == "walker":
                    act.apply_control(walker_step(a, act, k))
            place_live(k); place_novel(k)
            f = world.tick()
            got = collect(f)
            snap = world.get_snapshot()
            row = dict(tick=k, frame=f, t=snap.timestamp.elapsed_seconds, actors=actor_rows(snap))
            fj.write(json.dumps(row) + "\n")
            live_tf = {r["name"]: r["tf"] for r in LIVE[k]["actors"]}
            for r in row["actors"]:
                if r["name"] != "ego" and r["name"] in live_tf:
                    dev.setdefault(r["name"], []).append(math.dist(r["tf"][:3], live_tf[r["name"]][:3]))
            if k % EVERY == 0:
                cams_tf = {}
                for (dy, mod), d in got.items():
                    save_img(d, os.path.join(OUT, "novel", "off_%s" % dy, mod, "%04d.png" % k))
                    cams_tf["%s/%s" % (dy, mod)] = tf_list(d.transform)
                views.append(dict(frame=k, tick=k, cams=cams_tf))
            if k % 20 == 0:
                print("tick %d/%d  %.0f s" % (k, TICKS, time.time() - t_start), flush=True)
        fid = {n: dict(mean_m=round(sum(v) / len(v), 4), max_m=round(max(v), 4), n=len(v)) for n, v in dev.items()}
        print("re-simulation vs live, actor position deviation:", json.dumps(fid), flush=True)
        json.dump(dict(source="resimulation", live=NOVEL, every=EVERY, offsets=OFFS, replay_pose_error=fid,
                       rider_posed="cyclist_rider" in actors, frames=views), open(os.path.join(OUT, "replay.json"), "w"), indent=1)
    else:
        for k in range(TICKS + 1):
            for name, (a, act) in actors.items():
                if a["kind"] == "walker":
                    act.apply_control(walker_step(a, act, k))
            place_kinematic(k)
            f = world.tick()
            got = collect(f)
            snap = world.get_snapshot()
            assert snap.frame == f, (snap.frame, f)
            row = dict(tick=k, frame=f, t=snap.timestamp.elapsed_seconds, t_rec=snap.timestamp.elapsed_seconds - t_rec0,
                       ego=tf_list(snap.find(ego.id).get_transform()), sensors={}, actors=actor_rows(snap))
            for (cam, mod), d in got.items():
                row["sensors"]["%s/%s" % (cam, mod)] = tf_list(d.transform)
                if cam == "lidar":
                    name = "lidar" if mod == "lidar" else "semlidar"
                    open(os.path.join(OUT, name, "%04d.bin" % k), "wb").write(bytes(d.raw_data))
                else:
                    save_img(d, os.path.join(OUT, "cam", cam, mod, "%04d.png" % k))
            fj.write(json.dumps(row) + "\n")
            if k % 20 == 0:
                print("tick %d/%d  %.0f s  gpu %s" % (k, TICKS, time.time() - t_start, gpu_mem()), flush=True)
    meta["wall_s"] = time.time() - t_start
    meta["gpu_after"] = gpu_mem()
    meta["ground_miss"] = ground_miss[0]
    meta["ground_miss_precompute"] = ground_miss_precompute
    meta["ground_poses"] = len(GROUND_CACHE)
finally:
    fj.close()
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    if not PROBE and not NOVEL:
        client.stop_recorder()
    for sen in sensors:
        sen.stop(); sen.destroy()
    for name, (a, act) in actors.items():
        act.destroy()
    s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None; world.apply_settings(s)
    world.freeze_all_traffic_lights(False)
print("done", flush=True)
os._exit(0)   # the carla client can hang on interpreter exit
