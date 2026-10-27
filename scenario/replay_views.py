"""Re-render a recorded scenario from CARLA's recorder file at novel camera poses (venv38, server running).

Pass 1 (dry, no sensors): replay, read every replayed actor each tick, align replay ticks to the scenario frames
          (frames.jsonl) and measure the replayed actors' pose error against the live ground truth.
Pass 2 (render): replay again; the ego (hero) is not replayed and is parked far below the map (the reconstruction
          excludes the ego); the rider pose of the cyclist is re-applied (the recorder does not store bone poses);
          free cameras with the front camera's intrinsics follow the ego's live path at lateral offsets (ego frame,
          + = right) and render RGB, depth and instance segmentation every EVERY-th scenario frame.
Output (OUT/): novel/off_<dy>/{rgb,depth,inst}/NNNN.png, replay.json (alignment, pose errors, camera poses)
usage: replay_views.py REC_DIR OUT [--every 4] [--offsets -3,-2,-1,0,1,2,3]"""
import json, math, os, queue, sys, time, traceback
import carla

sys.excepthook = lambda *e: (traceback.print_exception(*e), sys.stdout.flush(), os._exit(1))
REC, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
EVERY = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 4
OFFS = [float(v) for v in (sys.argv[sys.argv.index("--offsets") + 1] if "--offsets" in sys.argv else "-3,-2,-1,0,1,2,3").split(",")]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
meta = json.load(open(os.path.join(REC, "meta.json")))
frames = [json.loads(l) for l in open(os.path.join(REC, "frames.jsonl"))]
RIDER = json.load(open(os.path.join(HERE, "rider_pose.json")))
DT = meta["dt"]
W, H = meta["res"]
CAM = meta["rig"]["cameras"][0]                                  # CAM_FRONT
assert CAM["name"] == "CAM_FRONT"
os.makedirs(OUT, exist_ok=True)


def ue_rot(p, y, r):
    cy, sy = math.cos(math.radians(y)), math.sin(math.radians(y))
    cr, sr = math.cos(math.radians(r)), math.sin(math.radians(r))
    cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr]]


def mat_euler(M):
    return (math.degrees(math.asin(max(-1.0, min(1.0, M[2][0])))), math.degrees(math.atan2(M[1][0], M[0][0])),
            math.degrees(math.atan2(-M[2][1], M[2][2])))


def mmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def compose(parent, child):
    """parent ∘ child for [x, y, z, pitch, yaw, roll] lists in Unreal's convention -> carla.Transform."""
    R = ue_rot(*parent[3:6])
    loc = [parent[i] + sum(R[i][j] * child[j] for j in range(3)) for i in range(3)]
    p, y, r = mat_euler(mmul(R, ue_rot(*child[3:6])))
    return carla.Transform(carla.Location(*loc), carla.Rotation(pitch=p, yaw=y, roll=r))


client = carla.Client("localhost", 2000); client.set_timeout(600.0)
world = client.get_world()
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = DT; world.apply_settings(s)
weather = world.get_weather()
for k, v in meta["weather"].items():
    try:
        setattr(weather, k, v)
    except Exception:
        pass
world.set_weather(weather)
bl = world.get_blueprint_library()
names = {v["id"]: n for n, v in meta["actors"].items()}
role_of = {n: v["role"] for n, v in meta["actors"].items()}
t_rec = [fr["t_rec"] for fr in frames]
n_ticks = int(round(t_rec[-1] / DT)) + 8


def replayed():
    """role name -> replayed actor (vehicles, walkers, props)."""
    out = {}
    for a in world.get_actors():
        rn = a.attributes.get("role_name", "")
        if a.type_id == "static.prop.mesh" and not rn:
            rn = "cyclist"                                       # the bike prop has no role_name attribute
        if rn and (a.type_id.startswith(("vehicle.", "walker.", "static.prop"))):
            out[rn] = a
    return out


def start_replay(ignore_hero):
    client.set_replayer_ignore_hero(ignore_hero)
    client.set_replayer_ignore_spectator(True)
    print(client.replay_file(meta["recorder"], 0.0, 0.0, 0, False), flush=True)
    world.tick()


# ---------------------------------------------------------------- pass 1: alignment + pose fidelity
start_replay(False)
lead_role = role_of["lead"]
gt_lead = [[a for a in fr["actors"] if a["name"] == "lead"][0]["tf"] for fr in frames]
trace = []
for j in range(n_ticks):
    world.tick()
    snap = {rn: a.get_transform() for rn, a in replayed().items()}
    trace.append({rn: [t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw, t.rotation.roll] for rn, t in snap.items()})
client.stop_replayer(False)
world.tick()
# tick j of the replay shows scenario frame k when the lead matches; offset c = j - k (constant)
best = None
for c in range(-5, n_ticks - len(frames) + 5):
    errs = []
    for k in range(0, len(frames), 10):
        j = k + c
        if 0 <= j < len(trace) and lead_role in trace[j]:
            p, q = trace[j][lead_role], gt_lead[k]
            errs.append(math.dist(p[:3], q[:3]))
    if errs and (best is None or max(errs) < best[0]):
        best = (max(errs), c)
OFFSET = best[1]
pose_err = {}
for k, fr in enumerate(frames):
    j = k + OFFSET
    for a in fr["actors"]:
        rn = role_of.get(a["name"])
        if 0 <= j < len(trace) and rn in trace[j]:
            e = math.dist(trace[j][rn][:3], a["tf"][:3])
            pose_err.setdefault(a["name"], []).append(e)
fid = {n: dict(mean_m=round(sum(v) / len(v), 4), max_m=round(max(v), 4), n=len(v)) for n, v in pose_err.items()}
print("alignment: replay tick = frame + %d (lead max error %.4f m)" % (OFFSET, best[0]), flush=True)
print("replay pose error vs live GT:", json.dumps(fid), flush=True)

# ---------------------------------------------------------------- pass 2: novel views
start_replay(True)
for dy in OFFS:
    for m in ("rgb", "depth", "inst"):
        os.makedirs(os.path.join(OUT, "novel", "off_%+.0f" % dy, m), exist_ok=True)
sensors, qs = [], {}
for dy in OFFS:
    for m, kind in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth"), ("inst", "sensor.camera.instance_segmentation")):
        b = bl.find(kind)
        for k_, v_ in (("image_size_x", str(W)), ("image_size_y", str(H)), ("fov", str(meta["rig"]["fov"]))):
            b.set_attribute(k_, v_)
        sen = world.spawn_actor(b, carla.Transform(carla.Location(0, 0, -500)))
        q = queue.Queue(); sen.listen(q.put); sensors.append((dy, m, sen)); qs[(dy, m)] = q
cam_rows, hidden, posed = [], False, False
t0 = time.time()
try:
    for j in range(len(frames) + OFFSET + 1):
        k = j - OFFSET                                           # scenario frame shown after this tick
        rp = replayed()
        hero = rp.get("hero")
        if hero is not None:
            hero.set_transform(carla.Transform(carla.Location(0, 0, -1000)))
        rider = rp.get("cyclist_rider")
        if rider is not None and not posed:
            rel = {b.name: b.relative for b in rider.get_bones().bone_transforms}
            out = []
            for n, d in RIDER["bones"].items():
                t = rel[n]
                p, y_, r = mat_euler(mmul(ue_rot(t.rotation.pitch, t.rotation.yaw, t.rotation.roll), ue_rot(*d)))
                out.append((n, carla.Transform(t.location, carla.Rotation(pitch=p, yaw=y_, roll=r))))
            rider.set_bones(carla.WalkerBoneControlIn(out)); rider.blend_pose(1.0); posed = True
        kk = min(max(k, 0), len(frames) - 1)
        ego = frames[kk]["ego"]
        for dy, m, sen in sensors:
            mount = list(CAM["pose"]); mount[1] += dy
            sen.set_transform(compose(ego, mount))
        f = world.tick()
        got = {}
        for key, q in qs.items():
            while True:
                d = q.get(timeout=120.0)
                if d.frame >= f:
                    break
            got[key] = d
        if 0 <= k < len(frames) and k % EVERY == 0:
            row = dict(frame=k, tick=j, cams={})
            for (dy, m), d in got.items():
                d.save_to_disk(os.path.join(OUT, "novel", "off_%+.0f" % dy, m, "%04d.png" % k))
                t = d.transform
                row["cams"]["%+.0f/%s" % (dy, m)] = [t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw, t.rotation.roll]
            row["actors"] = {rn: [a.get_transform().location.x, a.get_transform().location.y, a.get_transform().location.z]
                             for rn, a in replayed().items() if rn != "hero"}
            cam_rows.append(row)
            if k % 20 == 0:
                print("frame %d  %.0f s" % (k, time.time() - t0), flush=True)
finally:
    json.dump(dict(rec=REC, every=EVERY, offsets=OFFS, alignment_offset=OFFSET, lead_align_err_m=best[0],
                   replay_pose_error=fid, rider_posed=posed, frames=cam_rows, wall_s=time.time() - t0),
              open(os.path.join(OUT, "replay.json"), "w"), indent=1)
    for _, _, sen in sensors:
        sen.stop(); sen.destroy()
    client.stop_replayer(False)
    s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None; world.apply_settings(s)
print("done", flush=True)
os._exit(0)
