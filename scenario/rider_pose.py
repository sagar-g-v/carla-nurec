"""Build a cycling pose for a CARLA pedestrian on CARLA's CrossBike static mesh (venv38, server running).
This build registers no 2-wheeled vehicle, so the cyclist is the bike mesh (static.prop.mesh) plus a walker posed
through the bone API (set_bones / blend_pose), both moved kinematically by record.py.
  rider_pose.py OUT candidates          side-view renders of single-bone rotations (find the bone axes) + skeleton
  rider_pose.py OUT pose POSE.json      the full pose on the bike, rendered from three sides
POSE.json: {"walker": id, "seat": [x, y, z] walker origin in the bike frame (CARLA axes),
            "bones": {bone: [pitch, yaw, roll] rotation applied in the bone's own frame}}"""
import json, math, os, queue, sys, traceback
import carla

sys.excepthook = lambda *e: (traceback.print_exception(*e), sys.stdout.flush(), os._exit(1))
OUT, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BIKE_MESH = "/Game/Carla/Static/Bicycle/CrossBike/SM_sc_crossbike.SM_sc_crossbike"
client = carla.Client("localhost", 2000); client.set_timeout(600.0)
world = client.get_world()
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
bl = world.get_blueprint_library()
w = world.get_weather(); w.sun_altitude_angle = 55.0; w.sun_azimuth_angle = 200.0; w.cloudiness = 10.0; world.set_weather(w)
BASE = carla.Location(428.0, 506.0, 0.0)                    # junction centre (open, sunlit)
gz = world.ground_projection(carla.Location(BASE.x, BASE.y, 90.0), 20.0).location.z


# rotation algebra in Unreal's convention (measured: convert/conventions.json)
def mat(p, y, r):
    cy, sy = math.cos(math.radians(y)), math.sin(math.radians(y))
    cr, sr = math.cos(math.radians(r)), math.sin(math.radians(r))
    cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr]]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def euler(M):
    p = math.degrees(math.asin(max(-1.0, min(1.0, M[2][0]))))
    return p, math.degrees(math.atan2(M[1][0], M[0][0])), math.degrees(math.atan2(-M[2][1], M[2][2]))


def posed(rel, delta):
    """relative bone transform with an extra rotation (pitch, yaw, roll) in the bone's own frame."""
    p, y, r = euler(mul(mat(rel.rotation.pitch, rel.rotation.yaw, rel.rotation.roll), mat(*delta)))
    return carla.Transform(rel.location, carla.Rotation(pitch=p, yaw=y, roll=r))


def spawn_walker(wid, loc, yaw):
    b = bl.find("walker.pedestrian.%s" % wid)
    if b.has_attribute("is_invincible"):
        b.set_attribute("is_invincible", "false")
    for dz in (0.0, 0.3, 0.6, 1.0):
        a = world.try_spawn_actor(b, carla.Transform(loc + carla.Location(z=dz), carla.Rotation(yaw=yaw)))
        if a is not None:
            a.set_simulate_physics(False)
            return a
    raise RuntimeError("walker spawn failed at %s" % loc)


def apply_pose(walker, bones):
    rel = {b.name: b.relative for b in walker.get_bones().bone_transforms}
    walker.set_bones(carla.WalkerBoneControlIn([(n, posed(rel[n], d)) for n, d in bones.items() if n in rel]))
    walker.blend_pose(1.0)


def camera(tf, w=1280, h=720, fov=60):
    b = bl.find("sensor.camera.rgb")
    for k, v in (("image_size_x", str(w)), ("image_size_y", str(h)), ("fov", str(fov))):
        b.set_attribute(k, v)
    c = world.spawn_actor(b, tf); q = queue.Queue(); c.listen(q.put)
    return c, q


actors = []
try:
    if MODE == "candidates":
        wk = spawn_walker("0015", carla.Location(BASE.x - 8.0, BASE.y - 6.0, gz + 1.0), 0.0)
        actors.append(wk)
        for _ in range(3):
            world.tick()
        bones = wk.get_bones().bone_transforms
        json.dump([dict(name=b.name, rel=[b.relative.location.x, b.relative.location.y, b.relative.location.z,
                                          b.relative.rotation.pitch, b.relative.rotation.yaw, b.relative.rotation.roll],
                        comp=[b.component.location.x, b.component.location.y, b.component.location.z])
                   for b in bones], open(os.path.join(OUT, "skeleton.json"), "w"), indent=1)
        names = [b.name for b in bones]
        print("bones", len(names), names, flush=True)
        pick = lambda key: next((n for n in names if key in n.lower()), None)
        tests = [pick("thigh__l"), pick("leg__l"), pick("spine__c"), pick("arm__l")]
        rows = []
        for bi, bone in enumerate(tests):
            for ci, delta in enumerate(((60, 0, 0), (-60, 0, 0), (0, 60, 0), (0, -60, 0), (0, 0, 60), (0, 0, -60))):
                loc = carla.Location(BASE.x + ci * 1.6, BASE.y + bi * 2.5, gz + 1.0)
                a = spawn_walker("0015", loc, 0.0)                # facing +x: seen from the side by the camera
                actors.append(a); rows.append(dict(bone=bone, delta=delta, col=ci, row=bi))
        _b = bl.find("static.prop.mesh"); _b.set_attribute("mesh_path", BIKE_MESH)
        bike = world.spawn_actor(_b, carla.Transform(carla.Location(BASE.x - 2.5, BASE.y, gz + 0.02)))
        actors.append(bike)
        world.tick()
        for a, r in zip(actors[1:-1], rows):
            apply_pose(a, {r["bone"]: r["delta"]})
        cam, q = camera(carla.Transform(carla.Location(BASE.x + 4.0, BASE.y - 9.0, gz + 1.6), carla.Rotation(yaw=90, pitch=-8)), fov=75)
        actors.append(cam)
        for _ in range(6):
            world.tick()
        img = None
        while not q.empty():
            img = q.get()
        img.save_to_disk(os.path.join(OUT, "candidates.png"))
        json.dump(dict(rows=rows, note="columns: (+60,0,0) (-60,0,0) (0,+60,0) (0,-60,0) (0,0,+60) (0,0,-60) as (pitch,yaw,roll); rows: bones",
                       bike_bp_attrs=[a.id for a in _b]), open(os.path.join(OUT, "candidates.json"), "w"), indent=1)
    else:
        pose = json.load(open(sys.argv[3]))
        b = bl.find("static.prop.mesh"); b.set_attribute("mesh_path", BIKE_MESH)
        bike = world.spawn_actor(b, carla.Transform(carla.Location(BASE.x, BASE.y, gz + pose.get("bike_z", 0.0)), carla.Rotation(yaw=0)))
        actors.append(bike)
        sx, sy, sz = pose["seat"]
        wk = spawn_walker(str(pose["walker"]), carla.Location(BASE.x + sx, BASE.y + sy, gz + sz), 0.0)
        actors.append(wk)
        world.tick()
        apply_pose(wk, pose["bones"])
        views = {"right": (carla.Location(BASE.x, BASE.y + 4.5, gz + 1.2), -90), "left": (carla.Location(BASE.x, BASE.y - 4.5, gz + 1.2), 90),
                 "front": (carla.Location(BASE.x + 4.5, BASE.y, gz + 1.2), 180)}
        cams = {}
        for k, (loc, yaw) in views.items():
            c, q = camera(carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=-5)), 640, 480, 50); actors.append(c); cams[k] = q
        for _ in range(6):
            world.tick()
        for k, q in cams.items():
            img = None
            while not q.empty():
                img = q.get()
            img.save_to_disk(os.path.join(OUT, "pose_%s.png" % k))
        bb, wb = bike.bounding_box, wk.bounding_box
        json.dump(dict(bike_bbox=[bb.location.x, bb.location.y, bb.location.z, bb.extent.x, bb.extent.y, bb.extent.z],
                       walker_bbox=[wb.location.x, wb.location.y, wb.location.z, wb.extent.x, wb.extent.y, wb.extent.z],
                       bike_tf=[bike.get_transform().location.x, bike.get_transform().location.y, bike.get_transform().location.z],
                       ground_z=gz), open(os.path.join(OUT, "pose_info.json"), "w"), indent=1)
finally:
    for a in actors[::-1]:
        try:
            if a.type_id.startswith("sensor"):
                a.stop()
            a.destroy()
        except RuntimeError:
            pass
    s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None; world.apply_settings(s)
print("done", flush=True)
os._exit(0)
