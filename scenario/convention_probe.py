"""Which rotation convention does the CARLA server render with? (venv38, server running)
Semantic cameras 600 m above the map with known pitch / yaw / roll; the sky / not-sky split of each image is saved
with the transform the server reports, and the lidar of a pitched rig is saved too. convention_check.py (.venv)
predicts the horizon for both candidate matrices and scores them.
usage: convention_probe.py OUT"""
import json, os, queue, sys, traceback
import carla

sys.excepthook = lambda *e: (traceback.print_exception(*e), sys.stdout.flush(), os._exit(1))

OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
client = carla.Client("localhost", 2000); client.set_timeout(600.0)
world = client.get_world()
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
bl = world.get_blueprint_library()
CASES = [(0, 0, 0), (20, 0, 0), (-20, 0, 0), (0, 0, 20), (10, 30, 20), (-15, -60, -25)]
cams, qs = [], []
for i, (p, y, r) in enumerate(CASES):
    b = bl.find("sensor.camera.semantic_segmentation")
    for k, v in (("image_size_x", "320"), ("image_size_y", "240"), ("fov", "90")):
        b.set_attribute(k, v)
    c = world.spawn_actor(b, carla.Transform(carla.Location(430 + 5 * i, 508, 684), carla.Rotation(pitch=p, yaw=y, roll=r)))
    q = queue.Queue(); c.listen(q.put); cams.append(c); qs.append(q)
# a semantic lidar with pitch +20 at 2 m above the road at the ego start, plus a depth camera at the same pose
start = carla.Transform(carla.Location(366.9, 580.7, 85.3), carla.Rotation(pitch=20, yaw=-53, roll=10))
lb = bl.find("sensor.lidar.ray_cast_semantic")
for k, v in (("channels", "32"), ("range", "60"), ("points_per_second", "320000"), ("rotation_frequency", "10"),
             ("upper_fov", "10"), ("lower_fov", "-30")):
    lb.set_attribute(k, v)
lid = world.spawn_actor(lb, start); ql = queue.Queue(); lid.listen(ql.put)
db = bl.find("sensor.camera.depth")
for k, v in (("image_size_x", "640"), ("image_size_y", "360"), ("fov", "90")):
    db.set_attribute(k, v)
dep = world.spawn_actor(db, start); qd = queue.Queue(); dep.listen(qd.put)
try:
    for _ in range(8):
        f = world.tick()
    res = []
    for (p, y, r), q, c in zip(CASES, qs, cams):
        img = None
        while not q.empty():
            img = q.get()
        raw = bytes(img.raw_data)
        sky = bytearray(1 if raw[i + 2] == 11 else 0 for i in range(0, len(raw), 4))
        name = "cam_p%d_y%d_r%d" % (p, y, r)
        open(os.path.join(OUT, name + ".sky"), "wb").write(bytes(sky))
        t = img.transform
        res.append(dict(name=name, set=[p, y, r], reported=[t.location.x, t.location.y, t.location.z,
                        t.rotation.pitch, t.rotation.yaw, t.rotation.roll], w=img.width, h=img.height, fov=90))
    L = None
    while not ql.empty():
        L = ql.get()
    open(os.path.join(OUT, "lidar.bin"), "wb").write(bytes(L.raw_data))
    D = None
    while not qd.empty():
        D = qd.get()
    D.save_to_disk(os.path.join(OUT, "depth.png"))
    t, td = L.transform, D.transform
    json.dump(dict(cams=res, lidar=[t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw, t.rotation.roll],
                   depth=[td.location.x, td.location.y, td.location.z, td.rotation.pitch, td.rotation.yaw, td.rotation.roll],
                   depth_wh=[D.width, D.height], lidar_frame=L.frame, depth_frame=D.frame),
              open(os.path.join(OUT, "convention.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
finally:
    for a in cams + [lid, dep]:
        a.stop(); a.destroy()
    s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None; world.apply_settings(s)
print("done", flush=True)
os._exit(0)
