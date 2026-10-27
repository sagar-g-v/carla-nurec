"""Can a static.prop.mesh actor be moved kinematically in this CARLA build? (venv38, server running)
Variants: (a) plain spawn + set_transform, (b) spawned with mass > 0 (physics -> movable) then physics off +
set_transform, (c) attached to a walker that is moved. Reports the transform the server returns after each move and
saves a picture of each variant before / after.
usage: prop_motion_test.py OUT"""
import json, os, queue, sys, traceback
import carla

sys.excepthook = lambda *e: (traceback.print_exception(*e), sys.stdout.flush(), os._exit(1))
OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
MESH = "/Game/Carla/Static/Bicycle/CrossBike/SM_sc_crossbike.SM_sc_crossbike"
client = carla.Client("localhost", 2000); client.set_timeout(600.0)
world = client.get_world()
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.1; world.apply_settings(s)
bl = world.get_blueprint_library()
pb = bl.find("static.prop.mesh")
attrs = {a.id: str(a) for a in pb}
print("static.prop.mesh attributes:", json.dumps(attrs), flush=True)
gz = world.ground_projection(carla.Location(428, 506, 90), 20).location.z
res, actors = {}, []


def cam(loc, yaw):
    b = bl.find("sensor.camera.rgb")
    for k, v in (("image_size_x", "640"), ("image_size_y", "360"), ("fov", "70")):
        b.set_attribute(k, v)
    c = world.spawn_actor(b, carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=-10)))
    q = queue.Queue(); c.listen(q.put); actors.append(c)
    return q


try:
    variants = {}
    b = bl.find("static.prop.mesh"); b.set_attribute("mesh_path", MESH)
    variants["a_plain"] = world.spawn_actor(b, carla.Transform(carla.Location(422, 500, gz + 0.02)))
    b = bl.find("static.prop.mesh"); b.set_attribute("mesh_path", MESH)
    if b.has_attribute("mass"):
        b.set_attribute("mass", "15")
    variants["b_mass"] = world.spawn_actor(b, carla.Transform(carla.Location(422, 504, gz + 0.3)))
    wb = bl.find("walker.pedestrian.0015")
    walker = world.spawn_actor(wb, carla.Transform(carla.Location(422, 508, gz + 1.0)))
    walker.set_simulate_physics(False)
    b = bl.find("static.prop.mesh"); b.set_attribute("mesh_path", MESH)
    variants["c_attached"] = world.spawn_actor(b, carla.Transform(carla.Location(0, 0, -0.9)), attach_to=walker,
                                               attachment_type=carla.AttachmentType.Rigid)
    actors += list(variants.values()) + [walker]
    for _ in range(5):
        world.tick()
    variants["b_mass"].set_simulate_physics(False)
    q = cam(carla.Location(422, 494, gz + 4.0), 90)
    for _ in range(3):
        world.tick()
    start = {k: variants[k].get_transform().location for k in variants}
    img0 = None
    while not q.empty():
        img0 = q.get()
    img0.save_to_disk(os.path.join(OUT, "props_before.png"))
    for step in range(1, 11):                              # move everything +0.5 m in x per tick
        for k in ("a_plain", "b_mass"):
            t = variants[k].get_transform(); t.location.x = start[k].x + 0.5 * step
            variants[k].set_transform(t)
        t = walker.get_transform(); t.location.x = 422 + 0.5 * step
        walker.set_transform(t)
        world.tick()
    img1 = None
    while not q.empty():
        img1 = q.get()
    img1.save_to_disk(os.path.join(OUT, "props_after.png"))
    for k, a in variants.items():
        e = a.get_transform().location
        res[k] = dict(start_x=round(start[k].x, 3), end_x=round(e.x, 3), moved_m=round(e.x - start[k].x, 3))
    print(json.dumps(res), flush=True)
    json.dump(dict(attributes=attrs, result=res), open(os.path.join(OUT, "prop_motion.json"), "w"), indent=1)
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
