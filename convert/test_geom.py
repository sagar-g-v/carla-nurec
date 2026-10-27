"""The carla wheel's client-side Transform.get_matrix() of this ue5-dev build vs carla_geom: it equals
rotation_matrix(tf, "client") = D M D (D = diag(1, 1, -1)), i.e. the opposite pitch / roll sign of the Unreal
convention the server renders with (convention_check.py). Geometry must use rotation_matrix(tf, "ue").
  venv38:  test_geom.py dump OUT.json     (writes random transforms with CARLA's matrices)
  .venv :  test_geom.py check OUT.json    (fails above 1e-5)"""
import json, random, sys

if sys.argv[1] == "dump":
    import carla
    random.seed(3)
    rows = []
    for _ in range(200):
        tf = [random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-50, 50),
              random.uniform(-89, 89), random.uniform(-180, 180), random.uniform(-180, 180)]
        t = carla.Transform(carla.Location(*tf[:3]), carla.Rotation(pitch=tf[3], yaw=tf[4], roll=tf[5]))
        rows.append(dict(tf=tf, m=[list(r) for r in t.get_matrix()]))
    json.dump(rows, open(sys.argv[2], "w"))
    print("dumped", len(rows))
else:
    import os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from carla_geom import rotation_matrix
    rows = json.load(open(sys.argv[2]))
    err = max(np.abs(rotation_matrix(r["tf"], "client") - np.array(r["m"])).max() / max(1.0, np.abs(r["tf"][:3]).max()) for r in rows)
    ue = max(np.abs(rotation_matrix(r["tf"], "ue") - np.array(r["m"]))[:3, :3].max() for r in rows)
    print("client get_matrix vs D M D: max relative error %.2e; vs the Unreal convention: max abs difference %.2f" % (err, ue))
    sys.exit(0 if err < 1e-5 else 1)
