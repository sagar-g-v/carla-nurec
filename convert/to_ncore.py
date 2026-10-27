"""Reconstruction dataset (to_dataset.py) -> NCore V4 sequence, the input format of NVIDIA NuRec (and 3DGRUT).
Uses nvidia-ncore (Apache-2.0). Frames (right-handed, from dataset.json): rig = FLU, camera = OpenCV, world = CARLA
world with y negated; NCore poses T_a_b map frame a into frame b (p_b = R p_a + t).
  poses     dynamic ("rig", "world") every tick; static (<sensor>, "rig") per sensor
  cameras   camera_front ... camera_back: opencv-pinhole without distortion, global shutter, the recorded PNGs
  lidar     lidar_top: row-offset-spinning model (64 rows, CARLA's uniform elevations), one sweep per tick;
            rays = unit directions in the lidar frame + distance + intensity (CARLA computes a sweep at one instant)
  cuboids   every actor (ego excluded) every tick, reference frame "world", source GT_SYNTHETIC; classes in NuRec's
            vocabulary: automobile / pedestrian / cyclist (the cyclist box covers bike + rider)
Then the sequence is read back with NCore's loader and the cuboids are drawn into one camera image (check).
usage: to_ncore.py DATASET_DIR OUT_DIR [--name carla_la_dtla_j10295]"""
import io, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from upath import UPath
from ncore.impl.common.transformations import HalfClosedInterval
from ncore.data import (BBox3, CuboidTrackObservation, LabelSource, ShutterType, OpenCVPinholeCameraModelParameters,
                        RowOffsetStructuredSpinningLidarModelParameters)
from ncore.data.v4 import (SequenceComponentGroupsWriter, SequenceComponentGroupsReader, SequenceLoaderV4, PosesComponent,
                           IntrinsicsComponent, MasksComponent, CameraSensorComponent, LidarSensorComponent, CuboidsComponent)

DATA, OUT = sys.argv[1], sys.argv[2]
NAME = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "carla_la_dtla_j10295"
D = json.load(open(os.path.join(DATA, "dataset.json")))
src_meta = json.load(open(os.path.join(D["source"], "meta.json")))
os.makedirs(OUT, exist_ok=True)
CAM_ID = {"CAM_FRONT": "camera_front", "CAM_FRONT_LEFT": "camera_front_left", "CAM_FRONT_RIGHT": "camera_front_right",
          "CAM_BACK_LEFT": "camera_back_left", "CAM_BACK_RIGHT": "camera_back_right", "CAM_BACK": "camera_back"}
LID = "lidar_top"
CLASS = {"car": "automobile", "pedestrian": "pedestrian", "cyclist": "cyclist", "bicycle": "cyclist"}
frames = D["frames"]
ts = [int(round(f["t"] * 1e6)) for f in frames]
assert all(b > a for a, b in zip(ts, ts[1:]))
interval = HalfClosedInterval.from_start_end(ts[0], ts[-1])
w = SequenceComponentGroupsWriter(output_dir_path=UPath(OUT), store_base_name=NAME, sequence_id=NAME,
                                  sequence_timestamp_interval_us=interval, store_type="itar",
                                  generic_meta_data=dict(source="CARLA 0.10 (ue5-dev) LA_DTLA_ODR, scenario/record.py",
                                                         world="CARLA world with y negated (right-handed, z up, m)",
                                                         xodr=json.load(open(src_meta["plan"]))["meta"]["xodr"]))

# ---------------------------------------------------------------- poses
poses = w.register_component_writer(PosesComponent.Writer, "default")
T_world_rig = np.stack([np.array(f["T_world_rig"]) for f in frames])
poses.store_dynamic_pose("rig", "world", T_world_rig, np.array(ts, np.uint64))
dev = {}
for c, cid in CAM_ID.items():
    Ts = np.stack([np.linalg.inv(np.array(f["T_world_rig"])) @ np.array(f["cams"][c]["T_world_cam"]) for f in frames])
    dev[cid] = float(np.abs(Ts - Ts[0]).max())                         # rigid mount: must be constant
    poses.store_static_pose(cid, "rig", Ts[0])
Tl = np.stack([np.linalg.inv(np.array(f["T_world_rig"])) @ np.array(f["lidar"]["T_world_lidar"]) for f in frames])
dev[LID] = float(np.abs(Tl - Tl[0]).max())
poses.store_static_pose(LID, "rig", Tl[0])
assert max(dev.values()) < 1e-3, dev

# ---------------------------------------------------------------- intrinsics, masks
intr = w.register_component_writer(IntrinsicsComponent.Writer, "default")
masks = w.register_component_writer(MasksComponent.Writer, "default")
for c, cid in CAM_ID.items():
    K = np.array(D["cameras"][c]["K"])
    intr.store_camera_intrinsics(cid, OpenCVPinholeCameraModelParameters(
        resolution=np.array([D["cameras"][c]["width"], D["cameras"][c]["height"]], np.uint64), shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([K[0, 2], K[1, 2]], np.float32), focal_length=np.array([K[0, 0], K[1, 1]], np.float32),
        radial_coeffs=np.zeros(6, np.float32), tangential_coeffs=np.zeros(2, np.float32), thin_prism_coeffs=np.zeros(4, np.float32)))
    masks.store_camera_masks(cid, {})                                   # no static ego mask (cameras see no ego body)
la = src_meta["rig"]["lidar"]["attributes"]
n_rows = int(la["channels"])
n_cols = int(round(la["points_per_second"] / la["rotation_frequency"] / n_rows))
intr.store_lidar_intrinsics(LID, RowOffsetStructuredSpinningLidarModelParameters(
    spinning_frequency_hz=float(la["rotation_frequency"]), spinning_direction="cw",  # CARLA sweeps towards +y (right)
    n_rows=n_rows, n_columns=n_cols,
    row_elevations_rad=np.radians(np.linspace(la["upper_fov"], la["lower_fov"], n_rows)).astype(np.float32),
    column_azimuths_rad=np.linspace(np.pi, -np.pi, n_cols, endpoint=False).astype(np.float32),
    row_azimuth_offsets_rad=np.zeros(n_rows, np.float32)))

# ---------------------------------------------------------------- sensor frames
cams = {cid: w.register_component_writer(CameraSensorComponent.Writer, cid) for cid in CAM_ID.values()}
lid = w.register_component_writer(LidarSensorComponent.Writer, LID)
LIDAR_AXES = np.diag(D["conventions"]["lidar_axes"])
n_pts = []
for f, t in zip(frames, ts):
    for c, cid in CAM_ID.items():
        cams[cid].store_frame(image_binary_data=open(f["cams"][c]["rgb"], "rb").read(), image_format="png",
                              frame_timestamps_us=np.array([t, t], np.uint64), generic_data={}, generic_meta_data={})
    raw = np.fromfile(f["lidar"]["path"], np.float32).reshape(-1, 4)
    P = raw[:, :3].astype(np.float64) @ LIDAR_AXES.T * [1.0, -1.0, 1.0]  # CARLA sensor axes -> FLU
    r = np.linalg.norm(P, axis=1); ok = r > 1e-3
    n_pts.append(int(ok.sum()))
    lid.store_frame(direction=(P[ok] / r[ok, None]).astype(np.float32), timestamp_us=np.full(ok.sum(), t, np.uint64),
                    model_element=None, distance_m=r[ok][None].astype(np.float32),
                    intensity=np.clip(raw[ok, 3], 0, 1)[None].astype(np.float32),
                    frame_timestamps_us=np.array([t, t], np.uint64), generic_data={}, generic_meta_data={})

# ---------------------------------------------------------------- cuboids
obs = []
tick_ts = {f["tick"]: t for f, t in zip(frames, ts)}
for aid, a in D["actors"].items():
    off = np.array(a.get("box_offset", [0.0, 0.0, 0.0]))
    for k, T in a["track"].items():
        T = np.array(T); c = T[:3, :3] @ off + T[:3, 3]
        rot = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
        obs.append(CuboidTrackObservation(
            track_id="%s_%s" % (a["name"], aid), class_id=CLASS.get(a["cls"], a["cls"]), timestamp_us=int(tick_ts[int(k)]),
            reference_frame_id="world", reference_frame_timestamp_us=int(tick_ts[int(k)]),
            bbox3=BBox3(centroid=tuple(float(v) for v in c), dim=tuple(float(v) for v in a["size"]), rot=tuple(float(v) for v in rot)),
            source=LabelSource.GT_SYNTHETIC, source_version="carla-0.10-ue5dev"))
w.register_component_writer(CuboidsComponent.Writer, "default").store_observations(obs)
paths = w.finalize()
manifest = UPath(OUT) / ("%s.json" % NAME)
manifest.write_text(json.dumps(SequenceComponentGroupsReader(paths).get_sequence_meta().to_dict(), indent=2))
print("wrote", [str(p) for p in paths], manifest, "cuboid observations", len(obs), "lidar points/frame median", int(np.median(n_pts)))

# ---------------------------------------------------------------- read back + cuboid overlay (convention check)
L = SequenceLoaderV4(SequenceComponentGroupsReader([manifest]))
cam = L.get_camera_sensor("camera_front")
fi = len(frames) // 2
img = Image.fromarray(cam.get_frame_image_array(fi)); dr = ImageDraw.Draw(img)
T_cam_world = np.linalg.inv(np.array(cam.get_frames_T_sensor_target("world", fi)).reshape(4, 4))
p = cam.model_parameters
fx, fy = [float(v) for v in p.focal_length]; cx, cy = [float(v) for v in p.principal_point]
t_mid = ts[fi]
drawn = 0
for o in L.get_cuboid_track_observations():
    if int(o.timestamp_us) != t_mid:
        continue
    b = o.bbox3
    R = Rotation.from_euler("xyz", b.rot).as_matrix()
    corners = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * np.array(b.dim)
    Pw = corners @ R.T + np.array(b.centroid)
    Pc = Pw @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    if (Pc[:, 2] < 0.5).any():
        continue
    uv = np.stack([fx * Pc[:, 0] / Pc[:, 2] + cx, fy * Pc[:, 1] / Pc[:, 2] + cy], -1)
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                dr.line([tuple(uv[i]), tuple(uv[j])], fill=(255, 40, 40), width=2)
    drawn += 1
img.save(os.path.join(OUT, "check_cuboids_camera_front_%04d.png" % frames[fi]["tick"]))
info = dict(manifest=str(manifest), stores=[str(p) for p in paths], cameras=list(CAM_ID.values()), lidar=LID,
            frames=len(frames), interval_us=[ts[0], ts[-1] + 1], cuboid_observations=len(obs), mount_deviation=dev,
            readback=dict(camera_frames=cam.frames_count, lidar_frames=L.get_lidar_sensor(LID).frames_count, boxes_drawn=drawn))
json.dump(info, open(os.path.join(OUT, "ncore_info.json"), "w"), indent=1)
print(json.dumps(info["readback"]))
