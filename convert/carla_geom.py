"""CARLA -> right-handed conventions used by every converter (numpy, venv .venv).

CARLA / Unreal: left-handed, x forward, y right, z up, metres, Euler angles in degrees (pitch, yaw, roll).
Here:  world  = CARLA world with y negated (right-handed, z up)  ->  P_rh = S P_carla,  S = diag(1, -1, 1)
       body   = FLU (x forward, y left, z up): a CARLA actor / sensor frame with y negated
       camera = OpenCV (x right, y down, z forward)
A CARLA pose T (local -> world, both CARLA axes) becomes S4 T S4 (FLU local -> right-handed world)."""
import numpy as np

S4 = np.diag([1.0, -1.0, 1.0, 1.0])
FLU_FROM_CV = np.array([[0.0, 0.0, 1.0, 0.0],      # columns: OpenCV x (right) = -y_flu, y (down) = -z_flu,
                        [-1.0, 0.0, 0.0, 0.0],     # z (forward) = x_flu
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
SEM = {0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole", 7: "traffic_light",
       8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky", 12: "pedestrian", 13: "rider", 14: "car",
       15: "truck", 16: "bus", 17: "train", 18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic",
       22: "other", 23: "water", 24: "road_line", 25: "ground", 26: "bridge", 27: "rail_track", 28: "guard_rail"}
DYNAMIC_TAGS = (12, 13, 14, 15, 16, 17, 18, 19)    # classes that can move (masked out of the static background)


def carla_matrix(x, y, z, pitch, yaw, roll):
    """4x4 local -> world in CARLA axes; the same formula as carla::geom::Transform::GetMatrix."""
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
                     [sp, -cp * sr, cp * cr, z],
                     [0.0, 0.0, 0.0, 1.0]])


def rotation_matrix(tf, rotation="ue"):
    """4x4 local -> world in CARLA axes under the measured rotation convention: "ue" = carla_matrix (Unreal
    FRotator, positive pitch = nose up); "client" = this build's carla Transform.get_matrix() = D M D, D = diag(1,1,-1)."""
    M = carla_matrix(*tf)
    if rotation == "client":
        D = np.diag([1.0, 1.0, -1.0])
        M[:3, :3] = D @ M[:3, :3] @ D
    return M


def pose_rh(tf, rotation="ue"):
    """CARLA [x, y, z, pitch, yaw, roll] -> 4x4 FLU body -> right-handed world."""
    return S4 @ rotation_matrix(tf, rotation) @ S4


def cam_to_world_cv(tf, rotation="ue"):
    """CARLA sensor transform -> 4x4 OpenCV camera -> right-handed world."""
    return pose_rh(tf, rotation) @ FLU_FROM_CV


def intrinsics(w, h, fov_deg):
    """CARLA pinhole: f = w / (2 tan(fov/2)), principal point at the image centre (pixel-corner origin,
    i.e. the COLMAP / gsplat convention: the centre of pixel (0, 0) is (0.5, 0.5))."""
    f = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def decode_depth(rgb):
    """CARLA depth PNG (uint8 RGB) -> planar depth in metres (distance along the camera's forward axis)."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0


def decode_inst(rgb, order="gb"):
    """CARLA instance-segmentation PNG -> (semantic tag, instance id). order 'gb': id = G + 256 B."""
    tag = rgb[..., 0].astype(np.int32)
    g, b = rgb[..., 1].astype(np.int32), rgb[..., 2].astype(np.int32)
    return tag, (g + 256 * b) if order == "gb" else (256 * g + b)


def box_corners(bbox):
    """CARLA actor bounding box [lx, ly, lz, ex, ey, ez, pitch, yaw, roll] -> 8 corners in the actor frame
    (CARLA axes)."""
    lx, ly, lz, ex, ey, ez = bbox[:6]
    c = np.array([[sx * ex, sy * ey, sz * ez] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    R = carla_matrix(0, 0, 0, *bbox[6:9])[:3, :3]
    return c @ R.T + np.array([lx, ly, lz])
