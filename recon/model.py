"""Composite 3D Gaussian scene (gsplat 3DGUT): a static background plus one rigid Gaussian set per dynamic actor,
kept in the actor's box frame and placed by its 3D box in every frame (the static / dynamic split NuRec makes from
tracked cuboids), and a learnable sky (equirectangular map) behind everything. Shared by train.py and evaluate/."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization


def quat_from_mat(R):
    """rotation matrices (..., 3, 3) -> quaternions wxyz (...)"""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = torch.sqrt(torch.clamp(1 + t, min=1e-8)) / 2
    x = torch.sqrt(torch.clamp(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=1e-8)) / 2
    y = torch.sqrt(torch.clamp(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], min=1e-8)) / 2
    z = torch.sqrt(torch.clamp(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], min=1e-8)) / 2
    x = torch.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = torch.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = torch.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    return torch.stack([w, x, y, z], -1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def knn_scale(P, k=4):
    """initial Gaussian scale: mean distance to the k nearest neighbours (scipy, CPU)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P, k=k + 1)
    return np.clip(d[:, 1:].mean(1), 0.01, 3.0)


def init_group(P, rgb, sh_degree=3, opacity=0.1, device="cuda"):
    """Gaussian parameters from points (N, 3) and colours (N, 3) in [0, 1]."""
    N = len(P)
    C0 = 0.28209479177387814
    sh0 = ((torch.tensor(rgb, dtype=torch.float32) - 0.5) / C0)[:, None, :]
    shN = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3)
    s = torch.log(torch.tensor(knn_scale(P), dtype=torch.float32))[:, None].repeat(1, 3)
    q = torch.zeros(N, 4); q[:, 0] = 1.0
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(P, dtype=torch.float32)),
        "scales": torch.nn.Parameter(s), "quats": torch.nn.Parameter(q),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), opacity))),
        "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN)}).to(device)


class Sky(torch.nn.Module):
    """equirectangular RGB map indexed by the world ray direction (z up)."""

    def __init__(self, h=256, w=512):
        super().__init__()
        self.tex = torch.nn.Parameter(torch.zeros(1, 3, h, w))

    def forward(self, dirs):
        d = F.normalize(dirs, dim=-1)
        u = torch.atan2(d[..., 1], d[..., 0]) / math.pi                       # [-1, 1]
        v = 1.0 - 2.0 * torch.acos(torch.clamp(d[..., 2], -1, 1)) / math.pi   # +1 up, -1 down
        grid = torch.stack([u, -v], -1).reshape(1, -1, 1, 2)
        out = F.grid_sample(self.tex, grid, align_corners=False, padding_mode="border")
        return torch.sigmoid(out[0, :, :, 0].T.reshape(*dirs.shape[:-1], 3))


class BoxDelta(torch.nn.Module):
    """learnable per-(actor, tick) box correction: translation (m) and yaw (rad) about the box's up axis, applied
    in the world frame on top of the given track (used to recover boxes from noisy tracks)."""

    def __init__(self, keys):
        super().__init__()
        self.index = {k: i for i, k in enumerate(keys)}
        self.d = torch.nn.Parameter(torch.zeros(len(keys), 4))

    def apply(self, name, tick, T):
        d = self.d[self.index[(name, tick)]]
        c, s = torch.cos(d[3]), torch.sin(d[3])
        Rz = torch.stack([torch.stack([c, -s, torch.zeros_like(c)]), torch.stack([s, c, torch.zeros_like(c)]),
                          torch.tensor([0.0, 0.0, 1.0], device=T.device)])
        top = torch.cat([Rz @ T[:3, :3], (T[:3, 3] + d[:3])[:, None]], 1)
        return torch.cat([top, T[3:]], 0)


class Scene(torch.nn.Module):
    def __init__(self, groups, tracks, sh_degree=3):
        """groups: {"bg" or actor id: ParameterDict}; tracks: {actor id: {tick: 4x4 box -> local-world}}."""
        super().__init__()
        self.groups = torch.nn.ModuleDict({str(k): v for k, v in groups.items()})
        self.sky = Sky()
        self.sh_degree = sh_degree
        self.tracks = {str(a): {int(k): torch.tensor(np.array(T), dtype=torch.float32) for k, T in tr.items()}
                       for a, tr in tracks.items()}
        self.box_delta = None                                                 # optional pose refinement

    def gather(self, tick, actors=None):
        """all Gaussians of one frame in the (local) world frame; also the owner index of each Gaussian."""
        parts, owner = [], []
        for name, g in self.groups.items():
            if name == "bg":
                m, q = g["means"], g["quats"]
            else:
                if actors is not None and name not in actors:
                    continue
                T = self.tracks[name].get(tick)
                if T is None:
                    continue
                T = T.to(g["means"].device)
                if self.box_delta is not None and (name, tick) in self.box_delta.index:
                    T = self.box_delta.apply(name, tick, T)
                m = g["means"] @ T[:3, :3].T + T[:3, 3]
                q = quat_mul(quat_from_mat(T[:3, :3])[None].expand_as(g["quats"]), F.normalize(g["quats"], dim=-1))
            parts.append((m, q, g["scales"], g["opacities"], torch.cat([g["sh0"], g["shN"]], 1)))
            owner.append((name, len(m)))
        cat = [torch.cat([p[i] for p in parts]) for i in range(5)]
        return cat, owner

    def render(self, T_world_cam, K, W, H, tick, sh_degree=None, actors=None, with_sky=True, return_owner_alpha=None):
        """RGB (H, W, 3), expected depth (H, W) along the camera z axis, alpha (H, W)."""
        (means, quats, scales, opac, sh), owner = self.gather(tick, actors)
        dev = means.device
        viewmat = torch.linalg.inv(T_world_cam.to(dev))[None]
        colors, alphas, info = rasterization(
            means, quats, torch.exp(scales), torch.sigmoid(opac), sh, viewmat, K.to(dev)[None], W, H,
            sh_degree=self.sh_degree if sh_degree is None else sh_degree, render_mode="RGB+ED", packed=False,
            with_ut=True, with_eval3d=True, near_plane=0.2, far_plane=2000.0)
        rgb, depth, alpha = colors[0, ..., :3], colors[0, ..., 3], alphas[0, ..., 0]
        if with_sky:
            u, v = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
            d = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
            dirs = d @ T_world_cam[:3, :3].to(dev).T
            rgb = rgb + (1 - alpha[..., None]) * self.sky(dirs)
        return rgb, depth, alpha, info, owner
