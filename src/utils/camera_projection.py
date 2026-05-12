# -*- coding: utf-8 -*-
"""
Camera projection utilities for HumanEva-I.

This module supports:
    - loading HumanEva .cal files
    - projecting 3D world/body-pose points to image coordinates
    - testing multiple camera extrinsic conventions

Important:
    The default extrinsic convention is kept as:

        X_cam = R @ X_world + t

    because this is what the current projected dataset was exported with.
    Use the `extrinsic_mode` argument to test alternatives before changing
    the default permanently.
"""

from pathlib import Path
import numpy as np


# ============================================================
# Supported extrinsic conventions
# ============================================================

EXTRINSIC_R_X_PLUS_T = "R_X_plus_t"
EXTRINSIC_R_X_MINUS_T = "R_X_minus_t"
EXTRINSIC_R_X_MINUS_C = "R_X_minus_C"
EXTRINSIC_RT_X_MINUS_C = "RT_X_minus_C"
EXTRINSIC_RT_X_PLUS_T = "RT_X_plus_t"
EXTRINSIC_RT_X_MINUS_T = "RT_X_minus_t"

DEFAULT_EXTRINSIC_MODE = EXTRINSIC_R_X_PLUS_T

SUPPORTED_EXTRINSIC_MODES = [
    EXTRINSIC_R_X_PLUS_T,
    EXTRINSIC_R_X_MINUS_T,
    EXTRINSIC_R_X_MINUS_C,
    EXTRINSIC_RT_X_MINUS_C,
    EXTRINSIC_RT_X_PLUS_T,
    EXTRINSIC_RT_X_MINUS_T,
]


# ============================================================
# Calibration loading
# ============================================================

def load_humaneva_calibration(cal_path):
    """
    Load a HumanEva .cal file.

    Expected 22-line format:

        0:  fx
        1:  fy
        2:  cx
        3:  cy
        4:  skew
        5:  k1
        6:  k2
        7:  p1
        8:  p2
        9:  k3
        10-18: R, row-major 3x3
        19-21: t

    Parameters
    ----------
    cal_path : str or Path
        Path to HumanEva calibration file.

    Returns
    -------
    dict
        Dictionary containing K, R, t, distortion coefficients, and scalar
        intrinsics.
    """
    cal_path = Path(cal_path)
    values = []

    with open(cal_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(float(line))

    if len(values) != 22:
        raise ValueError(
            "Expected 22 calibration values in {}, got {}".format(
                cal_path,
                len(values),
            )
        )

    fx, fy, cx, cy = values[0], values[1], values[2], values[3]
    skew = values[4]

    k1, k2, p1, p2, k3 = values[5:10]

    R = np.array(values[10:19], dtype=np.float32).reshape(3, 3)
    t = np.array(values[19:22], dtype=np.float32).reshape(3, 1)

    K = np.array(
        [
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    return {
        "K": K,
        "R": R,
        "t": t,
        "dist": dist,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "skew": skew,
        "cal_path": str(cal_path),
    }


# ============================================================
# Shape helpers
# ============================================================

def _as_points_array(points_3d):
    """
    Convert input points to a float32 array of shape (..., 3).

    Returns
    -------
    points : np.ndarray
        Input points as float32.
    original_shape : tuple
        Original shape.
    points_flat : np.ndarray
        Flattened points with shape (N, 3).
    """
    points = np.asarray(points_3d, dtype=np.float32)

    if points.shape[-1] != 3:
        raise ValueError(
            "Expected points_3d to have last dimension 3, got shape {}".format(
                points.shape,
            )
        )

    original_shape = points.shape
    points_flat = points.reshape(-1, 3)

    return points, original_shape, points_flat


def _reshape_points(points_flat, original_shape):
    return points_flat.reshape(original_shape)


def _prepare_R_t(R, t):
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).reshape(1, 3)

    if R.shape != (3, 3):
        raise ValueError("Expected R shape (3, 3), got {}".format(R.shape))

    return R, t


# ============================================================
# World-to-camera transforms
# ============================================================

def world_to_camera(points_3d, R, t, extrinsic_mode=DEFAULT_EXTRINSIC_MODE):
    """
    Convert 3D world coordinates to camera coordinates.

    Parameters
    ----------
    points_3d : np.ndarray
        3D points with shape (N, 3), (T, N, 3), or generally (..., 3).
    R : np.ndarray
        Rotation matrix, shape (3, 3).
    t : np.ndarray
        Translation vector, shape (3,), (3, 1), or (1, 3).
    extrinsic_mode : str
        Camera extrinsic convention.

    Supported modes
    ---------------
    "R_X_plus_t":
        X_cam = R @ X_world + t

    "R_X_minus_t":
        X_cam = R @ X_world - t

    "R_X_minus_C":
        X_cam = R @ (X_world - t)

    "RT_X_minus_C":
        X_cam = R.T @ (X_world - t)

    "RT_X_plus_t":
        X_cam = R.T @ X_world + t

    "RT_X_minus_t":
        X_cam = R.T @ X_world - t

    Returns
    -------
    np.ndarray
        Camera-space points with the same shape as input.
    """
    _, original_shape, points_flat = _as_points_array(points_3d)
    R, t = _prepare_R_t(R, t)

    if extrinsic_mode == EXTRINSIC_R_X_PLUS_T:
        points_cam = (R @ points_flat.T).T + t

    elif extrinsic_mode == EXTRINSIC_R_X_MINUS_T:
        points_cam = (R @ points_flat.T).T - t

    elif extrinsic_mode == EXTRINSIC_R_X_MINUS_C:
        points_cam = (R @ (points_flat - t).T).T

    elif extrinsic_mode == EXTRINSIC_RT_X_MINUS_C:
        points_cam = (R.T @ (points_flat - t).T).T

    elif extrinsic_mode == EXTRINSIC_RT_X_PLUS_T:
        points_cam = (R.T @ points_flat.T).T + t

    elif extrinsic_mode == EXTRINSIC_RT_X_MINUS_T:
        points_cam = (R.T @ points_flat.T).T - t

    else:
        raise ValueError(
            "Unknown extrinsic_mode '{}'. Supported modes are: {}".format(
                extrinsic_mode,
                SUPPORTED_EXTRINSIC_MODES,
            )
        )

    return _reshape_points(points_cam.astype(np.float32), original_shape)


# ============================================================
# Image projection
# ============================================================

def distort_points_normalized(xy, dist):
    """
    Apply radial/tangential distortion to normalized image coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Normalized image coordinates, shape (..., 2).
    dist : np.ndarray
        Distortion coefficients [k1, k2, p1, p2, k3].

    Returns
    -------
    np.ndarray
        Distorted normalized coordinates, same shape as xy.
    """
    xy = np.asarray(xy, dtype=np.float32)
    original_shape = xy.shape

    if xy.shape[-1] != 2:
        raise ValueError("Expected xy last dimension 2, got {}".format(xy.shape))

    xy_flat = xy.reshape(-1, 2)
    dist = np.asarray(dist, dtype=np.float32).reshape(-1)

    if dist.size != 5:
        raise ValueError("Expected 5 distortion coefficients, got {}".format(dist.size))

    k1, k2, p1, p2, k3 = dist

    x = xy_flat[:, 0]
    y = xy_flat[:, 1]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x * radial + x_tan
    yd = y * radial + y_tan

    out = np.stack([xd, yd], axis=1)

    return out.reshape(original_shape).astype(np.float32)


def camera_to_image(points_cam, K, dist=None, eps=1e-8):
    """
    Project camera-space 3D points to image coordinates.

    Parameters
    ----------
    points_cam : np.ndarray
        Camera coordinates, shape (N, 3), (T, N, 3), or (..., 3).
    K : np.ndarray
        Intrinsic matrix, shape (3, 3).
    dist : np.ndarray or None
        Optional distortion coefficients [k1, k2, p1, p2, k3].
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Image coordinates with shape (..., 2).
    """
    points_cam = np.asarray(points_cam, dtype=np.float32)

    if points_cam.shape[-1] != 3:
        raise ValueError(
            "Expected points_cam last dimension 3, got {}".format(points_cam.shape)
        )

    original_shape = points_cam.shape[:-1]
    cam_flat = points_cam.reshape(-1, 3)

    K = np.asarray(K, dtype=np.float32)

    if K.shape != (3, 3):
        raise ValueError("Expected K shape (3, 3), got {}".format(K.shape))

    z = cam_flat[:, 2:3]

    # Preserve sign but avoid numerical division problems.
    z_safe = np.where(np.abs(z) < eps, eps * np.sign(z + eps), z)

    xy = cam_flat[:, :2] / z_safe

    if dist is not None:
        xy = distort_points_normalized(xy, dist)

    fx = K[0, 0]
    skew = K[0, 1]
    cx = K[0, 2]
    fy = K[1, 1]
    cy = K[1, 2]

    x = xy[:, 0]
    y = xy[:, 1]

    u = fx * x + skew * y + cx
    v = fy * y + cy

    points_2d = np.stack([u, v], axis=1)

    return points_2d.reshape(*original_shape, 2).astype(np.float32)


def project_world_to_image(
    points_3d,
    K,
    R,
    t,
    dist=None,
    extrinsic_mode=DEFAULT_EXTRINSIC_MODE,
):
    """
    Project 3D world points to 2D image coordinates.

    Parameters
    ----------
    points_3d : np.ndarray
        3D world points, shape (..., 3).
    K : np.ndarray
        Camera intrinsics, shape (3, 3).
    R : np.ndarray
        Rotation matrix, shape (3, 3).
    t : np.ndarray
        Translation vector.
    dist : np.ndarray or None
        Optional distortion coefficients.
    extrinsic_mode : str
        Extrinsic convention to use.

    Returns
    -------
    points_2d : np.ndarray
        Projected 2D points, shape (..., 2).
    points_cam : np.ndarray
        Camera-space 3D points, shape (..., 3).
    """
    points_cam = world_to_camera(
        points_3d,
        R=R,
        t=t,
        extrinsic_mode=extrinsic_mode,
    )

    points_2d = camera_to_image(
        points_cam,
        K=K,
        dist=dist,
    )

    return points_2d, points_cam


# ============================================================
# Debug helpers
# ============================================================

def project_world_to_image_all_extrinsics(points_3d, K, R, t, dist=None):
    """
    Project points using all supported extrinsic conventions.

    Useful for debugging HumanEva calibration alignment.

    Returns
    -------
    dict
        {
            extrinsic_mode: {
                "points_2d": np.ndarray,
                "points_cam": np.ndarray,
                "depth_min": float,
                "depth_max": float,
                "x_min": float,
                "x_max": float,
                "y_min": float,
                "y_max": float,
            },
            ...
        }
    """
    results = {}

    for mode in SUPPORTED_EXTRINSIC_MODES:
        points_2d, points_cam = project_world_to_image(
            points_3d,
            K=K,
            R=R,
            t=t,
            dist=dist,
            extrinsic_mode=mode,
        )

        results[mode] = {
            "points_2d": points_2d,
            "points_cam": points_cam,
            "depth_min": float(np.min(points_cam[..., 2])),
            "depth_max": float(np.max(points_cam[..., 2])),
            "x_min": float(np.min(points_2d[..., 0])),
            "x_max": float(np.max(points_2d[..., 0])),
            "y_min": float(np.min(points_2d[..., 1])),
            "y_max": float(np.max(points_2d[..., 1])),
        }

    return results


def print_projection_summary(points_2d, points_cam, prefix=""):
    """
    Print a compact projection diagnostic summary.
    """
    points_2d = np.asarray(points_2d)
    points_cam = np.asarray(points_cam)

    msg = (
        "{}2D x {:.2f} to {:.2f} | "
        "2D y {:.2f} to {:.2f} | "
        "depth {:.2f} to {:.2f}"
    ).format(
        prefix,
        float(np.min(points_2d[..., 0])),
        float(np.max(points_2d[..., 0])),
        float(np.min(points_2d[..., 1])),
        float(np.max(points_2d[..., 1])),
        float(np.min(points_cam[..., 2])),
        float(np.max(points_cam[..., 2])),
    )

    print(msg)