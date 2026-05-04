# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:19:16 2026

@author: user
"""

from pathlib import Path
import numpy as np


def load_humaneva_calibration(cal_path):
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
            "Expected 22 calibration values in {}, got {}".format(cal_path, len(values))
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
    }


def world_to_camera(points_3d, R, t):
    """
    points_3d: (N, 3)
    R: (3, 3)
    t: (3, 1)
    returns: (N, 3)
    """
    points_3d = np.asarray(points_3d, dtype=np.float32)
    cam = (R @ points_3d.T) + t
    return cam.T


def distort_points_normalized(xy, dist):
    """
    xy: (N, 2) normalized image coordinates
    dist: [k1, k2, p1, p2, k3]
    """
    k1, k2, p1, p2, k3 = dist
    x = xy[:, 0]
    y = xy[:, 1]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x * radial + x_tan
    yd = y * radial + y_tan

    return np.stack([xd, yd], axis=1)


def camera_to_image(points_cam, K, dist=None):
    """
    points_cam: (N, 3)
    returns: (N, 2)
    """
    z = points_cam[:, 2:3]
    eps = 1e-8
    xy = points_cam[:, :2] / np.maximum(z, eps)

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

    return np.stack([u, v], axis=1)


def project_world_to_image(points_3d, K, R, t, dist=None):
    points_cam = world_to_camera(points_3d, R, t)
    points_2d = camera_to_image(points_cam, K, dist=dist)
    return points_2d, points_cam