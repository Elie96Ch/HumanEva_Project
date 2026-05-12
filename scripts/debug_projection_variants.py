from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from src.utils.camera_projection import (
    load_humaneva_calibration,
    project_world_to_image_all_extrinsics,
)


ROOT_INDEX = 1


def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_pose(ax, pts, edges, color="yellow", label=None):
    pts = np.asarray(pts)

    ax.scatter(pts[:, 0], pts[:, 1], s=20, c=color, label=label)

    for i, j in edges:
        if np.all(np.isfinite(pts[[i, j]])):
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                color=color,
                linewidth=2,
            )


def main():
    raw_root = PROJECT_ROOT / "data" / "humaneva" / "raw"
    pose_root = PROJECT_ROOT / "data" / "humaneva" / "processed" / "body_pose_matlab"

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "debug_projection_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject = "S1"
    action = "Walking_1"
    camera = "C1"

    # Use same first valid frame from your test.
    mocap_frame_idx = 704

    # Current approximate mapping.
    # If needed, change this, but first keep it consistent with your visualization.
    video_frame_idx = int(round(mocap_frame_idx / 2.0))

    pose_path = pose_root / subject / f"{action}.mat"
    cal_path = raw_root / subject / "Calibration_Data" / f"{camera}.cal"
    video_path = raw_root / subject / "Image_Data" / f"{action}_({camera}).avi"

    pose_data = scipy.io.loadmat(pose_path, squeeze_me=True, struct_as_record=False)
    pose_3d = np.asarray(pose_data["pose_3d"], dtype=np.float32)
    valid = np.asarray(pose_data["valid"]).astype(np.uint8)

    if valid[mocap_frame_idx] != 1:
        print("[WARN] selected mocap frame is not valid:", mocap_frame_idx)

    pts3d = pose_3d[mocap_frame_idx]

    cam = load_humaneva_calibration(cal_path)
    frame = read_video_frame(video_path, video_frame_idx)

    h, w = frame.shape[:2]

    edges = [
        (1, 0),
        (0, 18),
        (18, 19),

        (0, 2),
        (2, 3),
        (3, 4),
        (4, 5),

        (0, 6),
        (6, 7),
        (7, 8),
        (8, 9),

        (1, 10),
        (10, 11),
        (11, 12),
        (12, 13),

        (1, 14),
        (14, 15),
        (15, 16),
        (16, 17),
    ]

    results = project_world_to_image_all_extrinsics(
        pts3d,
        K=cam["K"],
        R=cam["R"],
        t=cam["t"],
        dist=None,
    )

    print("subject:", subject)
    print("action:", action)
    print("camera:", camera)
    print("mocap frame:", mocap_frame_idx)
    print("video frame:", video_frame_idx)
    print("image size:", w, h)
    print("video path:", video_path)
    print("cal path:", cal_path)
    print()

    for mode, result in results.items():
        pts2d = result["points_2d"]
        ptscam = result["points_cam"]

        print(
            mode,
            "| x {:.1f}-{:.1f}".format(float(np.min(pts2d[:, 0])), float(np.max(pts2d[:, 0]))),
            "| y {:.1f}-{:.1f}".format(float(np.min(pts2d[:, 1])), float(np.max(pts2d[:, 1]))),
            "| z {:.1f}-{:.1f}".format(float(np.min(ptscam[:, 2])), float(np.max(ptscam[:, 2]))),
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)
        draw_pose(ax, pts2d, edges, color="yellow", label=mode)

        ax.set_title(
            f"{mode} | {subject} {action} {camera} | mocap {mocap_frame_idx} video {video_frame_idx}",
            fontsize=10,
        )
        ax.axis("off")
        ax.legend(loc="upper right")

        save_path = out_dir / f"{subject}_{action}_{camera}_m{mocap_frame_idx:06d}_v{video_frame_idx:06d}_{mode}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print("saved:", save_path)

    print()
    print("Now inspect the saved images and identify which mode, if any, lands on the person.")
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()