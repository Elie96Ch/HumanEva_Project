from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.mlp import PoseMLP
from src.utils.camera_projection import load_humaneva_calibration, project_world_to_image


ROOT_INDEX = 1  # torsoDistal


def normalize_2d_points(points_2d, root_index=ROOT_INDEX):
    root = points_2d[root_index:root_index + 1]
    centered = points_2d - root
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return centered / scale


def compute_3d_scale(points_3d):
    scale = np.max(np.linalg.norm(points_3d, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return scale


def root_center_pose(pose_3d, root_index=ROOT_INDEX):
    root = pose_3d[root_index:root_index + 1]
    return pose_3d - root


def draw_pose(ax, pts, color, edges, label=None, linewidth=2, markersize=18):
    pts = np.asarray(pts)

    finite = np.all(np.isfinite(pts), axis=1)
    if np.any(finite):
        ax.scatter(
            pts[finite, 0],
            pts[finite, 1],
            s=markersize,
            c=color,
            label=label,
        )

    for i, j in edges:
        if np.all(np.isfinite(pts[[i, j]])):
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                color=color,
                linewidth=linewidth,
            )


def load_model(checkpoint_path, device):
    model = PoseMLP(
        input_dim=40,
        output_dim=60,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def read_ofs(ofs_path):
    """
    Read HumanEva .ofs synchronization file.

    Official format:
        im_st
        mc_st
        mc_sc

    Official mapping:
        MocapIndex = mc_st + (ImageIndex - im_st) * mc_sc
        ImageIndex = im_st + (MocapIndex - mc_st) / mc_sc

    These indices are MATLAB-style 1-based indices.
    """
    ofs_path = Path(ofs_path)

    values = []
    with open(ofs_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.extend([float(x) for x in line.split()])

    if len(values) != 3:
        raise ValueError(
            "Expected 3 values in {}, got {}: {}".format(
                ofs_path,
                len(values),
                values,
            )
        )

    im_st, mc_st, mc_sc = values
    return im_st, mc_st, mc_sc


def mocap_frame_to_video_frame_python(mocap_frame_idx_python, im_st, mc_st, mc_sc):
    """
    Convert Python zero-based mocap frame index to Python zero-based video frame index.

    Python mocap frame 0 == MATLAB mocap frame 1.
    Python video frame 0 == MATLAB image frame 1.
    """
    mocap_idx_matlab = mocap_frame_idx_python + 1

    image_idx_matlab = im_st + (mocap_idx_matlab - mc_st) / mc_sc

    video_idx_python = int(round(image_idx_matlab)) - 1

    return video_idx_python, image_idx_matlab


def video_frame_to_mocap_frame_python(video_frame_idx_python, im_st, mc_st, mc_sc):
    """
    Convert Python zero-based video frame index to Python zero-based mocap frame index.

    Useful for debugging.
    """
    image_idx_matlab = video_frame_idx_python + 1

    mocap_idx_matlab = mc_st + (image_idx_matlab - im_st) * mc_sc

    mocap_idx_python = int(round(mocap_idx_matlab)) - 1

    return mocap_idx_python, mocap_idx_matlab


def sample_items(projected_root, raw_root, subject="S3", camera="C1", n=10, seed=42):
    """
    Sample only frames that can be mapped to valid video frames using .ofs.

    This avoids sampling early mocap frames that map to negative video frames.
    """
    random.seed(seed)

    subject_dir = Path(projected_root) / subject / camera
    npz_files = sorted(subject_dir.glob("*.npz"))

    items = []

    for npz_path in npz_files:
        action = npz_path.stem

        ofs_path = Path(raw_root) / subject / "Sync_Data" / "{}_({}).ofs".format(action, camera)
        video_path = Path(raw_root) / subject / "Image_Data" / "{}_({}).avi".format(action, camera)

        if not ofs_path.exists():
            print("[SKIP ACTION] Missing OFS:", ofs_path)
            continue

        if not video_path.exists():
            print("[SKIP ACTION] Missing video:", video_path)
            continue

        im_st, mc_st, mc_sc = read_ofs(ofs_path)
        num_video_frames = get_video_num_frames(video_path)

        if num_video_frames <= 0:
            print("[SKIP ACTION] Could not read video frame count:", video_path)
            continue

        data = np.load(npz_path, allow_pickle=True)
        valid = np.asarray(data["valid"]).astype(np.uint8)

        valid_idx = np.where(valid == 1)[0]
        if len(valid_idx) == 0:
            continue

        valid_and_visible = []

        for frame_idx in valid_idx.tolist():
            video_frame_idx, image_idx_matlab = mocap_frame_to_video_frame_python(
                frame_idx,
                im_st,
                mc_st,
                mc_sc,
            )

            if 0 <= video_frame_idx < num_video_frames:
                valid_and_visible.append(frame_idx)

        if len(valid_and_visible) == 0:
            print(
                "[SKIP ACTION] No valid mocap frames map into video for {} {} {}".format(
                    subject,
                    action,
                    camera,
                )
            )
            continue

        k = min(3, len(valid_and_visible))
        chosen = random.sample(valid_and_visible, k=k)

        for frame_idx in chosen:
            items.append((npz_path, frame_idx))

    if len(items) < n:
        return items

    return random.sample(items, n)


def get_video_num_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return num_frames


def frame_exists(video_path, frame_idx):
    num_frames = get_video_num_frames(video_path)

    if num_frames <= 0:
        return False, num_frames

    return 0 <= frame_idx < num_frames, num_frames


def read_video_frame(video_path, frame_idx):
    ok_exists, num_frames = frame_exists(video_path, frame_idx)

    if not ok_exists:
        raise RuntimeError(
            "Requested frame {} is out of range for {} total frames in {}".format(
                frame_idx,
                num_frames,
                video_path,
            )
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Could not read frame {} from {}".format(frame_idx, video_path))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def save_frame_window_debug(
    video_path,
    center_video_frame_idx,
    pts2d_gt,
    gt_2d_reproj,
    pred_2d,
    edges,
    out_dir,
    action,
    mocap_frame_idx,
    window=10,
):
    """
    Save nearby video frames around the OFS-mapped frame.

    If OFS sync is correct, the center frame should be the best one.
    """
    debug_dir = out_dir / "debug_ofs_frame_window" / "{}_m{:06d}".format(
        action,
        mocap_frame_idx,
    )
    debug_dir.mkdir(parents=True, exist_ok=True)

    num_frames = get_video_num_frames(video_path)
    saved = 0

    for delta in range(-window, window + 1):
        video_frame_idx = center_video_frame_idx + delta

        if video_frame_idx < 0 or video_frame_idx >= num_frames:
            continue

        try:
            frame = read_video_frame(video_path, video_frame_idx)
        except Exception:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)

        draw_pose(
            ax,
            pts2d_gt,
            color="lime",
            edges=edges,
            label="Stored GT 2D",
            linewidth=2,
            markersize=16,
        )

        draw_pose(
            ax,
            gt_2d_reproj,
            color="cyan",
            edges=edges,
            label="GT 3D reproj",
            linewidth=1,
            markersize=10,
        )

        draw_pose(
            ax,
            pred_2d,
            color="red",
            edges=edges,
            label="Pred reproj",
            linewidth=2,
            markersize=14,
        )

        ax.set_title(
            "{} | mocap {} | video {} | delta {:+d}".format(
                action,
                mocap_frame_idx,
                video_frame_idx,
                delta,
            ),
            fontsize=11,
        )

        ax.axis("off")
        ax.legend(loc="upper right")

        save_path = debug_dir / "v{:06d}_delta_{:+03d}.png".format(
            video_frame_idx,
            delta,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved += 1

    print("[DEBUG WINDOW] Saved {} images to {}".format(saved, debug_dir))


def main():
    projected_root = PROJECT_ROOT / "data" / "humaneva" / "processed" / "projected_bodypose_2d"
    raw_root = PROJECT_ROOT / "data" / "humaneva" / "raw"
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "mlp_projected_bodypose_best.pt"

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "qualitative_projected_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject = "S3"
    camera = "C1"
    n_samples = 10

    save_debug_windows = True
    debug_window = 10

    show_figures = True

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)

    items = sample_items(
        projected_root=projected_root,
        raw_root=raw_root,
        subject=subject,
        camera=camera,
        n=n_samples,
        seed=7,
    )

    if len(items) == 0:
        raise RuntimeError("No valid projected samples found that map into video frames.")

    cal_path = raw_root / subject / "Calibration_Data" / "{}.cal".format(camera)
    cam = load_humaneva_calibration(cal_path)

    shown_count = 0

    for sample_idx, (npz_path, frame_idx) in enumerate(items, start=1):
        data = np.load(npz_path, allow_pickle=True)

        pose_2d = np.asarray(data["pose_2d"], dtype=np.float32)
        pose_3d = np.asarray(data["pose_3d"], dtype=np.float32)

        action = npz_path.stem

        pts2d_gt = pose_2d[frame_idx]
        pts3d_world = pose_3d[frame_idx]

        # Reproject GT 3D using official calibration model.
        # This should match stored GT 2D if the dataset was re-exported with dist=cam["dist"].
        gt_2d_reproj, gt_depth = project_world_to_image(
            pts3d_world,
            K=cam["K"],
            R=cam["R"],
            t=cam["t"],
            dist=cam["dist"],
        )

        projection_diff = np.linalg.norm(pts2d_gt - gt_2d_reproj, axis=1)

        x_norm = normalize_2d_points(pts2d_gt, root_index=ROOT_INDEX)
        x_tensor = torch.tensor(x_norm.reshape(1, -1), dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_norm = model(x_tensor).cpu().numpy().reshape(20, 3)

        gt_centered = root_center_pose(pts3d_world, root_index=ROOT_INDEX)
        scale_3d = compute_3d_scale(gt_centered)
        root_world = pts3d_world[ROOT_INDEX:ROOT_INDEX + 1]

        pred_centered = pred_norm * scale_3d
        pred_world = pred_centered + root_world

        pred_2d, pred_depth = project_world_to_image(
            pred_world,
            K=cam["K"],
            R=cam["R"],
            t=cam["t"],
            dist=cam["dist"],
        )

        ofs_path = raw_root / subject / "Sync_Data" / "{}_({}).ofs".format(action, camera)
        video_path = raw_root / subject / "Image_Data" / "{}_({}).avi".format(action, camera)

        if not ofs_path.exists():
            print("[SKIP] Missing OFS:", ofs_path)
            continue

        if not video_path.exists():
            print("[SKIP] Missing video:", video_path)
            continue

        im_st, mc_st, mc_sc = read_ofs(ofs_path)

        video_frame_idx, image_idx_matlab = mocap_frame_to_video_frame_python(
            frame_idx,
            im_st,
            mc_st,
            mc_sc,
        )

        exists, num_frames = frame_exists(video_path, video_frame_idx)

        if not exists:
            print(
                "[SKIP] {} | mocap frame {} maps to video frame {} "
                "(MATLAB image {:.3f}), outside video range 0..{}".format(
                    action,
                    frame_idx,
                    video_frame_idx,
                    image_idx_matlab,
                    num_frames - 1,
                )
            )
            continue

        try:
            frame = read_video_frame(video_path, video_frame_idx)
        except Exception as e:
            print("[SKIP] {} | frame {} | {}".format(action, video_frame_idx, e))
            continue

        h, w = frame.shape[:2]

        print("\n" + "=" * 80)
        print("sample:", sample_idx)
        print("subject:", subject)
        print("action:", action)
        print("camera:", camera)
        print("npz:", npz_path)
        print("ofs:", ofs_path)
        print("video:", video_path)
        print("Python mocap frame:", frame_idx)
        print("MATLAB mocap frame:", frame_idx + 1)
        print("Mapped MATLAB image frame:", image_idx_matlab)
        print("Mapped Python video frame:", video_frame_idx)
        print("Video frames:", num_frames)
        print("Image size: width={} height={}".format(w, h))
        print("OFS im_st={} mc_st={} mc_sc={}".format(im_st, mc_st, mc_sc))
        print(
            "Stored GT vs GT reproj mean/max px: {:.6f} / {:.6f}".format(
                float(np.nanmean(projection_diff)),
                float(np.nanmax(projection_diff)),
            )
        )
        print(
            "GT 2D x {:.2f}-{:.2f} | y {:.2f}-{:.2f}".format(
                float(np.nanmin(pts2d_gt[:, 0])),
                float(np.nanmax(pts2d_gt[:, 0])),
                float(np.nanmin(pts2d_gt[:, 1])),
                float(np.nanmax(pts2d_gt[:, 1])),
            )
        )
        print(
            "GT depth min/max {:.2f} / {:.2f}".format(
                float(np.nanmin(gt_depth[:, 2] if gt_depth.ndim == 2 else gt_depth)),
                float(np.nanmax(gt_depth[:, 2] if gt_depth.ndim == 2 else gt_depth)),
            )
        )
        print("=" * 80)

        if save_debug_windows:
            save_frame_window_debug(
                video_path=video_path,
                center_video_frame_idx=video_frame_idx,
                pts2d_gt=pts2d_gt,
                gt_2d_reproj=gt_2d_reproj,
                pred_2d=pred_2d,
                edges=edges,
                out_dir=out_dir,
                action=action,
                mocap_frame_idx=frame_idx,
                window=debug_window,
            )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)

        draw_pose(
            ax,
            pts2d_gt,
            color="lime",
            edges=edges,
            label="Stored GT 2D",
            linewidth=2,
            markersize=18,
        )

        draw_pose(
            ax,
            gt_2d_reproj,
            color="cyan",
            edges=edges,
            label="GT 3D reproj",
            linewidth=1,
            markersize=10,
        )

        draw_pose(
            ax,
            pred_2d,
            color="red",
            edges=edges,
            label="Pred reproj",
            linewidth=2,
            markersize=16,
        )

        ax.set_title(
            "{} | mocap py {} / matlab {} | video py {} / matlab {:.2f}".format(
                action,
                frame_idx,
                frame_idx + 1,
                video_frame_idx,
                image_idx_matlab,
            ),
            fontsize=10,
        )

        ax.axis("off")
        ax.legend(loc="upper right")

        save_path = out_dir / "{:02d}_{}_m{:06d}_v{:06d}_ofs.png".format(
            sample_idx,
            action,
            frame_idx,
            video_frame_idx,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

        if show_figures:
            plt.show()

        plt.close(fig)

        shown_count += 1
        print("Saved:", save_path)

    print("\nDone.")
    print("Displayed/saved {} images.".format(shown_count))
    print("Output folder:", out_dir)
    print()
    print("Important:")
    print("This script uses .ofs synchronization:")
    print("  image = im_st + (mocap - mc_st) / mc_sc")
    print("If the overlay still does not align, inspect the debug window folder:")
    print(out_dir / "debug_ofs_frame_window")


if __name__ == "__main__":
    main()