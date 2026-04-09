import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import struct
import time
import glob
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


# ============================================================================
# Model loading
# ============================================================================

def load_da3_model(model_name="depth-anything/DA3NESTED-GIANT-LARGE"):
    """Initialize Depth-Anything-3 model on available device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    return model, device


# ============================================================================
# Image loading
# ============================================================================

def load_images_from_folder(data_path, extensions=['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']):
    """Scan folder and load all images with supported extensions"""
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(glob.glob(os.path.join(data_path, ext))))
    print(f"Found {len(image_files)} images in {data_path}")
    return image_files


# ============================================================================
# Inference
# ============================================================================

def run_da3_inference(model, image_files, process_res_method="upper_bound_resize"):
    """Run Depth-Anything-3 to get depth maps, camera poses, and intrinsics"""
    prediction = model.inference(
        image=image_files,
        infer_gs=False,
        process_res_method=process_res_method
    )
    print(f"Depth maps shape:  {prediction.depth.shape}")
    print(f"Extrinsics shape:  {prediction.extrinsics.shape}")
    print(f"Intrinsics shape:  {prediction.intrinsics.shape}")
    print(f"Confidence shape:  {prediction.conf.shape}")
    return prediction


# ============================================================================
# Point-cloud helpers
# ============================================================================

def depth_to_point_cloud(depth_map, rgb_image, intrinsics, extrinsics, conf_map=None, conf_thresh=0.5):
    """Back-project depth map to 3D points using camera parameters"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u, v, depth_map, rgb_image = u[valid_mask], v[valid_mask], depth_map[valid_mask], rgb_image[valid_mask]
    else:
        u, v, depth_map = u.flatten(), v.flatten(), depth_map.flatten()
        rgb_image = rgb_image.reshape(-1, 3)
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    points_cam = np.stack([x, y, z], axis=-1)
    # extrinsics is world-to-camera: p_cam = R @ p_world + t
    # Inverse: p_world = R^T @ (p_cam - t)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R
    colors = rgb_image.astype(np.float32) / 255.0
    return points_world, colors



def merge_point_clouds(prediction, conf_percentile=5.0, frame_step=10):
    """Combine sampled frames into a single point cloud using a percentile-based confidence filter.

    Only every ``frame_step``-th frame is back-projected (indices 0, frame_step,
    2*frame_step, …), reducing memory and compute while keeping good spatial coverage.
    The confidence percentile threshold is computed from the *sampled* frames only.

    Args:
        prediction:       Prediction object from run_da3_inference.
        conf_percentile:  Bottom percentile to discard (default: 90.0 → keep top 10%).
        frame_step:       Use one frame every this many frames (default: 10).
    """
    n_frames  = len(prediction.depth)
    frame_ids = list(range(0, n_frames, frame_step))
    print(f"Frame sampling: step={frame_step} → using {len(frame_ids)}/{n_frames} frames "
          f"(indices {frame_ids[0]}…{frame_ids[-1]})")

    # Compute a single global threshold from the confidence distribution of sampled frames
    all_conf = np.concatenate([prediction.conf[i].ravel() for i in frame_ids])
    conf_thresh = float(np.percentile(all_conf, conf_percentile))
    print(
        f"Confidence filter: bottom {conf_percentile:.1f}% discarded "
        f"→ threshold = {conf_thresh:.4f}  "
        f"(keeping top {100.0 - conf_percentile:.1f}% of pixels)"
    )

    all_points = []
    all_colors = []
    for i in frame_ids:
        points, colors = depth_to_point_cloud(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            conf_thresh
        )
        all_points.append(points)
        all_colors.append(colors)
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    print(f"Merged point cloud: {len(merged_points):,} points")
    return merged_points, merged_colors, all_points, all_colors


def clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=2.0):
    """Cleans a point cloud using SOR via Scipy cKDTree."""
    tree = cKDTree(points_3d)
    distances, _ = tree.query(points_3d, k=nb_neighbors + 1, workers=-1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    distance_threshold = global_mean + (std_ratio * global_std)
    mask = mean_distances < distance_threshold
    return points_3d[mask], colors_3d[mask]


# ============================================================================
# Save helpers
# ============================================================================

def save_point_cloud_ply(points, colors, scene_name, suffix=None, results_root=None):
    """Save point cloud as a CloudCompare-compatible PLY file.

    Writes to: <results_root>/<scene_name>/point_cloud/<scene_name>[_<suffix>].ply

    Args:
        points:       (N, 3) float numpy array of XYZ coordinates.
        colors:       (N, 3) float numpy array of RGB values in [0, 1].
        scene_name:   Name of the scene (used as sub-folder and file name).
        suffix:       Optional filename suffix, e.g. 'raw' or 'cleaned'.
        results_root: Absolute path to the RESULTS directory.
                      Defaults to <project_root>/RESULTS.
    """
    if results_root is None:
        results_root = os.path.join(_PROJECT_ROOT, 'RESULTS')
    out_dir = os.path.join(results_root, scene_name, 'point_cloud')
    os.makedirs(out_dir, exist_ok=True)
    filename = f'{scene_name}_{suffix}.ply' if suffix else f'{scene_name}.ply'
    out_path = os.path.join(out_dir, filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64), 0.0, 1.0))
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    print(f"Saved point cloud → {out_path}  ({len(points):,} points)")


def save_depth_maps(prediction, scene_name, results_root=None):
    """Save per-frame depth and confidence visualizations as PNG images.

    Writes to:
        <results_root>/<scene_name>/depth_map/rgb/<frame_idx:04d>.png
        <results_root>/<scene_name>/depth_map/depth/<frame_idx:04d>.png
        <results_root>/<scene_name>/depth_map/conf/<frame_idx:04d>.png

    Args:
        prediction:   Prediction object from run_da3_inference.
        scene_name:   Name of the scene (used as sub-folder).
        results_root: Absolute path to the RESULTS directory.
                      Defaults to <project_root>/RESULTS.
    """
    if results_root is None:
        results_root = os.path.join(_PROJECT_ROOT, 'RESULTS')
    base = os.path.join(results_root, scene_name, 'depth_map')
    rgb_dir   = os.path.join(base, 'rgb')
    depth_dir = os.path.join(base, 'depth')
    conf_dir  = os.path.join(base, 'conf')
    for d in (rgb_dir, depth_dir, conf_dir):
        os.makedirs(d, exist_ok=True)
    n_frames = len(prediction.depth)
    cm_conf = plt.get_cmap('viridis')
    for i in range(n_frames):
        fname = f'{i:04d}.png'
        plt.imsave(os.path.join(rgb_dir, fname), prediction.processed_images[i])
        depth_vis = visualize_depth(prediction.depth[i], cmap='Spectral')
        plt.imsave(os.path.join(depth_dir, fname), depth_vis)
        conf_map = prediction.conf[i]
        conf_norm = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min() + 1e-8)
        conf_vis = (cm_conf(conf_norm)[..., :3] * 255).astype(np.uint8)
        plt.imsave(os.path.join(conf_dir, fname), conf_vis)
    print(f"Saved {n_frames} depth-map frames → {base}")


def save_colmap_data(prediction, points, colors, scene_name, results_root=None):
    """Save a COLMAP-format sparse reconstruction consumable by gsplat.

    Always writes .txt files (universal fallback).  On platforms where
    pycolmap's binary reader is safe (struct.calcsize('L') == 8, i.e.
    Linux / macOS), .bin files are written alongside so pycolmap picks
    them up first.  On Windows (struct.calcsize('L') == 4) pycolmap's
    binary reader hard-codes f.read(8) for the count field which would
    cause a struct unpack error, so only .txt is written there and
    pycolmap falls back to .txt automatically.

    Directory layout:
        <results_root>/<scene_name>/colmap/
            images/           ← processed RGB frames (resolution matches intrinsics)
            sparse/0/
                cameras.txt   ← always written
                images.txt    ← always written
                points3D.txt  ← always written
                cameras.bin   ← Linux / macOS only
                images.bin    ← Linux / macOS only
                points3D.bin  ← Linux / macOS only

    DA3 extrinsics are already world-to-camera (native COLMAP convention),
    so no inversion is needed.  Each frame gets its own camera_id to
    accommodate per-frame intrinsics.

    Args:
        prediction:   Prediction object from run_da3_inference.
        points:       (N, 3) float array — cleaned 3-D point positions (world).
        colors:       (N, 3) float array — RGB values in [0, 1] for each point.
        scene_name:   Name of the scene (sub-folder key).
        results_root: Absolute path to the RESULTS directory.
                      Defaults to <project_root>/RESULTS.
    """
    if results_root is None:
        results_root = os.path.join(_PROJECT_ROOT, 'RESULTS')

    colmap_root = os.path.join(results_root, scene_name, 'colmap')
    images_dir  = os.path.join(colmap_root, 'images')
    sparse_dir  = os.path.join(colmap_root, 'sparse', '0')
    for d in (images_dir, sparse_dir):
        os.makedirs(d, exist_ok=True)

    n_frames = len(prediction.depth)
    H, W = prediction.processed_images[0].shape[:2]

    # pycolmap binary reader uses native 'L' (no prefix) and hard-codes
    # f.read(8) for the count field.  On Windows, sizeof(unsigned long)==4
    # so the unpack would fail.  Only write .bin where it is safe.
    _write_bin = (struct.calcsize('L') == 8)

    # ------------------------------------------------------------------
    # Prepare per-frame pose data (shared by both writers)
    # ------------------------------------------------------------------
    poses = []   # list of (qw, qx, qy, qz, tx, ty, tz) per frame
    for i in range(n_frames):
        w2c    = prediction.extrinsics[i]
        R_w2c  = w2c[:3, :3]
        t_w2c  = w2c[:3, 3]
        # scipy as_quat() → [qx, qy, qz, qw]; COLMAP wants [qw, qx, qy, qz]
        q_xyzw = Rotation.from_matrix(R_w2c).as_quat()
        poses.append((
            float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]),
            float(t_w2c[0]),  float(t_w2c[1]),  float(t_w2c[2]),
        ))

    pts     = points
    cols    = colors
    cols_u8 = (np.clip(cols, 0.0, 1.0) * 255).astype(np.uint8)

    # ==================================================================
    # TEXT writers (.txt) — always produced, safe on all platforms
    # ==================================================================

    # cameras.txt
    with open(os.path.join(sparse_dir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'# Number of cameras: {n_frames}\n')
        for i in range(n_frames):
            K      = prediction.intrinsics[i]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            f.write(f'{i + 1} PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n')

    # images.txt
    with open(os.path.join(sparse_dir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write(f'# Number of images: {n_frames}, mean observations per image: 0\n')
        for i, (qw, qx, qy, qz, tx, ty, tz) in enumerate(poses):
            f.write(
                f'{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} '
                f'{tx:.9f} {ty:.9f} {tz:.9f} {i + 1} {i:04d}.png\n'
            )
            f.write('\n')   # empty POINTS2D line (required by format)

    # points3D.txt
    with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write(f'# Number of points: {len(pts)}, mean track length: 0\n')
        for j in range(len(pts)):
            x, y, z = float(pts[j, 0]), float(pts[j, 1]), float(pts[j, 2])
            r, g, b = int(cols_u8[j, 0]), int(cols_u8[j, 1]), int(cols_u8[j, 2])
            f.write(f'{j + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n')

    # ==================================================================
    # BINARY writers (.bin) — only on Linux / macOS (L == 8 bytes)
    # ==================================================================
    if _write_bin:
        # cameras.bin
        # count: native 'L' (8 bytes here); per-cam: IiLL + 4d params
        with open(os.path.join(sparse_dir, 'cameras.bin'), 'wb') as f:
            f.write(struct.pack('L', n_frames))
            for i in range(n_frames):
                K      = prediction.intrinsics[i]
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])
                f.write(struct.pack('IiLL', i + 1, 1, W, H))   # 1 = PINHOLE
                f.write(struct.pack('4d', fx, fy, cx, cy))

        # images.bin
        # count: native 'L'; per-image: <I4d3dI + name\0 + Q(0)
        _img_struct = struct.Struct('<I4d3dI')
        with open(os.path.join(sparse_dir, 'images.bin'), 'wb') as f:
            f.write(struct.pack('L', n_frames))
            for i, (qw, qx, qy, qz, tx, ty, tz) in enumerate(poses):
                f.write(_img_struct.pack(i + 1, qw, qx, qy, qz, tx, ty, tz, i + 1))
                f.write(f'{i:04d}.png'.encode() + b'\x00')
                f.write(struct.pack('Q', 0))    # num_points2D = 0

        # points3D.bin
        # count: native 'L'; per-point: <Q3d3BdQ (track_len=0)
        _pt_struct = struct.Struct('<Q3d3BdQ')
        with open(os.path.join(sparse_dir, 'points3D.bin'), 'wb') as f:
            f.write(struct.pack('L', len(pts)))
            for j in range(len(pts)):
                x, y, z = float(pts[j, 0]), float(pts[j, 1]), float(pts[j, 2])
                r, g, b = int(cols_u8[j, 0]), int(cols_u8[j, 1]), int(cols_u8[j, 2])
                f.write(_pt_struct.pack(j + 1, x, y, z, r, g, b, 0.0, 0))

    # ------------------------------------------------------------------
    # images/ — processed RGB frames (resolution matches stored intrinsics)
    # ------------------------------------------------------------------
    for i in range(n_frames):
        plt.imsave(os.path.join(images_dir, f'{i:04d}.png'), prediction.processed_images[i])

    bin_note = " + .bin" if _write_bin else " (.bin skipped on Windows)"
    print(f"Saved COLMAP data → {colmap_root}")
    print(f"  cameras  : {n_frames} PINHOLE entries (.txt{bin_note.replace(' + ', '/')})")
    print(f"  images   : {n_frames} poses           (.txt{bin_note.replace(' + ', '/')})")
    print(f"  points3D : {len(pts):,} points          (.txt{bin_note.replace(' + ', '/')})")
    print(f"  images/  : {n_frames} PNG frames")
    print()

# ============================================================================
# Main
# ============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')

parser = argparse.ArgumentParser(description="DA3 inference → depth maps + point cloud + COLMAP data for gsplat")
parser.add_argument("scene_name", type=str, help="Name of the scene folder under DATA/")
parser.add_argument("--conf_percentile", type=float, default=5.0,
                    help="Bottom percentile of confidence scores to discard during point-cloud "
                         "merging (default: 90.0 → keeps only the top 10%% most confident pixels)")
parser.add_argument("--frame_step", type=int, default=10,
                    help="Use one frame every this many frames for point-cloud merging "
                         "(default: 10 → frames 0, 10, 20, …)")
args = parser.parse_args()
SCENE_NAME = args.scene_name

# Step 1 — load model
model, device = load_da3_model()
print("DA3 model loaded successfully\n")

# Step 2 — load images
image_files = load_images_from_folder(os.path.join(_PROJECT_ROOT, 'DATA', SCENE_NAME))

# Step 3 — run inference
prediction = run_da3_inference(model, image_files)

# Step 4 — build point cloud
points_3d, colors_3d, per_frame_points, per_frame_colors = merge_point_clouds(
    prediction, conf_percentile=args.conf_percentile, frame_step=args.frame_step
)

# Step 5 — clean point cloud
start = time.time()
clean_pts, clean_cols = clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=1.0)
print(f"[Scipy SOR] Cleaned: {clean_pts.shape[0]:,} points  ({time.time() - start:.2f}s)\n")

# Step 6 — persist results
save_point_cloud_ply(points_3d,  colors_3d,  SCENE_NAME, suffix='raw')
save_point_cloud_ply(clean_pts,  clean_cols, SCENE_NAME, suffix='cleaned')
save_depth_maps(prediction, SCENE_NAME)
save_colmap_data(prediction, points_3d, colors_3d, SCENE_NAME)