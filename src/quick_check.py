import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
import glob
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def visualize_depth_and_confidence(processed_images, depth, conf, sample_idx=0):
    """Visualize input image, depth map, and confidence map for a given sample index."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image
    if processed_images is not None:
        axes[0].imshow(processed_images[sample_idx])
    axes[0].set_title(f"Input Image ({sample_idx})")
    axes[0].axis('off')

    # Depth map
    depth_vis = visualize_depth(depth[sample_idx], cmap="Spectral")
    axes[1].imshow(depth_vis)
    axes[1].set_title(f"Depth Map ({sample_idx})")
    axes[1].axis('off')

    # Confidence map
    conf_map = conf[sample_idx]
    conf_norm = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min() + 1e-8)
    axes[2].imshow(conf_norm, cmap='viridis')
    axes[2].set_title(f"Confidence ({sample_idx})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


#%% Step 1.
def load_da3_model(model_name="depth-anything/DA3NESTED-GIANT-LARGE"):
    """Initialize Depth-Anything-3 model on available device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    return model, device

# Time to test step 1: Load the DA3 model
model, device = load_da3_model()
print("DA3 model loaded successfully")


def load_images_from_folder(data_path, extensions=['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']):
    """Scan folder and load all images with supported extensions"""
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(glob.glob(os.path.join(data_path, ext))))
    print(f"Found {len(image_files)} images in {data_path}")
    return image_files

# Time to Load your images
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')

parser = argparse.ArgumentParser(description="Depth-Anything-3 quick check")
parser.add_argument("scene_name", type=str, help="Name of the scene folder under DATA/")
args = parser.parse_args()
SCENE_NAME = args.scene_name

image_files = load_images_from_folder(os.path.join(_PROJECT_ROOT, 'DATA', SCENE_NAME))

#%% Step 3.
def run_da3_inference(model, image_files, process_res_method="upper_bound_resize"):
    """Run Depth-Anything-3 to get depth maps, camera poses, and intrinsics"""
    prediction = model.inference(
        image=image_files,
        infer_gs=False,
        process_res_method=process_res_method
    )
    print(f"Depth maps shape: {prediction.depth.shape}")
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"Confidence shape: {prediction.conf.shape}")
    return prediction

# Time to test step 3: Run DA3 inference
prediction = run_da3_inference(model, image_files)
# visualize_depth_and_confidence(prediction.processed_images, prediction.depth, prediction.conf, sample_idx=0)

def depth_to_point_cloud(depth_map, rgb_image, intrinsics, extrinsics, conf_map=None, conf_thresh=0.5):
    """Back-project depth map to 3D points using camera parameters"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # Filter by confidence if provided
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u, v, depth_map, rgb_image = u[valid_mask], v[valid_mask], depth_map[valid_mask], rgb_image[valid_mask]
    else:
        u, v, depth_map = u.flatten(), v.flatten(), depth_map.flatten()
        rgb_image = rgb_image.reshape(-1, 3)
    # Back-project to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    points_cam = np.stack([x, y, z], axis=-1)
    # Transform to world coordinates using extrinsics (w2c format)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R  # Inverse transform
    colors = rgb_image.astype(np.float32) / 255.0
    return points_world, colors


def visualize_point_cloud_open3d(points, colors=None, window_name="Point Cloud", point_size=1.0):
    """Visualize a 3D point cloud using Open3D.

    Args:
        points:      (N, 3) numpy array of XYZ coordinates.
        colors:      (N, 3) numpy array of RGB values in [0, 1], or None for uniform grey.
        window_name: Title of the Open3D viewer window.
        point_size:  Rendered point size (pixels).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        colors_f64 = np.clip(colors.astype(np.float64), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors_f64)

    # Optional: remove statistical outliers for a cleaner view
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.get_view_control().set_zoom(0.8)
    vis.run()
    vis.destroy_window()


def merge_point_clouds(prediction, conf_thresh=0.5):
    """Combine all frames into single point cloud, also returning per-frame clouds."""
    all_points = []
    all_colors = []
    n_frames = len(prediction.depth)
    for i in range(n_frames):
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
    print(f"Merged point cloud: {len(merged_points)} points")
    return merged_points, merged_colors, all_points, all_colors

# Generate 3D point cloud
points_3d, colors_3d, per_frame_points, per_frame_colors = merge_point_clouds(prediction, conf_thresh=0.4)

def clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=2.0):
    """
    Cleans a point cloud using SOR via Scipy cKDTree.
    """
    # 1. Build KD-Tree
    tree = cKDTree(points_3d)
    # 2. Query neighbors
    # k needs to be nb_neighbors + 1 because the point itself is included in results
    distances, _ = tree.query(points_3d, k=nb_neighbors + 1, workers=-1)
    # Exclude the first column (distance to self, which is 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    # 3. Calculate statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    # 4. Generate Mask
    distance_threshold = global_mean + (std_ratio * global_std)
    mask = mean_distances < distance_threshold
    return points_3d[mask], colors_3d[mask]

# Try Scipy method
start = time.time()
clean_pts_sci, clean_cols_sci = clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=1)
end = time.time()
print(f"\n[Scipy] Cleaned shape: {clean_pts_sci.shape}")
print(f"[Scipy] Time taken: {end - start:.4f} seconds")

# visualize_point_cloud_open3d(points_3d, colors_3d, window_name="Full Scene Point Cloud")


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

        # RGB input image  (uint8, H×W×3)
        plt.imsave(os.path.join(rgb_dir, fname), prediction.processed_images[i])

        # Depth map colourised with Spectral (uint8, H×W×3)
        depth_vis = visualize_depth(prediction.depth[i], cmap='Spectral')
        plt.imsave(os.path.join(depth_dir, fname), depth_vis)

        # Confidence map colourised with viridis (float [0,1] → RGBA → uint8)
        conf_map = prediction.conf[i]
        conf_norm = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min() + 1e-8)
        conf_vis = (cm_conf(conf_norm)[..., :3] * 255).astype(np.uint8)
        plt.imsave(os.path.join(conf_dir, fname), conf_vis)

    print(f"Saved {n_frames} depth-map frames → {base}")


# Persist results
save_point_cloud_ply(points_3d, colors_3d, SCENE_NAME, suffix='raw')
save_point_cloud_ply(clean_pts_sci, clean_cols_sci, SCENE_NAME, suffix='cleaned')
save_depth_maps(prediction, SCENE_NAME)
