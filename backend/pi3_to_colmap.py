"""
Pi³ to COLMAP Sparse Reconstruction Converter

Converts Pi³ model predictions (camera poses, 3D points, confidence) to COLMAP format
for bundle adjustment and downstream tasks like Gaussian Splatting.

Key conversions:
- Pi³ camera_poses (cam2world) → COLMAP world2cam
- Pi³ confidence filtering → COLMAP 3D points
- Projection-based 2D-3D correspondences → COLMAP tracks
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pycolmap
from PIL import Image


def invert_pose(T_c2w: np.ndarray) -> np.ndarray:
    """
    Invert SE(3) transformation matrix.
    
    Args:
        T_c2w: Camera-to-world transformation (4x4)
    
    Returns:
        T_w2c: World-to-camera transformation (4x4)
    """
    T_w2c = np.eye(4, dtype=T_c2w.dtype)
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t
    
    T_w2c[:3, :3] = R_inv
    T_w2c[:3, 3] = t_inv
    
    return T_w2c


def estimate_focal_length(width: int, height: int, fov_factor: float = 1.2) -> float:
    """
    Estimate initial focal length for pinhole camera.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_factor: Multiplier for focal length (higher = narrower FOV)
    
    Returns:
        Estimated focal length in pixels
    """
    return max(width, height) * fov_factor


def project_points_to_image(
    points_3d: np.ndarray,
    T_w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    margin: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image plane and filter visible points.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        T_w2c: World-to-camera transformation (4x4)
        fx, fy: Focal lengths
        cx, cy: Principal point
        width, height: Image dimensions
        margin: Pixel margin from image boundary
    
    Returns:
        points_2d: 2D projections (M, 2) for visible points
        visible_mask: Boolean mask (N,) indicating visibility
    """
    # Transform to camera coordinates
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]
    points_cam = (R @ points_3d.T).T + t
    
    # Filter points behind camera
    visible_mask = points_cam[:, 2] > 0.1  # Min depth 0.1
    
    # Project to image plane
    points_2d = np.zeros((len(points_3d), 2))
    points_2d[:, 0] = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    points_2d[:, 1] = fy * points_cam[:, 1] / points_cam[:, 2] + cy
    
    # Filter points outside image bounds (with margin)
    in_bounds = (
        (points_2d[:, 0] >= margin) &
        (points_2d[:, 0] < width - margin) &
        (points_2d[:, 1] >= margin) &
        (points_2d[:, 1] < height - margin)
    )
    
    visible_mask = visible_mask & in_bounds
    
    return points_2d, visible_mask


def filter_points_by_confidence(
    points: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float = 0.1,
    max_points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter 3D points by confidence and optionally limit total count.
    
    Args:
        points: 3D points (N, H, W, 3) or (M, 3)
        conf: Confidence scores (N, H, W, 1) or (M, 1)
        conf_threshold: Minimum confidence threshold
        max_points: Maximum number of points to keep (random sampling if exceeded)
    
    Returns:
        filtered_points: Filtered 3D points (K, 3)
        filtered_conf: Filtered confidence scores (K,)
    """
    # Flatten if necessary
    if points.ndim == 4:
        points = points.reshape(-1, 3)
    if conf.ndim == 4:
        conf = conf.reshape(-1, 1)
    
    conf = conf.squeeze()
    
    # Apply confidence threshold
    mask = conf >= conf_threshold
    filtered_points = points[mask]
    filtered_conf = conf[mask]
    
    # Limit number of points if requested
    if max_points is not None and len(filtered_points) > max_points:
        indices = np.random.choice(len(filtered_points), max_points, replace=False)
        filtered_points = filtered_points[indices]
        filtered_conf = filtered_conf[indices]
    
    return filtered_points, filtered_conf


def load_image_metadata(images_dir: Path) -> List[Dict]:
    """
    Load metadata for all images in a directory.
    
    Args:
        images_dir: Directory containing images
    
    Returns:
        List of dicts with keys: 'path', 'name', 'width', 'height'
    """
    image_files = sorted(
        [f for f in images_dir.glob("*") 
         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    )
    
    metadata = []
    for img_path in image_files:
        img = Image.open(img_path)
        metadata.append({
            'path': img_path,
            'name': img_path.name,
            'width': img.width,
            'height': img.height
        })
    
    return metadata


def create_point2d_correspondences(
    points_3d: np.ndarray,
    camera_params: List[Dict],
    poses_w2c: List[np.ndarray],
    min_track_length: int = 2
) -> Tuple[List[List[Tuple[int, np.ndarray]]], np.ndarray]:
    """
    Create 2D-3D correspondences by projecting points to images.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        camera_params: List of camera parameter dicts (fx, fy, cx, cy, width, height)
        poses_w2c: List of world-to-camera poses (4x4)
        min_track_length: Minimum number of views for a valid track
    
    Returns:
        tracks: List of tracks, each track is list of (image_id, point2d)
        valid_point_mask: Boolean mask indicating points with valid tracks
    """
    num_points = len(points_3d)
    num_images = len(camera_params)
    
    # For each point, collect which images it's visible in
    tracks = [[] for _ in range(num_points)]
    
    for img_id in range(num_images):
        cam = camera_params[img_id]
        T_w2c = poses_w2c[img_id]
        
        points_2d, visible = project_points_to_image(
            points_3d, T_w2c,
            cam['fx'], cam['fy'], cam['cx'], cam['cy'],
            cam['width'], cam['height']
        )
        
        # Add visible points to their tracks
        for pt_id in np.where(visible)[0]:
            tracks[pt_id].append((img_id, points_2d[pt_id]))
    
    # Filter points with insufficient track length
    valid_point_mask = np.array([len(track) >= min_track_length for track in tracks])
    
    return tracks, valid_point_mask


def pi3_to_colmap_sparse(
    predictions_path: Path,
    images_dir: Path,
    output_dir: Path,
    conf_threshold: float = 0.1,
    max_points: int = 100000,
    min_track_length: int = 2,
    shared_camera: bool = True
) -> pycolmap.Reconstruction:
    """
    Convert Pi³ predictions to COLMAP sparse reconstruction format.
    
    Args:
        predictions_path: Path to predictions.pt file
        images_dir: Directory containing original images
        output_dir: Directory to save COLMAP sparse reconstruction
        conf_threshold: Minimum confidence for points
        max_points: Maximum number of 3D points
        min_track_length: Minimum track length for valid points
        shared_camera: Use same intrinsics for all images
    
    Returns:
        COLMAP Reconstruction object
    """
    print("Loading Pi³ predictions...")
    predictions = torch.load(predictions_path, map_location='cpu', weights_only=False)
    
    # Extract data
    camera_poses_c2w = predictions['camera_poses'].numpy()  # (B, N, 4, 4)
    points_3d = predictions['points'].numpy()  # (B, N, H, W, 3)
    conf = predictions['conf'].numpy()  # (B, N, H, W, 1)
    
    # Remove batch dimension (assume B=1)
    if camera_poses_c2w.shape[0] == 1:
        camera_poses_c2w = camera_poses_c2w[0]
        points_3d = points_3d[0]
        conf = conf[0]
    
    num_images = len(camera_poses_c2w)
    print(f"Found {num_images} images in predictions")
    
    # Load image metadata
    image_metadata = load_image_metadata(images_dir)
    assert len(image_metadata) == num_images, \
        f"Mismatch: {len(image_metadata)} images vs {num_images} predictions"
    
    # Filter 3D points by confidence
    print(f"Filtering points (threshold={conf_threshold}, max={max_points})...")
    filtered_points, filtered_conf = filter_points_by_confidence(
        points_3d, conf, conf_threshold, max_points
    )
    print(f"Kept {len(filtered_points)} points after filtering")
    
    # Create COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction()
    
    # Create camera models
    print("Creating camera models...")
    camera_params = []
    
    for img_id, img_meta in enumerate(image_metadata):
        width = img_meta['width']
        height = img_meta['height']
        
        # Estimate intrinsics
        focal = estimate_focal_length(width, height)
        fx = fy = focal
        cx = width / 2.0
        cy = height / 2.0
        
        camera_params.append({
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': width, 'height': height
        })
        
        # Add camera to reconstruction
        if shared_camera:
            if img_id == 0:  # Only create one shared camera
                camera = pycolmap.Camera(
                    model='PINHOLE',
                    width=width,
                    height=height,
                    params=[fx, fy, cx, cy]
                )
                camera_id = reconstruction.add_camera(camera)
        else:
            camera = pycolmap.Camera(
                model='PINHOLE',
                width=width,
                height=height,
                params=[fx, fy, cx, cy]
            )
            camera_id = reconstruction.add_camera(camera)
    
    # Convert poses to world-to-camera
    print("Converting camera poses...")
    poses_w2c = [invert_pose(pose_c2w) for pose_c2w in camera_poses_c2w]
    
    # Create 2D-3D correspondences
    print("Creating 2D-3D correspondences...")
    tracks, valid_point_mask = create_point2d_correspondences(
        filtered_points, camera_params, poses_w2c, min_track_length
    )
    
    # Filter points with valid tracks
    filtered_points = filtered_points[valid_point_mask]
    filtered_conf = filtered_conf[valid_point_mask]
    tracks = [track for track, valid in zip(tracks, valid_point_mask) if valid]
    print(f"Kept {len(filtered_points)} points with sufficient track length")
    
    # Add 3D points to reconstruction
    print("Adding 3D points to reconstruction...")
    point_ids = []
    for pt_idx, (pt_3d, track) in enumerate(zip(filtered_points, tracks)):
        point3D = pycolmap.Point3D()
        point3D.xyz = pt_3d
        point3D.color = np.array([128, 128, 128], dtype=np.uint8)  # Gray color
        point3D.error = 0.0
        
        # Add track elements
        for img_id, pt_2d in track:
            track_element = pycolmap.Track()
            track_element.image_id = img_id + 1  # COLMAP uses 1-based indexing
            track_element.point2D_idx = pt_idx  # Will be updated when adding images
            point3D.track.add_element(track_element)
        
        point_id = reconstruction.add_point3D(point3D)
        point_ids.append(point_id)
    
    # Add images to reconstruction
    print("Adding images to reconstruction...")
    for img_id, (img_meta, pose_w2c) in enumerate(zip(image_metadata, poses_w2c)):
        # Extract rotation and translation
        qvec = pycolmap.rotmat_to_qvec(pose_w2c[:3, :3])
        tvec = pose_w2c[:3, 3]
        
        # Create image
        image = pycolmap.Image()
        image.name = img_meta['name']
        image.camera_id = 1 if shared_camera else img_id + 1
        
        # Set pose
        image.qvec = qvec
        image.tvec = tvec
        
        # Add 2D points
        for pt_idx, track in enumerate(tracks):
            for track_img_id, pt_2d in track:
                if track_img_id == img_id:
                    point2D = pycolmap.Point2D()
                    point2D.xy = pt_2d
                    point2D.point3D_id = point_ids[pt_idx]
                    image.points2D.append(point2D)
        
        # Add image to reconstruction
        reconstruction.add_image(image)
    
    # Save reconstruction
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving reconstruction to {output_dir}...")
    reconstruction.write(output_dir)
    
    # Print summary
    print(f"\nReconstruction summary:")
    print(f"  Cameras: {reconstruction.num_cameras()}")
    print(f"  Images: {reconstruction.num_images()}")
    print(f"  Points: {reconstruction.num_points3D()}")
    
    return reconstruction

