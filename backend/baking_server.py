import modal
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import time
import base64

# ANSI Color codes for logging
class Colors:
    CYAN = '\033[96m'      # Function entry/exit
    YELLOW = '\033[93m'    # Timing info
    RED = '\033[91m'       # Warnings/Errors
    GREEN = '\033[92m'     # Success
    MAGENTA = '\033[95m'   # Data info
    RESET = '\033[0m'      # Reset color

def log_time(label, start_time):
    """Helper to log elapsed time in yellow"""
    elapsed = time.time() - start_time
    print(f"{Colors.YELLOW}⏱️  [{label}] took {elapsed:.3f}s{Colors.RESET}")
    return elapsed


cuda_version = "12.6.3"  # should be no greater than host CUDA version
flavor = "cudnn-devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# environment stuff
image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10").apt_install(
        "git",  # if you need it
        "libgl1-mesa-glx",  # Provides libGL.so.1
        "libglib2.0-0",     # Often needed by Open3D
    ).pip_install([
        "fastapi[standard]",
        "pillow",
        "python-multipart",
        "torch",
        "torchvision",
        "numpy",
        "open3d",
        "requests-toolbelt",
        "pycolmap==3.10.0",
    ])
    

volume = modal.Volume.from_name(name="ut3c-heritage", create_if_missing=True)


app = modal.App("ut3c-heritage-baking", image=image)
vol_mnt_loc = Path("/mnt/volume")
@app.function(
    image=image, 
    volumes={vol_mnt_loc: volume},
    timeout=1800,  # 30 minute timeout for COLMAP conversion
    scaledown_window=240,  # keep warm for 5 minutes
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Response, status, BackgroundTasks
    from fastapi.responses import JSONResponse, FileResponse
    from PIL import Image
    from fastapi import UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from requests_toolbelt.multipart.encoder import MultipartEncoder
    import os
    import uuid
    import io
    import json
    import torch
    import numpy as np
    import open3d as o3d
    import pycolmap
    import subprocess

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "*"  # For testing - remove in production
        ],
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allow all headers
    )

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
        images: np.ndarray,
        conf_threshold: float = 0.1,
        max_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter 3D points by confidence and extract corresponding RGB colors.
        
        Args:
            points: 3D points (N, H, W, 3) or (M, 3)
            conf: Confidence scores (N, H, W, 1) or (M, 1)
            images: RGB images (N, H, W, 3)
            conf_threshold: Minimum confidence threshold
            max_points: Maximum number of points to keep (random sampling if exceeded)
        
        Returns:
            filtered_points: Filtered 3D points (K, 3)
            filtered_conf: Filtered confidence scores (K,)
            filtered_colors: Filtered RGB colors (K, 3) in [0, 255] uint8
        """
        # Flatten if necessary
        if points.ndim == 4:
            points = points.reshape(-1, 3)
        if conf.ndim == 4:
            conf = conf.reshape(-1, 1)
        if images.ndim == 4:
            images = images.reshape(-1, 3)
        
        conf = conf.squeeze()
        
        # Apply confidence threshold
        mask = conf >= conf_threshold
        filtered_points = points[mask]
        filtered_conf = conf[mask]
        filtered_colors = images[mask]
        
        # Limit number of points if requested
        if max_points is not None and len(filtered_points) > max_points:
            indices = np.random.choice(len(filtered_points), max_points, replace=False)
            filtered_points = filtered_points[indices]
            filtered_conf = filtered_conf[indices]
            filtered_colors = filtered_colors[indices]
        
        # Convert colors from [0, 1] to [0, 255] uint8
        filtered_colors = (filtered_colors * 255).astype(np.uint8)
        
        return filtered_points, filtered_conf, filtered_colors


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
        images = predictions['images'].numpy()  # (B, N, H, W, 3)
        
        # Remove batch dimension (assume B=1)
        if camera_poses_c2w.shape[0] == 1:
            camera_poses_c2w = camera_poses_c2w[0]
            points_3d = points_3d[0]
            conf = conf[0]
            images = images[0]
        
        num_images = len(camera_poses_c2w)
        print(f"Found {num_images} images in predictions")
        
        # Load image metadata
        image_metadata = load_image_metadata(images_dir)
        assert len(image_metadata) == num_images, \
            f"Mismatch: {len(image_metadata)} images vs {num_images} predictions"
        
        # Filter 3D points by confidence
        print(f"Filtering points (threshold={conf_threshold}, max={max_points})...")
        filtered_points, filtered_conf, filtered_colors = filter_points_by_confidence(
            points_3d, conf, images, conf_threshold, max_points
        )
        print(f"Kept {len(filtered_points)} points after filtering")
        
        # Create COLMAP reconstruction
        reconstruction = pycolmap.Reconstruction()
        
        # Create camera models
        print("Creating camera models...")
        camera_params = []
        created_camera_ids = []
        
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
                    # Explicitly set camera ID to 1 for shared camera
                    camera_id = 1
                    camera = pycolmap.Camera(
                        model='PINHOLE',
                        width=width,
                        height=height,
                        params=[fx, fy, cx, cy],
                        camera_id=camera_id
                    )
                    reconstruction.add_camera(camera)
                    created_camera_ids.append(camera_id)
                    print(f"  Created shared camera ID: {camera_id}")
                    print(f"    Model: PINHOLE, Size: {width}x{height}")
                    print(f"    fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            else:
                # Use 1-based camera IDs
                camera_id = img_id + 1
                camera = pycolmap.Camera(
                    model='PINHOLE',
                    width=width,
                    height=height,
                    params=[fx, fy, cx, cy],
                    camera_id=camera_id
                )
                reconstruction.add_camera(camera)
                created_camera_ids.append(camera_id)
                print(f"  Created camera {img_id} with ID: {camera_id}")
        
        print(f"Total cameras created: {len(created_camera_ids)}")
        print(f"Camera IDs: {created_camera_ids}")
        
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
        filtered_colors = filtered_colors[valid_point_mask]
        tracks = [track for track, valid in zip(tracks, valid_point_mask) if valid]
        print(f"Kept {len(filtered_points)} points with sufficient track length")
        
        # Add 3D points to reconstruction (without track elements initially)
        print("Adding 3D points to reconstruction...")
        point_ids = []
        for pt_idx, (pt_3d, color) in enumerate(zip(filtered_points, filtered_colors)):
            # Create empty track
            track = pycolmap.Track()
            # Add point with actual RGB color from Pi³
            point_id = reconstruction.add_point3D(
                pt_3d,
                track,
                color  # Use actual color from images!
            )
            point_ids.append(point_id)
        
        # Add images to reconstruction
        print("Adding images to reconstruction...")
        for img_id, (img_meta, pose_w2c) in enumerate(zip(image_metadata, poses_w2c)):
            # Extract rotation and translation
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(pose_w2c[:3, :3]),
                pose_w2c[:3, 3]
            )
            
            # Determine camera_id for this image
            assigned_camera_id = created_camera_ids[0] if shared_camera else created_camera_ids[img_id]
            
            print(f"  Image {img_id}: '{img_meta['name']}' -> camera_id={assigned_camera_id}")
            
            # Create image
            image = pycolmap.Image(
                id=img_id + 1,
                name=img_meta['name'],
                camera_id=assigned_camera_id,
                cam_from_world=cam_from_world
            )
            
            # Build Point2D list for this image
            points2D_list = []
            point2D_idx = 0
            
            # Iterate through all 3D points and check if visible in this image
            for pt_idx, track in enumerate(tracks):
                point3D_id = point_ids[pt_idx]
                
                # Check if this point is visible in current image
                for track_img_id, pt_2d in track:
                    if track_img_id == img_id:
                        # Add Point2D with reference to 3D point
                        points2D_list.append(
                            pycolmap.Point2D(pt_2d, point3D_id)
                        )
                        
                        # Add track element to the 3D point
                        reconstruction.points3D[point3D_id].track.add_element(
                            img_id + 1,  # image_id (1-based)
                            point2D_idx  # point2D index in this image's point list
                        )
                        
                        point2D_idx += 1
                        break  # Each point appears at most once per image
            
            # Assign Point2D list to image
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
            
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
        
        # Print detailed camera info
        print(f"\nDetailed camera information:")
        for cam_id, camera in reconstruction.cameras.items():
            print(f"  Camera ID {cam_id}:")
            print(f"    Model: {camera.model}")
            print(f"    Size: {camera.width}x{camera.height}")
            print(f"    Params: {camera.params}")
        
        # Print image-camera mappings
        print(f"\nImage-to-camera mappings:")
        for img_id, img in reconstruction.images.items():
            print(f"  Image ID {img_id}: '{img.name}' -> camera_id={img.camera_id}, points2D={len(img.points2D)}")
        
        return reconstruction


    """
    COLMAP Bundle Adjustment Pipeline

    Runs bundle adjustment on COLMAP sparse reconstructions to refine:
    - Camera intrinsics (focal length, principal point)
    - Camera poses (rotation, translation)
    - 3D point positions

    This refines the initial Pi³ estimates to produce accurate camera parameters
    for downstream tasks like Gaussian Splatting.
    """

    import pycolmap
    from pathlib import Path
    from typing import Dict, Tuple, Optional
    import numpy as np


    def configure_bundle_adjustment_options(
        refine_focal_length: bool = True,
        refine_principal_point: bool = True,
        refine_extra_params: bool = False,
        use_robust_loss: bool = True,
        loss_function_type: str = "CAUCHY",
        loss_function_scale: float = 1.0,
        max_num_iterations: int = 100,
        max_linear_solver_iterations: int = 200,
        function_tolerance: float = 1e-6,
        gradient_tolerance: float = 1e-10,
        parameter_tolerance: float = 1e-8,
    ) -> pycolmap.BundleAdjustmentOptions:
        """
        Configure bundle adjustment options for COLMAP.
        
        Args:
            refine_focal_length: Optimize focal length parameters
            refine_principal_point: Optimize principal point
            refine_extra_params: Optimize distortion parameters
            use_robust_loss: Use robust loss function (recommended)
            loss_function_type: Type of robust loss (TRIVIAL, SOFT_L1, CAUCHY, HUBER)
            loss_function_scale: Scale parameter for robust loss
            max_num_iterations: Maximum BA iterations
            max_linear_solver_iterations: Maximum linear solver iterations
            function_tolerance: Convergence tolerance for cost function
            gradient_tolerance: Convergence tolerance for gradient
            parameter_tolerance: Convergence tolerance for parameters
        
        Returns:
            Configured BundleAdjustmentOptions
        """
        options = pycolmap.BundleAdjustmentOptions()
        
        # What to refine
        options.refine_focal_length = refine_focal_length
        options.refine_principal_point = refine_principal_point
        options.refine_extra_params = refine_extra_params
        options.refine_extrinsics = True  # Always refine poses
        
        # Robust loss function
        if use_robust_loss:
            if loss_function_type == "SOFT_L1":
                options.loss_function_type = pycolmap.LossFunctionType.SoftLOne
            elif loss_function_type == "CAUCHY":
                options.loss_function_type = pycolmap.LossFunctionType.Cauchy
            elif loss_function_type == "HUBER":
                options.loss_function_type = pycolmap.LossFunctionType.Huber
            else:
                options.loss_function_type = pycolmap.LossFunctionType.Trivial
            
            options.loss_function_scale = loss_function_scale
        
        # Solver configuration
        options.solver_options.max_num_iterations = max_num_iterations
        options.solver_options.max_linear_solver_iterations = max_linear_solver_iterations
        options.solver_options.function_tolerance = function_tolerance
        options.solver_options.gradient_tolerance = gradient_tolerance
        options.solver_options.parameter_tolerance = parameter_tolerance
        
        # Print summary
        options.print_summary = True
        
        return options


    def compute_reprojection_errors(reconstruction: pycolmap.Reconstruction) -> Dict[str, float]:
        """
        Compute reprojection error statistics for a reconstruction.
        
        Args:
            reconstruction: COLMAP Reconstruction object
        
        Returns:
            Dictionary with error statistics (mean, median, max, etc.)
        """
        errors = []
        
        for image_id, image in reconstruction.images.items():
            camera = reconstruction.cameras[image.camera_id]
            
            for point2D in image.points2D:
                if point2D.point3D_id == -1:
                    continue
                
                point3D = reconstruction.points3D[point2D.point3D_id]
                
                # Project 3D point to image
                projected = camera.img_from_cam(
                    image.project(point3D.xyz)
                )
                
                # Compute reprojection error
                error = np.linalg.norm(projected - point2D.xy)
                errors.append(error)
        
        errors = np.array(errors)
        
        if len(errors) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'num_observations': 0
            }
        
        return {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'num_observations': len(errors)
        }


    def validate_reconstruction(
        reconstruction: pycolmap.Reconstruction,
        max_reprojection_error: float = 10.0,
        min_focal_length: float = 0.5,
        max_focal_length: float = 3.0
    ) -> Tuple[bool, str]:
        """
        Validate reconstruction quality.
        
        Args:
            reconstruction: COLMAP Reconstruction object
            max_reprojection_error: Maximum acceptable mean reprojection error
            min_focal_length: Minimum acceptable focal length (as fraction of image size)
            max_focal_length: Maximum acceptable focal length (as fraction of image size)
        
        Returns:
            (is_valid, message): Validation result and description
        """
        # Check if reconstruction has content
        if reconstruction.num_cameras() == 0:
            return False, "No cameras in reconstruction"
        
        if reconstruction.num_images() == 0:
            return False, "No images in reconstruction"
        
        if reconstruction.num_points3D() == 0:
            return False, "No 3D points in reconstruction"
        
        # Check reprojection errors
        errors = compute_reprojection_errors(reconstruction)
        if errors['mean'] > max_reprojection_error:
            return False, f"Mean reprojection error too high: {errors['mean']:.2f} pixels"
        
        # Check camera parameters
        for camera_id, camera in reconstruction.cameras.items():
            # Get focal length as fraction of image size
            focal_x = camera.params[0] / camera.width
            focal_y = camera.params[1] / camera.height
            
            if focal_x < min_focal_length or focal_x > max_focal_length:
                return False, f"Camera {camera_id}: fx out of range ({focal_x:.2f})"
            
            if focal_y < min_focal_length or focal_y > max_focal_length:
                return False, f"Camera {camera_id}: fy out of range ({focal_y:.2f})"
            
            # Check principal point is near center
            cx_normalized = camera.params[2] / camera.width
            cy_normalized = camera.params[3] / camera.height
            
            if abs(cx_normalized - 0.5) > 0.3:
                return False, f"Camera {camera_id}: cx too far from center ({cx_normalized:.2f})"
            
            if abs(cy_normalized - 0.5) > 0.3:
                return False, f"Camera {camera_id}: cy too far from center ({cy_normalized:.2f})"
        
        return True, f"Valid reconstruction with {errors['mean']:.2f}px mean error"


    def run_bundle_adjustment(
        input_dir: Path,
        output_dir: Path,
        ba_options: Optional[pycolmap.BundleAdjustmentOptions] = None,
        validate: bool = True
    ) -> Dict:
        """
        Run bundle adjustment on a COLMAP sparse reconstruction.
        
        Args:
            input_dir: Directory containing initial sparse reconstruction
            output_dir: Directory to save refined reconstruction
            ba_options: Bundle adjustment options (uses defaults if None)
            validate: Whether to validate reconstruction after BA
        
        Returns:
            Dictionary with results and statistics
        """
        print(f"Loading reconstruction from {input_dir}...")
        reconstruction = pycolmap.Reconstruction(input_dir)
        
        # Print initial statistics
        print(f"\nInitial reconstruction:")
        print(f"  Cameras: {reconstruction.num_cameras()}")
        print(f"  Images: {reconstruction.num_images()}")
        print(f"  Points: {reconstruction.num_points3D()}")
        
        initial_errors = compute_reprojection_errors(reconstruction)
        print(f"  Initial reprojection error: {initial_errors['mean']:.3f} ± {initial_errors['std']:.3f} pixels")
        
        # Configure BA options
        if ba_options is None:
            ba_options = configure_bundle_adjustment_options()
        
        # Run bundle adjustment
        print("\nRunning bundle adjustment...")
        summary = pycolmap.bundle_adjustment(reconstruction, ba_options)
        
        # Compute final errors
        final_errors = compute_reprojection_errors(reconstruction)
        print(f"\nFinal reprojection error: {final_errors['mean']:.3f} ± {final_errors['std']:.3f} pixels")
        
        # Print camera parameters
        print("\nRefined camera parameters:")
        for camera_id, camera in reconstruction.cameras.items():
            print(f"  Camera {camera_id}: {camera.model}")
            print(f"    Size: {camera.width}x{camera.height}")
            if camera.model == 'PINHOLE':
                fx, fy, cx, cy = camera.params
                print(f"    fx={fx:.2f}, fy={fy:.2f}")
                print(f"    cx={cx:.2f}, cy={cy:.2f}")
                print(f"    Focal (normalized): fx={fx/camera.width:.3f}, fy={fy/camera.height:.3f}")
        
        # Validate reconstruction
        if validate:
            is_valid, message = validate_reconstruction(reconstruction)
            print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'} - {message}")
        else:
            is_valid = True
            message = "Validation skipped"
        
        # Save refined reconstruction
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving refined reconstruction to {output_dir}...")
        reconstruction.write(output_dir)
        
        # Return results
        results = {
            'success': is_valid,
            'validation_message': message,
            'initial_error': initial_errors,
            'final_error': final_errors,
            'num_cameras': reconstruction.num_cameras(),
            'num_images': reconstruction.num_images(),
            'num_points3D': reconstruction.num_points3D(),
            'camera_params': {}
        }
        
        # Store camera parameters
        for camera_id, camera in reconstruction.cameras.items():
            results['camera_params'][camera_id] = {
                'model': str(camera.model),
                'width': camera.width,
                'height': camera.height,
                'params': camera.params.tolist()
            }
        
        return results


    def run_full_pipeline(
        predictions_path: Path,
        images_dir: Path,
        output_base_dir: Path,
        **kwargs
    ) -> Dict:
        """
        Run the complete Pi³ → COLMAP → BA pipeline.
        
        Args:
            predictions_path: Path to predictions.pt
            images_dir: Directory with original images
            output_base_dir: Base directory for outputs (will create sparse_initial/ and sparse_ba/)
            **kwargs: Additional arguments for pi3_to_colmap_sparse and run_bundle_adjustment
        
        Returns:
            Dictionary with pipeline results
        """
        # from pi3_to_colmap import pi3_to_colmap_sparse
        
        # Step 1: Convert Pi³ to COLMAP
        sparse_initial_dir = output_base_dir / "sparse_initial"
        print("="*80)
        print("STEP 1: Converting Pi³ predictions to COLMAP format")
        print("="*80)
        
        reconstruction_initial = pi3_to_colmap_sparse(
            predictions_path,
            images_dir,
            sparse_initial_dir,
            **kwargs
        )
        
        # Step 2: Run bundle adjustment
        sparse_ba_dir = output_base_dir / "sparse_ba"
        print("\n" + "="*80)
        print("STEP 2: Running bundle adjustment")
        print("="*80)
        
        ba_results = run_bundle_adjustment(
            sparse_initial_dir,
            sparse_ba_dir
        )
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Initial reconstruction: {sparse_initial_dir}")
        print(f"Refined reconstruction: {sparse_ba_dir}")
        print(f"Mean reprojection error: {ba_results['initial_error']['mean']:.3f} → {ba_results['final_error']['mean']:.3f} pixels")
        
        return {
            'initial_dir': str(sparse_initial_dir),
            'refined_dir': str(sparse_ba_dir),
            'ba_results': ba_results
        }

    

    async def process_colmap_ba(scene_id: str):
        """
        Background task to run COLMAP bundle adjustment pipeline.
        
        Args:
            scene_id: The scene/reconstruction ID
        """
        func_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}🔧 [BACKGROUND: process_colmap_ba] STARTING for scene {scene_id}{Colors.RESET}")
        print(f"{'='*80}\n")
        
        try:
            # Reload volume to get latest data
            reload_start = time.time()
            volume.reload()
            log_time("Volume reload", reload_start)
            
            # Define paths
            scene_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / scene_id
            predictions_path = scene_path / "predictions.pt"
            images_dir = scene_path / "images"
            colmap_base_dir = scene_path / "colmap"
            
            print(f"{Colors.MAGENTA}   Predictions: {predictions_path}{Colors.RESET}")
            print(f"{Colors.MAGENTA}   Images dir: {images_dir}{Colors.RESET}")
            print(f"{Colors.MAGENTA}   Output dir: {colmap_base_dir}{Colors.RESET}")
            
            # Validate inputs
            if not predictions_path.exists():
                raise FileNotFoundError(f"predictions.pt not found at {predictions_path}")
            
            if not images_dir.exists() or not any(images_dir.iterdir()):
                raise FileNotFoundError(f"No images found in {images_dir}")
            
            # Import conversion and BA modules
            import sys
            backend_path = Path(__file__).parent
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
            
            # from pi3_to_colmap import pi3_to_colmap_sparse
            # from colmap_ba_pipeline import run_bundle_adjustment
            
            # Step 1: Convert Pi³ to COLMAP
            sparse_initial_dir = colmap_base_dir / "sparse_initial"
            print(f"\n{Colors.CYAN}STEP 1: Converting Pi³ predictions to COLMAP...{Colors.RESET}")
            convert_start = time.time()
            
            reconstruction_initial = pi3_to_colmap_sparse(
                predictions_path=predictions_path,
                images_dir=images_dir,
                output_dir=sparse_initial_dir,
                conf_threshold=0.1,
                max_points=100000,
                min_track_length=2,
                shared_camera=True
            )
            
            log_time("Pi³ to COLMAP conversion", convert_start)
            print(f"{Colors.GREEN}✅ Initial COLMAP reconstruction saved to {sparse_initial_dir}{Colors.RESET}")
            
            # Commit volume so splat_server can read COLMAP data
            commit_start = time.time()
            volume.commit()
            log_time("Volume commit", commit_start)
            print(f"{Colors.GREEN}✅ COLMAP data committed to volume{Colors.RESET}")
            
            # Step 2: Call splat_server to train Gaussian Splatting
            print(f"\n{Colors.CYAN}STEP 2: Calling splat_server for gsplat training...{Colors.RESET}")
            splat_start = time.time()
            
            try:
                # Import splat server function using Modal's cross-app calling
                train_gsplat_fn = modal.Function.from_name(
                    "ut3c-heritage-splat", 
                    "train_gsplat_modal"
                )
                
                # Call and wait for completion (blocking)
                train_gsplat_fn.remote(scene_id, data_factor=1, max_steps=30000)
                
                log_time("Gsplat training", splat_start)
                print(f"{Colors.GREEN}✅ Gsplat training completed successfully{Colors.RESET}")
                
            except Exception as e:
                # Fail the entire pipeline if gsplat fails
                error_msg = f"Gsplat training failed: {str(e)}"
                print(f"{Colors.RED}❌ {error_msg}{Colors.RESET}")
                raise RuntimeError(error_msg)
            
            print(f"\n{'='*80}")
            print(f"{Colors.GREEN}✅ [BACKGROUND: process_colmap_ba] COMPLETE for scene {scene_id}{Colors.RESET}")
            print(f"   COLMAP conversion and gsplat training successful")
            log_time("🎯 TOTAL BACKGROUND TASK TIME", func_start)
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"{Colors.RED}❌ [BACKGROUND: process_colmap_ba] FAILED for scene {scene_id}{Colors.RESET}")
            print(f"{Colors.RED}   Error: {str(e)}{Colors.RESET}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()

    @web_app.post("/scene/{id}/colmap_ba")
    async def run_colmap_ba_pipeline(id: str, background_tasks: BackgroundTasks):
        """
        Post-processing: Convert Pi³ predictions to COLMAP format and run bundle adjustment.
        Returns immediately with job status, processing happens in background.
        
        This endpoint:
        1. Converts Pi³ predictions to COLMAP sparse reconstruction format
        2. Runs bundle adjustment to refine camera poses and estimate accurate intrinsics
        3. Saves results to /backend_data/reconstructions/{id}/colmap/
        
        Args:
            id: Scene/reconstruction ID
        
        Returns:
            JSON with status and job information
        """
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}🚀 [POST /scene/{id}/colmap_ba] ENDPOINT CALLED{Colors.RESET}")
        print(f"{'='*80}\n")
        
        # Validate that predictions.pt exists
        predictions_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / id / "predictions.pt"
        
        if not predictions_path.exists():
            print(f"{Colors.RED}❌ ERROR: predictions.pt not found at {predictions_path}{Colors.RESET}")
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for scene {id}. Run inference first."
            )
        
        # Add background task
        background_tasks.add_task(process_colmap_ba, id)
        
        print(f"{Colors.GREEN}✅ Background task scheduled for scene {id}{Colors.RESET}")
        log_time("Endpoint response time", endpoint_start)
        print(f"{'='*80}\n")
        
        return {
            "status": "processing",
            "scene_id": id,
            "message": "COLMAP bundle adjustment pipeline started in background",
            "check_status_at": f"/scene/{id}/colmap_ba/status"
        }

    @web_app.get("/scene/{id}/colmap_ba/status")
    async def get_colmap_ba_status(id: str):
        """
        Check the status of COLMAP bundle adjustment processing for a scene.
        
        Args:
            id: Scene/reconstruction ID
        
        Returns:
            JSON with processing status and results (if complete)
        """
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}📊 [GET /scene/{id}/colmap_ba/status] ENDPOINT CALLED{Colors.RESET}")
        print(f"{'='*80}\n")
        
        colmap_dir = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / id / "colmap"
        sparse_initial_dir = colmap_dir / "sparse_initial"
        sparse_ba_dir = colmap_dir / "sparse_ba"
        results_file = colmap_dir / "ba_results.json"
        
        # Check if processing is complete
        if sparse_ba_dir.exists() and results_file.exists():
            # Load results
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print(f"{Colors.GREEN}✅ COLMAP BA processing complete for scene {id}{Colors.RESET}")
                log_time("Status check", endpoint_start)
                print(f"{'='*80}\n")
                
                return {
                    "status": "complete",
                    "scene_id": id,
                    "initial_reconstruction": str(sparse_initial_dir),
                    "refined_reconstruction": str(sparse_ba_dir),
                    "results": results
                }
            except Exception as e:
                print(f"{Colors.RED}⚠️  WARNING: Could not load results file: {e}{Colors.RESET}")
                return {
                    "status": "complete",
                    "scene_id": id,
                    "initial_reconstruction": str(sparse_initial_dir),
                    "refined_reconstruction": str(sparse_ba_dir),
                    "error": "Results file exists but could not be read"
                }
        
        elif sparse_initial_dir.exists():
            print(f"{Colors.YELLOW}⏳ COLMAP BA processing in progress for scene {id}{Colors.RESET}")
            log_time("Status check", endpoint_start)
            print(f"{'='*80}\n")
            
            return {
                "status": "processing",
                "scene_id": id,
                "message": "Initial conversion complete, bundle adjustment in progress"
            }
        
        else:
            print(f"{Colors.YELLOW}⏳ COLMAP BA not started or failed for scene {id}{Colors.RESET}")
            log_time("Status check", endpoint_start)
            print(f"{'='*80}\n")
            
            return {
                "status": "not_started",
                "scene_id": id,
                "message": "COLMAP bundle adjustment not started or failed"
            }

    return web_app