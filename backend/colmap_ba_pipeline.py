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
    from pi3_to_colmap import pi3_to_colmap_sparse
    
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

