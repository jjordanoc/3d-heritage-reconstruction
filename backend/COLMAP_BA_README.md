# Pi³ to COLMAP Bundle Adjustment Pipeline

This pipeline converts Pi³ model predictions to COLMAP sparse reconstruction format and runs bundle adjustment to refine camera poses and estimate accurate intrinsics for downstream tasks like Gaussian Splatting.

## Overview

The pipeline consists of two main steps:

1. **Pi³ → COLMAP Conversion**: Converts Pi³ predictions (camera poses, 3D points, confidence) to COLMAP format
2. **Bundle Adjustment**: Refines camera intrinsics, poses, and 3D point positions

## Directory Structure

After running the pipeline, your reconstruction will have:

```
/backend_data/reconstructions/{scene_id}/
  ├── images/                   # Original input images
  ├── thumbnails/               # Scene thumbnails
  ├── predictions.pt            # Pi³ raw predictions
  ├── models/
  │   └── latest.ply           # Current point cloud
  └── colmap/                   # NEW: COLMAP outputs
      ├── sparse_initial/       # Initial Pi³→COLMAP conversion
      │   ├── cameras.bin
      │   ├── images.bin
      │   └── points3D.bin
      ├── sparse_ba/            # Refined after bundle adjustment
      │   ├── cameras.bin
      │   ├── images.bin
      │   └── points3D.bin
      └── ba_results.json       # Bundle adjustment statistics
```

## API Endpoints

### 1. Start Bundle Adjustment Processing

```bash
POST /scene/{scene_id}/colmap_ba
```

Starts the COLMAP bundle adjustment pipeline in the background.

**Requirements:**
- The scene must have `predictions.pt` (run Pi³ inference first)
- Original images must exist in the `images/` directory

**Response:**
```json
{
  "status": "processing",
  "scene_id": "auditorio",
  "message": "COLMAP bundle adjustment pipeline started in background",
  "check_status_at": "/scene/auditorio/colmap_ba/status"
}
```

**Example:**
```bash
curl -X POST http://your-modal-app.modal.run/scene/auditorio/colmap_ba
```

### 2. Check Processing Status

```bash
GET /scene/{scene_id}/colmap_ba/status
```

Check the status of bundle adjustment processing.

**Response (Processing):**
```json
{
  "status": "processing",
  "scene_id": "auditorio",
  "message": "Initial conversion complete, bundle adjustment in progress"
}
```

**Response (Complete):**
```json
{
  "status": "complete",
  "scene_id": "auditorio",
  "initial_reconstruction": "/mnt/volume/backend_data/reconstructions/auditorio/colmap/sparse_initial",
  "refined_reconstruction": "/mnt/volume/backend_data/reconstructions/auditorio/colmap/sparse_ba",
  "results": {
    "success": true,
    "initial_error": {
      "mean": 2.456,
      "median": 1.823,
      "std": 1.234
    },
    "final_error": {
      "mean": 0.678,
      "median": 0.512,
      "std": 0.345
    },
    "num_cameras": 1,
    "num_images": 10,
    "num_points3D": 45678,
    "camera_params": {
      "1": {
        "model": "PINHOLE",
        "width": 512,
        "height": 384,
        "params": [614.32, 612.89, 256.0, 192.0]
      }
    }
  }
}
```

**Example:**
```bash
curl http://your-modal-app.modal.run/scene/auditorio/colmap_ba/status
```

## Pipeline Details

### Step 1: Pi³ to COLMAP Conversion

**What happens:**
- Loads `predictions.pt` containing Pi³ outputs
- Extracts camera poses (cam2world), 3D points, and confidence scores
- Loads original image metadata (dimensions, filenames)
- Filters 3D points by confidence (threshold: 0.1)
- Estimates initial camera intrinsics:
  - Focal length: `f = max(width, height) * 1.2`
  - Principal point: `(cx, cy) = (width/2, height/2)`
- Converts camera poses from cam2world to world2cam
- Projects 3D points to images to create 2D-3D correspondences
- Creates tracks (points visible in multiple views)
- Saves COLMAP sparse reconstruction

**Configuration:**
- `conf_threshold`: 0.1 (minimum confidence for points)
- `max_points`: 100,000 (maximum 3D points to keep)
- `min_track_length`: 2 (minimum views per point)
- `shared_camera`: True (same intrinsics for all images)

### Step 2: Bundle Adjustment

**What happens:**
- Loads initial sparse reconstruction
- Configures bundle adjustment options:
  - Refine focal length ✓
  - Refine principal point ✓
  - Refine camera poses ✓
  - Refine 3D point positions ✓
  - Use Cauchy robust loss function
- Runs iterative optimization to minimize reprojection error
- Saves refined reconstruction and statistics

**Output:**
- Refined camera intrinsics (now accurate!)
- Refined camera poses
- Refined 3D point positions
- Reprojection error statistics

## Camera Intrinsics Estimation

**Important:** Pi³ doesn't output camera intrinsics, so we estimate them through bundle adjustment:

1. **Initial Estimate** (Step 1): We provide rough initial values based on image size
2. **Refinement** (Step 2): Bundle adjustment optimizes these parameters using the scene geometry

Bundle adjustment can accurately recover intrinsics when given:
- Good initial pose estimates (✓ from Pi³)
- Multiple overlapping views (✓ from multi-view reconstruction)
- Sufficient 2D-3D correspondences (✓ from point projections)

The refined intrinsics in `sparse_ba/cameras.bin` are suitable for Gaussian Splatting and other downstream tasks.

## Using COLMAP Outputs

### View Reconstruction in COLMAP GUI

```bash
# Copy files from Modal volume to local machine
# Then visualize:
colmap gui
# File → Import model → Select sparse_ba/ directory
```

### Convert to Text Format

```python
import pycolmap

reconstruction = pycolmap.Reconstruction("sparse_ba/")
reconstruction.write_text("sparse_ba_text/")
# Creates cameras.txt, images.txt, points3D.txt
```

### Extract Camera Parameters

```python
import pycolmap

reconstruction = pycolmap.Reconstruction("sparse_ba/")

for camera_id, camera in reconstruction.cameras.items():
    print(f"Camera {camera_id}:")
    print(f"  Model: {camera.model}")
    print(f"  Size: {camera.width}x{camera.height}")
    
    if camera.model == 'PINHOLE':
        fx, fy, cx, cy = camera.params
        print(f"  Focal: fx={fx:.2f}, fy={fy:.2f}")
        print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
```

## Next Steps: Gaussian Splatting

The refined COLMAP reconstruction in `sparse_ba/` is ready for Gaussian Splatting training. You can use it with:

- [3D Gaussian Splatting (original)](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [Nerfstudio](https://docs.nerf.studio/)

Example with original 3DGS:
```bash
python train.py -s /path/to/scene --colmap_dir colmap/sparse_ba
```

## Troubleshooting

### "predictions.pt not found"
- Run Pi³ inference first by uploading images to the scene
- Check that `/backend_data/reconstructions/{scene_id}/predictions.pt` exists

### High reprojection errors
- Ensure input images have good overlap
- Check that images are not too blurry or low quality
- Try adjusting `conf_threshold` or `min_track_length` parameters

### Bundle adjustment fails
- Check logs for specific error messages
- Ensure there are enough 3D-2D correspondences
- Verify images directory contains valid images

### Camera parameters seem unreasonable
- Check that focal length is approximately 0.8-1.5× image size
- Verify principal point is near image center
- Review input image quality and camera motion

## Configuration Options

You can modify the pipeline behavior by editing `process_colmap_ba()` in `modal_server.py`:

```python
# In pi3_to_colmap_sparse()
conf_threshold=0.1,        # Higher = fewer but more confident points
max_points=100000,         # Maximum 3D points to use
min_track_length=2,        # Minimum views per point
shared_camera=True,        # Use same intrinsics for all images

# In run_bundle_adjustment()
ba_options = configure_bundle_adjustment_options(
    refine_focal_length=True,
    refine_principal_point=True,
    loss_function_type="CAUCHY",
    max_num_iterations=100
)
```

## References

- [COLMAP Documentation](https://colmap.github.io/)
- [PyColmap API](https://github.com/colmap/pycolmap)
- [Bundle Adjustment Explanation](https://en.wikipedia.org/wiki/Bundle_adjustment)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

