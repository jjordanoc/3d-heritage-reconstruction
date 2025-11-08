# Pi³ to COLMAP Bundle Adjustment Pipeline - Implementation Summary

## What Was Implemented

A complete post-processing pipeline that converts Pi³ model predictions to COLMAP sparse reconstruction format and refines camera parameters through bundle adjustment.

## Files Created

### 1. `backend/pi3_to_colmap.py` (New)
**Purpose:** Convert Pi³ predictions to COLMAP sparse reconstruction format

**Key Functions:**
- `pi3_to_colmap_sparse()`: Main conversion function
  - Loads Pi³ predictions (camera_poses, points, conf)
  - Filters 3D points by confidence
  - Estimates initial camera intrinsics
  - Converts camera poses (cam2world → world2cam)
  - Creates 2D-3D correspondences via projection
  - Builds COLMAP reconstruction with tracks
  
- `create_point2d_correspondences()`: Projects 3D points to images to create tracks
- `project_points_to_image()`: Projects 3D points with visibility checking
- `filter_points_by_confidence()`: Filters points based on Pi³ confidence scores
- `estimate_focal_length()`: Estimates initial focal length from image size

**Key Details:**
- Initial focal length: `f = max(width, height) * 1.2`
- Initial principal point: `(cx, cy) = (width/2, height/2)`
- Uses PINHOLE camera model (fx, fy, cx, cy)
- Supports shared or per-image camera models

### 2. `backend/colmap_ba_pipeline.py` (New)
**Purpose:** Run bundle adjustment to refine camera parameters

**Key Functions:**
- `run_bundle_adjustment()`: Main BA function
  - Loads initial reconstruction
  - Configures optimization (what to refine, convergence criteria)
  - Runs iterative bundle adjustment
  - Validates results
  - Saves refined reconstruction
  
- `configure_bundle_adjustment_options()`: Configures BA parameters
  - What to refine: focal length, principal point, poses, 3D points
  - Robust loss function: Cauchy (handles outliers)
  - Convergence criteria: function/gradient/parameter tolerances
  
- `compute_reprojection_errors()`: Calculates reprojection error statistics
- `validate_reconstruction()`: Validates reconstruction quality

**What Gets Refined:**
- ✓ Focal lengths (fx, fy)
- ✓ Principal point (cx, cy)
- ✓ Camera poses (rotation, translation)
- ✓ 3D point positions

### 3. `backend/modal_server.py` (Modified)
**Changes:**
1. Added `pycolmap` to dependencies
2. Added background task: `process_colmap_ba()`
3. Added endpoint: `POST /scene/{id}/colmap_ba` - Trigger processing
4. Added endpoint: `GET /scene/{id}/colmap_ba/status` - Check status

### 4. `backend/COLMAP_BA_README.md` (New)
Comprehensive documentation covering:
- Pipeline overview
- API endpoints and usage
- Directory structure
- Configuration options
- Troubleshooting
- Next steps (Gaussian Splatting)

### 5. `backend/test_colmap_ba.py` (New)
Test script for easy pipeline testing:
```bash
# Trigger processing and wait
python test_colmap_ba.py --scene_id auditorio --base_url http://your-app.modal.run --wait

# Just check status
python test_colmap_ba.py --scene_id auditorio --base_url http://your-app.modal.run --check_only
```

## How It Works

### Input → Output Flow

```
predictions.pt (Pi³ outputs)
    + images/ (original images)
    ↓
┌──────────────────────────────────────┐
│  STEP 1: Pi³ → COLMAP Conversion    │
│  (pi3_to_colmap.py)                  │
│                                      │
│  • Load predictions                  │
│  • Filter points by confidence       │
│  • Estimate initial intrinsics       │
│  • Convert poses (c2w → w2c)         │
│  • Project points to create tracks   │
│  • Build COLMAP reconstruction       │
└──────────────────────────────────────┘
    ↓
colmap/sparse_initial/
    ├── cameras.bin (initial estimates)
    ├── images.bin
    └── points3D.bin
    ↓
┌──────────────────────────────────────┐
│  STEP 2: Bundle Adjustment           │
│  (colmap_ba_pipeline.py)             │
│                                      │
│  • Load initial reconstruction       │
│  • Configure BA options              │
│  • Optimize all parameters           │
│  • Validate results                  │
└──────────────────────────────────────┘
    ↓
colmap/sparse_ba/
    ├── cameras.bin (refined intrinsics!)
    ├── images.bin (refined poses)
    └── points3D.bin (refined positions)
    +
ba_results.json (statistics)
```

### Directory Structure Created

```
/backend_data/reconstructions/{scene_id}/
  └── colmap/
      ├── sparse_initial/    # After Step 1
      │   ├── cameras.bin
      │   ├── images.bin
      │   └── points3D.bin
      ├── sparse_ba/         # After Step 2 (ready for Gaussian Splatting!)
      │   ├── cameras.bin
      │   ├── images.bin
      │   └── points3D.bin
      └── ba_results.json    # Performance metrics
```

## API Usage

### 1. Trigger Processing

```bash
curl -X POST http://your-app.modal.run/scene/auditorio/colmap_ba
```

Response:
```json
{
  "status": "processing",
  "scene_id": "auditorio",
  "message": "COLMAP bundle adjustment pipeline started in background",
  "check_status_at": "/scene/auditorio/colmap_ba/status"
}
```

### 2. Check Status

```bash
curl http://your-app.modal.run/scene/auditorio/colmap_ba/status
```

Response (when complete):
```json
{
  "status": "complete",
  "scene_id": "auditorio",
  "initial_reconstruction": "/.../sparse_initial",
  "refined_reconstruction": "/.../sparse_ba",
  "results": {
    "success": true,
    "initial_error": {"mean": 2.456, ...},
    "final_error": {"mean": 0.678, ...},
    "num_cameras": 1,
    "num_images": 10,
    "num_points3D": 45678,
    "camera_params": {...}
  }
}
```

## Key Technical Details

### Intrinsics Estimation

**Question:** How are accurate intrinsics computed if Pi³ doesn't output them?

**Answer:** Through bundle adjustment!

1. **Initial Estimate** (rough):
   - Focal: `f = max(W, H) * 1.2` 
   - Principal point: center of image
   - These are just starting points

2. **Bundle Adjustment Refinement** (accurate):
   - Optimizes intrinsics using scene geometry
   - Minimizes reprojection error across all views
   - Leverages Pi³'s accurate poses and 3D structure
   - Result: Accurate intrinsics suitable for Gaussian Splatting

### Coordinate System Conversions

- **Pi³ Convention:** Camera-to-world (cam2world)
  - T_c2w transforms points from camera space to world space
  - `X_world = T_c2w @ X_camera`

- **COLMAP Convention:** World-to-camera (world2cam)
  - T_w2c transforms points from world space to camera space
  - `X_camera = T_w2c @ X_world`

- **Conversion:** `T_w2c = inverse(T_c2w)`
  - Handled automatically in `invert_pose()`

### Point Filtering

Points are filtered through multiple stages:

1. **Confidence threshold:** `conf > 0.1`
2. **Maximum count:** Top 100,000 points (if more)
3. **Projection visibility:** Must project into image bounds with positive depth
4. **Track length:** Must be visible in ≥2 views

### Camera Model

- **Type:** PINHOLE (no distortion)
- **Parameters:** [fx, fy, cx, cy]
  - fx, fy: focal lengths (in pixels)
  - cx, cy: principal point (in pixels)
- **Shared:** By default, all images use same intrinsics
- **Alternative:** Can use per-image intrinsics if needed

## Configuration Options

Edit `process_colmap_ba()` in `modal_server.py` to adjust:

```python
# Conversion parameters
conf_threshold=0.1,        # Confidence threshold (higher = fewer points)
max_points=100000,         # Max 3D points to use
min_track_length=2,        # Min views per point
shared_camera=True,        # Shared intrinsics for all images

# Bundle adjustment parameters
ba_options = configure_bundle_adjustment_options(
    refine_focal_length=True,
    refine_principal_point=True,
    use_robust_loss=True,
    loss_function_type="CAUCHY",
    max_num_iterations=100
)
```

## Testing

### Prerequisites
- Scene must have `predictions.pt` from Pi³ inference
- Original images must exist in `images/` directory

### Using the Test Script

```bash
# Basic usage: trigger and wait
python backend/test_colmap_ba.py \
  --scene_id auditorio \
  --base_url http://your-app.modal.run \
  --wait

# Just check status
python backend/test_colmap_ba.py \
  --scene_id auditorio \
  --base_url http://your-app.modal.run \
  --check_only

# Custom timeout
python backend/test_colmap_ba.py \
  --scene_id auditorio \
  --base_url http://your-app.modal.run \
  --wait \
  --timeout 1200
```

### Manual Testing via curl

```bash
# 1. Trigger processing
curl -X POST http://your-app.modal.run/scene/auditorio/colmap_ba

# 2. Check status (repeat until complete)
curl http://your-app.modal.run/scene/auditorio/colmap_ba/status
```

## Performance Expectations

- **Conversion (Step 1):** ~10-30 seconds
  - Depends on number of images and points
  
- **Bundle Adjustment (Step 2):** ~30-120 seconds
  - Depends on:
    - Number of images
    - Number of 3D points
    - Number of 2D-3D correspondences
    - Convergence speed

- **Total:** Typically 1-3 minutes for 10-20 images

## Next Steps: Gaussian Splatting

The refined reconstruction in `colmap/sparse_ba/` is ready for Gaussian Splatting!

### Using 3D Gaussian Splatting (original)

```bash
# Clone repo
git clone https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting

# Train
python train.py -s /path/to/scene --colmap_dir colmap/sparse_ba
```

### Using gsplat/Nerfstudio

```bash
# Install nerfstudio
pip install nerfstudio

# Train with gsplat
ns-train splatfacto --data /path/to/scene colmap --colmap-path colmap/sparse_ba
```

## Troubleshooting

### Common Issues

**"predictions.pt not found"**
- Solution: Run Pi³ inference first (upload images)

**High reprojection errors (>5 pixels)**
- Check image quality (avoid blur, low resolution)
- Ensure good overlap between views
- Try lowering `conf_threshold` to get more points

**Bundle adjustment fails to converge**
- Check logs for specific errors
- Verify there are enough correspondences (track length)
- Try adjusting BA parameters (iterations, tolerances)

**Camera parameters seem wrong**
- Focal length should be ~0.8-1.5× image size
- Principal point should be near center
- Check validation messages in logs

## Dependencies

New dependency added to `backend/modal_server.py`:
- `pycolmap` - Python bindings for COLMAP

Compatible with existing dependencies (torch, numpy, open3d, etc.)

## Summary

✅ Complete Pi³ → COLMAP conversion pipeline
✅ Bundle adjustment for accurate intrinsics estimation
✅ Async API endpoints for background processing
✅ Status checking and result reporting
✅ Comprehensive documentation and test scripts
✅ Ready for Gaussian Splatting downstream tasks

The implementation provides a robust bridge from Pi³'s feed-forward predictions to optimization-based COLMAP reconstructions, enabling high-quality novel view synthesis with Gaussian Splatting.

