# Gaussian Splatting Training with gsplat

This pipeline trains 3D Gaussian Splatting models from COLMAP reconstructions using the `gsplat` library.

## Overview

The gsplat training pipeline takes the initial COLMAP reconstruction (from Pi³ predictions) and trains a high-quality 3D Gaussian Splatting model that can be rendered from novel viewpoints.

## Directory Structure

```
/backend_data/reconstructions/{scene_id}/
  ├── images/                        # Input images
  ├── colmap/
  │   └── sparse_initial/            # COLMAP reconstruction (input)
  │       ├── cameras.bin
  │       ├── images.bin
  │       └── points3D.bin
  ├── sparse/                        # Created for gsplat compatibility
  │   └── 0/                         # Symlink to sparse_initial/
  └── gsplat/                        # Training outputs
      ├── config.json                # Training configuration
      ├── results.ply                # Final trained Gaussians (MAIN OUTPUT!)
      ├── results/                   # Intermediate outputs
      │   ├── training_log.txt       # Training logs
      │   └── checkpoints/           # Model checkpoints
      └── error.log                  # Error log (if training fails)
```

## API Endpoints

### 1. Start Training

```bash
POST /scene/{scene_id}/train_gsplat?data_factor=1&max_steps=30000
```

Starts Gaussian Splatting training in the background.

**Prerequisites:**
- COLMAP sparse reconstruction exists (`colmap/sparse_initial/`)
- Images directory exists and contains images

**Parameters:**
- `data_factor` (int, optional): Image downscale factor
  - `1`: Full resolution (default, best quality)
  - `2`: Half resolution (2x faster)
  - `4`: Quarter resolution (4x faster, for testing)
- `max_steps` (int, optional): Training iterations (default: 30000)
  - More steps = better quality but slower
  - Typical values: 7000 (fast), 15000 (good), 30000 (high quality)

**Response:**
```json
{
  "status": "processing",
  "scene_id": "mijato",
  "message": "Gaussian Splatting training started in background",
  "config": {
    "data_factor": 1,
    "max_steps": 30000
  },
  "check_status_at": "/scene/mijato/train_gsplat/status"
}
```

**Example:**
```bash
# Full resolution, 30K iterations (high quality, ~15 min)
curl -X POST "http://your-app.modal.run/scene/mijato/train_gsplat?data_factor=1&max_steps=30000"

# Quarter resolution, 7K iterations (fast test, ~2 min)
curl -X POST "http://your-app.modal.run/scene/mijato/train_gsplat?data_factor=4&max_steps=7000"
```

### 2. Check Training Status

```bash
GET /scene/{scene_id}/train_gsplat/status
```

Check the status of training.

**Response (Processing):**
```json
{
  "status": "processing",
  "scene_id": "mijato",
  "message": "Gaussian Splatting training in progress",
  "progress": "Iteration 15234/30000 | Loss: 0.0234"
}
```

**Response (Complete):**
```json
{
  "status": "complete",
  "scene_id": "mijato",
  "model_path": "/mnt/volume/backend_data/reconstructions/mijato/gsplat/results.ply",
  "model_size_mb": 127.45,
  "config": {
    "scene_id": "mijato",
    "data_factor": 1,
    "max_steps": 30000,
    "timestamp": 1699564821.234
  },
  "download_url": "/scene/mijato/gsplat/model"
}
```

**Response (Failed):**
```json
{
  "status": "failed",
  "scene_id": "mijato",
  "error": "CUDA out of memory. Try reducing data_factor or image resolution."
}
```

**Example:**
```bash
curl "http://your-app.modal.run/scene/mijato/train_gsplat/status"
```

### 3. Download Trained Model

```bash
GET /scene/{scene_id}/gsplat/model
```

Download the trained .ply file containing the 3D Gaussians.

**Response:**
- File download: `{scene_id}_gaussian_splatting.ply`
- Content-Type: `application/octet-stream`

**Example:**
```bash
# Download to file
curl "http://your-app.modal.run/scene/mijato/gsplat/model" \
  -o mijato_gaussian_splatting.ply

# Or use wget
wget "http://your-app.modal.run/scene/mijato/gsplat/model" \
  -O mijato_gaussian_splatting.ply
```

## Complete Workflow

### Step-by-step Process

```bash
# 1. Upload images and run Pi³ inference (creates predictions.pt)
curl -X POST "http://your-app.modal.run/scene/mijato" \
  -F "thumbnail=@thumbnail.jpg"

# Upload images
for img in images/*.jpg; do
  curl -X POST "http://your-app.modal.run/pointcloud/mijato" \
    -F "file=@$img"
done

# 2. Convert to COLMAP format (creates sparse_initial/)
curl -X POST "http://your-app.modal.run/scene/mijato/colmap_ba"

# Wait for COLMAP conversion to complete
while true; do
  STATUS=$(curl -s "http://your-app.modal.run/scene/mijato/colmap_ba/status" | jq -r '.status')
  echo "COLMAP status: $STATUS"
  [[ "$STATUS" == "complete" ]] && break
  sleep 10
done

# 3. Train Gaussian Splatting
curl -X POST "http://your-app.modal.run/scene/mijato/train_gsplat?max_steps=30000"

# Wait for training to complete
while true; do
  STATUS=$(curl -s "http://your-app.modal.run/scene/mijato/train_gsplat/status" | jq -r '.status')
  echo "Training status: $STATUS"
  [[ "$STATUS" == "complete" ]] && break
  sleep 15
done

# 4. Download trained model
curl "http://your-app.modal.run/scene/mijato/gsplat/model" \
  -o mijato_gaussian_splatting.ply
```

### Using the Test Script

```bash
# Full workflow: trigger and wait for completion
python backend/test_gsplat.py \
  --scene_id mijato \
  --base_url http://your-app.modal.run \
  --wait \
  --download

# Just check status
python backend/test_gsplat.py \
  --scene_id mijato \
  --base_url http://your-app.modal.run \
  --check_only

# Fast test with low quality
python backend/test_gsplat.py \
  --scene_id mijato \
  --base_url http://your-app.modal.run \
  --data_factor 4 \
  --max_steps 7000 \
  --wait
```

## Configuration Guide

### data_factor (Image Resolution)

| Value | Resolution | Training Time | Quality | Use Case |
|-------|------------|---------------|---------|----------|
| 1 | Full | ~15-20 min | Highest | Production |
| 2 | Half (2x faster) | ~7-10 min | Good | Quick results |
| 4 | Quarter (4x faster) | ~3-5 min | Acceptable | Testing |

**Recommendation:** Start with `data_factor=4` for testing, then use `data_factor=1` for final production model.

### max_steps (Training Iterations)

| Steps | Training Time | Quality | Use Case |
|-------|---------------|---------|----------|
| 7,000 | ~2-5 min | Good | Quick preview |
| 15,000 | ~5-10 min | Very good | Fast production |
| 30,000 | ~10-20 min | Excellent | High quality |

**Recommendation:** 30,000 steps for final models, 7,000 for quick tests.

## Performance Expectations

### Training Time (on A10G GPU)

| Configuration | Typical Duration |
|---------------|------------------|
| 5 images, 1/4 res, 7K steps | ~2-3 minutes |
| 5 images, full res, 7K steps | ~5-7 minutes |
| 10 images, full res, 30K steps | ~15-20 minutes |
| 20 images, full res, 30K steps | ~25-35 minutes |

### GPU Memory Usage

- Typical: 6-10 GB VRAM
- Peak: Up to 12 GB VRAM for large scenes
- Recommendation: A10G (24GB) or better

### Model File Sizes

- Small scenes (5-10 images): 50-150 MB
- Medium scenes (10-20 images): 150-300 MB
- Large scenes (20+ images): 300-500 MB

## Rendering the Results

### Using gsplat Viewer

```python
import torch
from gsplat import rasterization

# Load trained model
model = torch.load("results.ply")

# Render from novel viewpoint
# (requires camera parameters)
rendered_image = rasterization(model, camera_params)
```

### Using Online Viewers

The trained `.ply` file contains 3D Gaussians and can be:
- Viewed with Gaussian Splatting web viewers
- Rendered with gsplat's built-in viewer
- Integrated into 3D applications

### Serving to Frontend

The .ply file can be served directly to web-based viewers for interactive visualization.

## Troubleshooting

### Common Errors

**"COLMAP reconstruction not found"**
- Solution: Run the COLMAP conversion endpoint first
- Command: `POST /scene/{id}/colmap_ba`

**"No images found"**
- Solution: Upload images first
- Check that images exist in `{scene_id}/images/`

**"CUDA out of memory"**
- Solution: Reduce `data_factor` (try 2 or 4)
- Or reduce image resolution before upload
- Or use fewer images

**Training crashes / hangs**
- Check training log: `{scene_id}/gsplat/results/training_log.txt`
- Verify COLMAP reconstruction is valid
- Try with `data_factor=4` first

**"No .ply file generated"**
- Training may have failed silently
- Check error log: `{scene_id}/gsplat/error.log`
- Verify gsplat is installed correctly

### Checking Logs

```bash
# View training log via Modal
modal volume get ut3c-heritage \
  backend_data/reconstructions/mijato/gsplat/results/training_log.txt

# Check for errors
modal volume get ut3c-heritage \
  backend_data/reconstructions/mijato/gsplat/error.log
```

## Technical Details

### gsplat Data Structure

gsplat's `simple_trainer.py` expects:
```
data_dir/
  ├── images/              # RGB images
  │   ├── DSC_001.jpg
  │   └── ...
  └── sparse/
      └── 0/               # COLMAP reconstruction
          ├── cameras.bin
          ├── images.bin
          └── points3D.bin
```

Our pipeline creates a symlink:
- `sparse/0/` → `colmap/sparse_initial/`

This allows gsplat to read our COLMAP reconstruction without copying files.

### Training Process

1. **Data Loading**: gsplat loads images and COLMAP cameras
2. **Initialization**: 3D Gaussians initialized from COLMAP point cloud
3. **Optimization**: Iteratively refine Gaussian parameters to match images
4. **Densification**: Add/remove Gaussians to improve quality
5. **Export**: Save final Gaussians to .ply file

### Output Format

The `results.ply` file contains:
- 3D Gaussian centers (xyz positions)
- Colors (RGB)
- Opacity values
- Covariance matrices (shape/orientation)
- Spherical harmonics (view-dependent appearance)

## Integration with Frontend

### Streaming the Model

```javascript
// Download and display
fetch('/scene/mijato/gsplat/model')
  .then(response => response.blob())
  .then(blob => {
    // Load into Gaussian Splatting viewer
    loadGaussianSplattingModel(blob);
  });
```

### Progressive Loading

For large models, consider:
1. Streaming download with progress updates
2. Level-of-detail loading
3. Spatial partitioning for large scenes

## Comparison with Other Methods

| Method | Quality | Speed | Memory | Rendering |
|--------|---------|-------|--------|-----------|
| **Gaussian Splatting** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Real-time |
| NeRF | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Slow |
| Mesh Reconstruction | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time |

**Why Gaussian Splatting:**
- ✅ Photo-realistic quality
- ✅ Real-time rendering (60+ FPS)
- ✅ View-dependent effects
- ✅ Compact representation
- ✅ Fast training

## References

- [gsplat Documentation](https://github.com/nerfstudio-project/gsplat)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP Documentation](https://colmap.github.io/)
- [Pi³ Model](https://github.com/yyfz/Pi3)

## Support

For issues with:
- **API endpoints**: Check server logs
- **Training failures**: Check error.log and training_log.txt
- **Performance**: Try reducing data_factor or max_steps
- **Quality**: Increase max_steps, use data_factor=1

