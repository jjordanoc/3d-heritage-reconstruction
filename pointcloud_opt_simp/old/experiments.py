import os
import shutil
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import open3d as o3d

def preprocess_images(src_fldr: str, dst_fldr: str, major_axis_size: int = 512):
    """
    Resizes images in a source folder so their major (longer) axis is the specified size,
    and saves them to a destination folder.

    Args:
        src_fldr: Path to the source directory containing images.
        dst_fldr: Path to the destination directory for resized images.
        major_axis_size: The target size for the major (longer) axis (e.g., 512).
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(dst_fldr):
        os.makedirs(dst_fldr)
        print(f"Created destination folder: {dst_fldr}")

    # List of common image extensions to process
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    # Iterate through all files in the source folder
    for i, filename in enumerate(os.listdir(src_fldr)):
        if filename.lower().endswith(image_extensions):
            src_path = os.path.join(src_fldr, filename)
            dst_path = os.path.join(dst_fldr, f"{i:04d}.png")

            try:
                # Open the image
                img = Image.open(src_path)
                width, height = img.size

                # 1. Determine the scaling factor based on the major axis
                if width >= height:
                    # Major axis is width
                    scale_factor = major_axis_size / width
                else:
                    # Major axis is height
                    scale_factor = major_axis_size / height
                
                # 2. Calculate new dimensions
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                new_size = (new_width, new_height)

                # 3. Resize and save
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save the resized image to the destination folder
                # We use the original format and quality settings
                resized_img.save(dst_path,format="png")

                print(f"Processed: {filename} ({width}x{height} -> {new_width}x{new_height})")

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")


def find_keyframes(src, dst, n_keyframes=15, device="cuda"):
    os.makedirs(dst, exist_ok=True)

    # --- 1. Model and transform setup ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # remove classifier head
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # --- 2. Gather embeddings ---
    files = sorted([f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    embeds, valid_files = [], []

    for f in tqdm(files, desc="Embedding images"):
        try:
            img = Image.open(os.path.join(src, f)).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x).cpu().numpy().flatten()
            embeds.append(feat / np.linalg.norm(feat))
            valid_files.append(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    embeds = np.vstack(embeds)
    n_total = len(valid_files)
    print(f"\nExtracted embeddings for {n_total} images.")

    if n_total == 0:
        print("No valid images found.")
        return

    # --- 3. Diversity selection ---
    selected = [0]  # start with first image
    idxs = list(range(1, n_total))

    while len(selected) < min(n_keyframes, n_total):
        sims = cosine_similarity(embeds[idxs], embeds[selected])
        min_sims = sims.max(axis=1)  # similarity to nearest selected
        next_idx = idxs[int(np.argmin(min_sims))]  # most different
        selected.append(next_idx)
        idxs.remove(next_idx)

    # --- 4. Save selected keyframes ---
    for i, idx in enumerate(selected):
        shutil.copy(os.path.join(src, valid_files[idx]), os.path.join(dst, valid_files[idx]))
        print(f"Keyframe {i+1}: {valid_files[idx]}")

    print(f"\n✅ Saved {len(selected)} diverse keyframes to {dst}")

def prediction_to_ply(prediction_file: str, models_folder: str, conf_thres: float = 1) -> list[str]:
    """
    Loads a PyTorch prediction file (.pt) with shape (1, S, H, W, F),
    squeezes the leading dimension, applies confidence filtering, and outputs
    N+1 PLY files: 1 for the whole scene and 1 for each of the N frames.
    
    Args:
        prediction_file (str): Full path to the input PyTorch predictions file (.pt).
        models_folder (str): The directory where the PLY files will be saved.
        conf_thres (float): Percentage threshold (0-100) for point confidence filtering.

    Returns:
        list[str]: A list of file paths for the generated PLY files.
    """
    # 1. Setup Output Paths
    output_path = Path(models_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []
    conf_threshold = conf_thres / 100.0
    
    print(f"Loading predictions from: {prediction_file}")

    # 2. Load Predictions
    try:
        predictions = torch.load(prediction_file, weights_only=False, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading prediction file {prediction_file}: {e}")
        return []

    # 3. Extract, Squeeze, and Prepare Data (NumPy arrays)
    
    # ⚠️ CRITICAL FIX: Squeeze the leading dimension (index 0) to get (S, H, W, F)
    pred_world_points = predictions["points"].squeeze(0).numpy().astype(np.float64)  # Now (18, 434, 574, 3)
    images = predictions["images"].squeeze(0).numpy().astype(np.float64)            # Now (18, 434, 574, 3)

    # Handle optional 'conf' key, also squeezing the leading dimension
    pred_world_points_conf = predictions.get(
        "conf", 
        torch.ones_like(predictions["points"][..., 0])
    ).squeeze(0).numpy().astype(np.float64) # Now (18, 434, 574)
    
    # Handle image format (check for NCHW and transpose to NHWC is still relevant if model output is NCHW)
    if images.ndim == 4 and images.shape[1] == 3:
        # If the shape was (S, 3, H, W), it would be transposed here. 
        # Since your shape is (18, 434, 574, 3), it's already in NHWC. 
        # This check remains as a good practice against model format variability.
        images = np.transpose(images, (0, 2, 3, 1))

    # Determine the number of frames (S = 18)
    num_frames = pred_world_points.shape[0]
    
    if num_frames == 0:
        print("Error: Extracted data has zero frames.")
        return []
        
    print(f"Detected {num_frames} frames for processing.")
    
    all_vertices = []
    all_colors = []

    # 4. Process Per-Frame PLY Files
    for i in range(num_frames):
        # Frame slicing is now correct, iterating over the 18 frames
        frame_points = pred_world_points[i].reshape(-1, 3)
        frame_colors_uint8 = (images[i].reshape(-1, 3) * 255.0).astype(np.uint8) 
        frame_conf = pred_world_points_conf[i].reshape(-1)

        # Apply Confidence Filter
        mask = (frame_conf >= conf_threshold) & (frame_conf > 1e-5)
        
        filtered_vertices = frame_points[mask]
        filtered_colors = frame_colors_uint8[mask]

        if filtered_vertices.size > 0:
            # Accumulate data for the whole scene PLY
            all_vertices.append(filtered_vertices)
            all_colors.append(filtered_colors)
            
            # Create and save Open3D point cloud for the frame
            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(filtered_vertices)
            pcd_frame.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)

            frame_ply_path = output_path / f"frame_{i:04d}.ply"
            o3d.io.write_point_cloud(str(frame_ply_path), pcd_frame, write_ascii=False)
            saved_files.append(str(frame_ply_path))
            
    # 5. Create Whole Scene PLY File
    if all_vertices:
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_colors = np.concatenate(all_colors, axis=0)

        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(final_vertices)
        pcd_full.colors = o3d.utility.Vector3dVector(final_colors / 255.0)
        
        full_ply_path = output_path / "latest_full_scene.ply"
        full_ply_path_down = output_path / "latest_full_scene_down.ply"
        o3d.io.write_point_cloud(str(full_ply_path), pcd_full, write_ascii=False)
        down = pcd_full.random_down_sample(sampling_ratio=0.9)
        o3d.io.write_point_cloud(str(full_ply_path_down), down, write_ascii=False)
        saved_files.append(str(full_ply_path))
        print(f"Successfully saved {len(saved_files)} PLY files to {models_folder}.")
    else:
        print("Warning: No points exceeded the confidence threshold across all frames.")

    return saved_files


if __name__ == "__main__":
    #preprocess_images("pointcloud_opt_simp/images","pointcloud_opt_simp/images_preprocessed")
    #find_keyframes("pointcloud_opt_simp/images_preprocessed","pointcloud_opt_simp/kayframes")
    prediction_to_ply("pointcloud_opt_simp/predictions/predictions.pt","pointcloud_opt_simp/models")