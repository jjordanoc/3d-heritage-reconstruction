import modal
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
from pathlib import Path
import time

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

# environment stuff
image = modal.Image.debian_slim(python_version="3.10").apt_install(
        "git",  # if you need it
        "libgl1-mesa-glx",  # Provides libGL.so.1
        "libglib2.0-0",     # Often needed by Open3D
    ).pip_install([
        "fastapi[standard]",
        "pillow",
        "python-multipart",
        "torch",
        "numpy",
        "open3d",
        "requests-toolbelt"
    ])

volume = modal.Volume.from_name(name="ut3c-heritage")


#application stuff
app = modal.App("ut3c-heritage-backend-brute", image=image)
vol_mnt_loc = Path("/mnt/volume")
@app.function(image=image,volumes={vol_mnt_loc:volume})
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
    Pi3Remote = modal.Cls.from_name("pi3-inference", "ModelInference")
    inference_obj = Pi3Remote()

    @web_app.on_event("startup")
    async def lifespan():
        folder_path = vol_mnt_loc / "backend_data" / "reconstructions" / "auditorio" / "images"
        #shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized folder: {folder_path}")
    

    def simplify_cloud(cld):
        cl, _ = cld.remove_statistical_outlier(20, 2.0)
        return cl.voxel_down_sample(0.05)

    def extract_last_image_pointcloud(inf_data_path, id, conf_thres=50):
        """
        Extracts the pointcloud and camera pose for the LAST (most recent) image.
        Returns: tuple (ply_path, camera_pose)
        """
        func_start = time.time()
        print(f"{Colors.CYAN}🔍 [extract_last_image_pointcloud] ENTER - id={id}, conf_thres={conf_thres}{Colors.RESET}")
        print(f"{Colors.MAGENTA}   Input path: {inf_data_path}{Colors.RESET}")
        
        load_start = time.time()
        predictions = torch.load(inf_data_path, map_location="cpu")
        log_time("torch.load predictions", load_start)
        
        # Extract camera pose for the last image [batch=0, image=-1]
        pose_start = time.time()
        camera_poses = predictions.get("camera_poses")
        camera_pose = None
        if camera_poses is not None:
            # Handle batch dimension [1, N, ...] format
            if camera_poses.dim() >= 2:
                camera_pose = camera_poses[0][-1].numpy().tolist()
            else:
                camera_pose = camera_poses[-1].numpy().tolist()
            print(f"{Colors.MAGENTA}   Camera pose extracted: shape={camera_poses.shape}, dims={camera_poses.dim()}{Colors.RESET}")
        else:
            print(f"{Colors.RED}⚠️  WARNING: No camera_poses found in predictions{Colors.RESET}")
        log_time("Extract camera pose", pose_start)
        
        # Extract pointcloud data for last image
        extract_start = time.time()
        pred_world_points = predictions["points"]
        pred_world_points_conf = predictions.get("conf", torch.ones_like(predictions["points"]))
        images = predictions["images"]
        
        print(f"{Colors.MAGENTA}   Predictions keys: {list(predictions.keys())}{Colors.RESET}")
        print(f"{Colors.MAGENTA}   Points shape: {pred_world_points.shape}, dtype: {pred_world_points.dtype}{Colors.RESET}")
        print(f"{Colors.MAGENTA}   Images shape: {images.shape}, dtype: {images.dtype}{Colors.RESET}")
        
        # Get last image data [batch=0, image=-1]
        last_points = pred_world_points[0][-1].numpy()  # [H*W, 3]
        last_conf = pred_world_points_conf[0][-1].numpy()  # [H*W]
        last_colors = images[0][-1].numpy()  # [H*W, 3]
        
        print(f"{Colors.MAGENTA}   Last image points: {last_points.shape}, conf: {last_conf.shape}, colors: {last_colors.shape}{Colors.RESET}")
        log_time("Extract pointcloud data arrays", extract_start)
        
        # Reshape and apply confidence threshold
        filter_start = time.time()
        vertices_3d = last_points.reshape(-1, 3)
        colors_rgb = (last_colors.reshape(-1, 3) * 255).astype(np.uint8)
        conf = last_conf.reshape(-1)
        
        print(f"{Colors.MAGENTA}   Conf range: [{conf.min():.4f}, {conf.max():.4f}], mean: {conf.mean():.4f}{Colors.RESET}")
        
        conf_threshold = conf_thres / 100.0
        mask = (conf >= conf_threshold) & (conf > 1e-5)
        print(f"{Colors.MAGENTA}   Confidence threshold: {conf_threshold}, kept {mask.sum()} / {len(mask)} points ({100*mask.sum()/len(mask):.1f}%){Colors.RESET}")
        
        vertices_3d = vertices_3d[mask]
        colors_rgb = colors_rgb[mask]
        
        if len(vertices_3d) == 0:
            print(f"{Colors.RED}❌ ERROR: No points kept after filtering!{Colors.RESET}")
            log_time("extract_last_image_pointcloud (FAILED)", func_start)
            return None, camera_pose
        
        log_time("Filter points by confidence", filter_start)
        
        # Create Open3D point cloud
        pcd_start = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
        log_time("Create Open3D point cloud", pcd_start)
        
        # Save temporary PLY
        save_start = time.time()
        temp_folder = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / str(id) / "models"
        temp_folder.mkdir(parents=True, exist_ok=True)
        temp_ply_path = temp_folder / "temp_latest.ply"
        
        o3d.io.write_point_cloud(str(temp_ply_path), pcd)
        log_time("Save PLY file", save_start)
        
        print(f"{Colors.GREEN}✅ [extract_last_image_pointcloud] SUCCESS - saved to {temp_ply_path}{Colors.RESET}")
        log_time("extract_last_image_pointcloud TOTAL", func_start)
        
        return str(temp_ply_path), camera_pose

    async def upload_image(id: str, file: UploadFile = File(...)):
        """
            Takes a project id and a file and preprocesses and puts the image 
            into the correct path
            id: str an id that matches one of the existing reconstructions
            file: UploadFile a file represnrting an image
            returns: Path, the direction of the stored image
        """
        func_start = time.time()
        print(f"{Colors.CYAN}📤 [upload_image] ENTER - id={id}, filename={file.filename}, content_type={file.content_type}{Colors.RESET}")
        
        path = vol_mnt_loc / "backend_data" / "reconstructions" / id / "images"
        print(f"{Colors.MAGENTA}   Target path: {path}{Colors.RESET}")
        
        if not os.path.exists(path) or not os.path.isdir(path):
            print(f"{Colors.RED}❌ ERROR: Path does not exist or is not a directory: {path}{Colors.RESET}")
            raise HTTPException(status_code=400,
                                 detail="The provided recosntruction id does not exist")
        if not file.content_type.startswith("image/"):
            print(f"{Colors.RED}❌ ERROR: Invalid content type: {file.content_type}{Colors.RESET}")
            raise HTTPException(status_code=400, detail="The sent file is not an image")
        
        # Load image with Pillow
        read_start = time.time()
        img = None
        try:
            contents = await file.read()
            file_size = len(contents)
            print(f"{Colors.MAGENTA}   File size: {file_size / 1024:.2f} KB{Colors.RESET}")
            img = Image.open(io.BytesIO(contents))
            print(f"{Colors.MAGENTA}   Original image size: {img.size}, mode: {img.mode}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ ERROR: Failed to load image: {e}{Colors.RESET}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        finally:
            file.file.seek(0)  # reset pointer if needed
        log_time("Read and load image", read_start)

        target_size = 512
        resize_start = time.time()
        
        width, height = img.size
        if width >= height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)

        print(f"{Colors.MAGENTA}   Resizing from {width}x{height} to {new_width}x{new_height}{Colors.RESET}")
        img = img.resize((new_width, new_height), Image.BICUBIC)
        log_time("Resize image", resize_start)
        
        save_start = time.time()
        uuidname = uuid.uuid4()
        save_path = path / f"{uuidname}.png"
        img.save(save_path, format="PNG")
        log_time("Save image to disk", save_start)
        
        commit_start = time.time()
        volume.commit()
        log_time("Volume commit", commit_start)
        
        print(f"{Colors.GREEN}✅ [upload_image] SUCCESS - saved to {save_path}{Colors.RESET}")
        log_time("upload_image TOTAL", func_start)
        return save_path

    async def run_inference(id: str):
        """
        Runs inference on the specified project.
        id: the id of a project
        returns: Path the path of the .pt object
        """
        func_start = time.time()
        print(f"{Colors.CYAN}🧠 [run_inference] ENTER - id={id}{Colors.RESET}")
        
        images_path = f"/backend_data/reconstructions/{id}/images"
        print(f"{Colors.MAGENTA}   Images path (relative to volume): {images_path}{Colors.RESET}")
        
        remote_start = time.time()
        print(f"{Colors.YELLOW}⏳ Starting remote inference call...{Colors.RESET}")
        inf_res_path = inference_obj.run_inference.remote(images_path)
        log_time("REMOTE INFERENCE (Pi3)", remote_start)
        
        result_path = str(vol_mnt_loc) + inf_res_path + "/predictions.pt"
        print(f"{Colors.GREEN}✅ [run_inference] SUCCESS - predictions at {result_path}{Colors.RESET}")
        log_time("run_inference TOTAL", func_start)
        return result_path
    
    def process_infered_data(inf_data_path,id,conf_thres = 50):
        """
        Process full reconstruction (all images combined)
        """
        func_start = time.time()
        print(f"{Colors.CYAN}🔧 [process_infered_data] ENTER - id={id}, conf_thres={conf_thres}{Colors.RESET}")
        
        pty_location = str(vol_mnt_loc) + f"/backend_data/reconstructions/{id}/models/latest.ply"
        pty_folder = str(vol_mnt_loc) + f"/backend_data/reconstructions/{id}/models"
        print(f"{Colors.MAGENTA}   Target PLY: {pty_location}{Colors.RESET}")
        
        Path(pty_folder).mkdir(parents=True, exist_ok=True)
        
        vol_start = time.time()
        volume.commit()
        volume.reload()
        log_time("Volume commit/reload", vol_start)
        
        load_start = time.time()
        predictions = torch.load(inf_data_path,weights_only=False,map_location=torch.device('cpu'))
        log_time("Load predictions.pt", load_start)

        process_start = time.time()
        pred_world_points = predictions["points"].numpy()
        pred_world_points_conf = predictions.get(
            "conf",
            torch.ones_like(predictions["points"])
        ).numpy()
        images = predictions["images"].numpy()
        
        print(f"{Colors.MAGENTA}   Points shape: {pred_world_points.shape}, dtype: {pred_world_points.dtype}{Colors.RESET}")
        print(f"{Colors.MAGENTA}   Images shape: {images.shape}, dtype: {images.dtype}{Colors.RESET}")

        vertices_3d = pred_world_points.reshape(-1, 3)
        colors_rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)
        conf = pred_world_points_conf.reshape(-1)
        
        print(f"{Colors.MAGENTA}   Total points before filter: {len(vertices_3d)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}   Conf range: [{conf.min():.4f}, {conf.max():.4f}]{Colors.RESET}")

        conf_threshold = conf_thres / 100
        mask = (conf >= conf_threshold) & (conf > 1e-5)
        
        print(f"{Colors.MAGENTA}   Filtered points: {mask.sum()} / {len(mask)} ({100*mask.sum()/len(mask):.1f}%){Colors.RESET}")

        vertices_3d = vertices_3d[mask]
        colors_rgb = colors_rgb[mask]
        log_time("Process and filter point data", process_start)

        # Create Open3D point cloud
        pcd_start = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
        log_time("Create Open3D point cloud", pcd_start)

        # Save as PLY
        save_start = time.time()
        # pcd = simplify_cloud(pcd)
        o3d.io.write_point_cloud(pty_location, pcd)
        log_time("Write PLY file", save_start)
        
        commit_start = time.time()
        volume.commit()
        log_time("Volume commit", commit_start)
        
        print(f"{Colors.GREEN}✅ [process_infered_data] SUCCESS - saved {len(vertices_3d)} points to {pty_location}{Colors.RESET}")
        log_time("process_infered_data TOTAL", func_start)
        return pty_location

    def process_infered_data_per_image(inf_data_path, id, conf_thres=50):
        """
        Loads inference data and writes one PLY file per image.
        Handles both batched [1, N, ...] and unbatched [N, ...] formats.
        Prints debug info to understand shapes and counts.
        """
        func_start = time.time()
        print(f"{Colors.CYAN}📦 [process_infered_data_per_image] ENTER - id={id}, conf_thres={conf_thres}{Colors.RESET}")
        
        base_folder = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / str(id) / "models"
        base_folder.mkdir(parents=True, exist_ok=True)
        print(f"{Colors.MAGENTA}   Output folder: {base_folder}{Colors.RESET}")

        load_start = time.time()
        predictions = torch.load(inf_data_path, map_location="cpu")
        log_time("Load predictions.pt", load_start)

        # --- Keys and basic info ---
        inspect_start = time.time()
        print(f"{Colors.MAGENTA}   Predictions keys: {list(predictions.keys())}{Colors.RESET}")
        pred_world_points = predictions["points"]
        print(f"{Colors.MAGENTA}   Points dtype: {pred_world_points.dtype}, shape: {pred_world_points.shape}{Colors.RESET}")

        if "conf" in predictions:
            print(f"{Colors.MAGENTA}   Conf shape: {predictions['conf'].shape}{Colors.RESET}")
        else:
            print(f"{Colors.RED}⚠️  No confidence array found, using default ones{Colors.RESET}")

        images = predictions["images"]
        print(f"{Colors.MAGENTA}   Images shape: {images.shape}, range: [{images.min().item():.3f}, {images.max().item():.3f}]{Colors.RESET}")
        log_time("Inspect prediction data", inspect_start)

        # --- Convert to numpy ---
        convert_start = time.time()
        pred_world_points = pred_world_points.numpy()
        pred_world_points_conf = predictions.get("conf", torch.ones_like(predictions["points"])).numpy()
        images = images.numpy()
        log_time("Convert to numpy", convert_start)

        # --- Remove batch dimension if present ---
        if pred_world_points.shape[0] == 1 and images.shape[0] == 1:
            print(f"{Colors.MAGENTA}   Removing leading batch dimension (shape [1, N, ...]){Colors.RESET}")
            pred_world_points = pred_world_points[0]
            pred_world_points_conf = pred_world_points_conf[0]
            images = images[0]

        # --- Determine number of images ---
        num_images = pred_world_points.shape[0]
        conf_threshold = conf_thres / 100.0
        print(f"{Colors.MAGENTA}   Detected {num_images} image(s) to process{Colors.RESET}")

        ply_paths = []
        loop_start = time.time()

        for i in range(num_images):
            img_start = time.time()
            print(f"{Colors.CYAN}   --- Processing image {i}/{num_images-1} ---{Colors.RESET}")
            
            vertices_3d = pred_world_points[i].reshape(-1, 3)
            colors_rgb = (images[i].reshape(-1, 3) * 255).astype(np.uint8)
            conf = pred_world_points_conf[i].reshape(-1)

            print(f"{Colors.MAGENTA}      vertices_3d: {vertices_3d.shape}, colors_rgb: {colors_rgb.shape}, conf: {conf.shape}{Colors.RESET}")
            print(f"{Colors.MAGENTA}      conf range: [{conf.min():.4f}, {conf.max():.4f}]{Colors.RESET}")

            mask = (conf >= conf_threshold) & (conf > 1e-5)
            kept_points = mask.sum()
            total_points = len(mask)
            print(f"{Colors.MAGENTA}      mask kept {kept_points} / {total_points} points ({100*kept_points/total_points:.1f}%){Colors.RESET}")

            vertices_3d = vertices_3d[mask]
            colors_rgb = colors_rgb[mask]

            if len(vertices_3d) == 0:
                print(f"{Colors.RED}      ⚠️  WARNING: no points kept after filtering, skipping{Colors.RESET}")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices_3d)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

            ply_path = base_folder / f"{i:04d}.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"{Colors.GREEN}      ✅ Saved {ply_path.name}{Colors.RESET}")
            log_time(f"Process image {i}", img_start)
            ply_paths.append(ply_path)

        log_time("Process all images loop", loop_start)
        print(f"{Colors.GREEN}✅ [process_infered_data_per_image] SUCCESS - saved {len(ply_paths)} PLY files to {base_folder}{Colors.RESET}")
        log_time("process_infered_data_per_image TOTAL", func_start)
        return [str(p) for p in ply_paths]


    @web_app.post("/pointcloud/{id}")
    async def new_image(id: str, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}🚀 [POST /pointcloud/{id}] ENDPOINT CALLED{Colors.RESET}")
        print(f"{Colors.MAGENTA}   File: {file.filename}, Type: {file.content_type}{Colors.RESET}")
        print(f"{'='*80}\n")
        
        # STEP 1: Upload and preprocess image
        step1_start = time.time()
        uploaded = await upload_image(id, file)
        log_time("STEP 1: Upload Image", step1_start)
        print(f"{Colors.GREEN}✅ Image uploaded successfully to {uploaded}{Colors.RESET}\n")
        
        # STEP 2: Run inference
        step2_start = time.time()
        inference_path = await run_inference(id)
        log_time("STEP 2: Run Inference", step2_start)
        print(f"{Colors.GREEN}✅ Inference results at {inference_path}{Colors.RESET}\n")
        
        # STEP 3: Save predictions and prepare volume
        step3_start = time.time()
        standard_predictions_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / id / "predictions.pt"
        standard_predictions_path.parent.mkdir(parents=True, exist_ok=True)

        # refresh volume with latest changes
        vol_reload_start = time.time()
        volume.reload()
        log_time("Volume reload", vol_reload_start)

        import shutil
        copy_start = time.time()
        shutil.copy2(inference_path, str(standard_predictions_path))
        log_time("Copy predictions.pt", copy_start)
        print(f"{Colors.GREEN}✅ Saved predictions to {standard_predictions_path}{Colors.RESET}")
        log_time("STEP 3: Save Predictions", step3_start)

        # STEP 4: Extract last image's pointcloud and camera pose for immediate response
        step4_start = time.time()
        temp_ply_path, camera_pose = extract_last_image_pointcloud(str(inference_path), id)
        log_time("STEP 4: Extract Last Image PLY", step4_start)
        
        if temp_ply_path is None:
            print(f"{Colors.RED}❌ ERROR: Failed to extract pointcloud from last image{Colors.RESET}")
            raise HTTPException(status_code=500, detail="Failed to extract pointcloud from last image")
        
        # STEP 5: Schedule full reconstruction in the background (non-blocking)
        step5_start = time.time()
        if background_tasks:
            background_tasks.add_task(process_infered_data, str(inference_path), id, 0)
            background_tasks.add_task(process_infered_data_per_image, str(inference_path), id, 0)
            print(f"{Colors.CYAN}🔄 Scheduled background tasks for full reconstruction{Colors.RESET}")
        else:
            print(f"{Colors.RED}⚠️  WARNING: No background_tasks available, skipping background processing{Colors.RESET}")
        log_time("STEP 5: Schedule Background Tasks", step5_start)
        
        # STEP 6: Read PLY and prepare response
        step6_start = time.time()
        read_start = time.time()
        with open(temp_ply_path, 'rb') as f:
            ply_data = f.read()
        ply_size = len(ply_data)
        print(f"{Colors.MAGENTA}   Read PLY file: {ply_size / 1024:.2f} KB{Colors.RESET}")
        log_time("Read PLY file", read_start)
        
        # Build multipart response with pointcloud and camera pose
        encode_start = time.time()
        multipart_data = MultipartEncoder(
            fields={
                'pointcloud': (f"{id}_latest.ply", ply_data, 'application/octet-stream'),
                'camera_pose': json.dumps(camera_pose) if camera_pose is not None else json.dumps(None)
            }
        )
        log_time("Encode multipart response", encode_start)
        log_time("STEP 6: Prepare Response", step6_start)
        
        print(f"\n{'='*80}")
        print(f"{Colors.GREEN}✅ [POST /pointcloud/{id}] ENDPOINT COMPLETE{Colors.RESET}")
        log_time("🎯 TOTAL ENDPOINT TIME", endpoint_start)
        print(f"{'='*80}\n")
        
        return Response(
            content=multipart_data.to_string(),
            media_type=multipart_data.content_type
        )

    @web_app.get("/pointcloud/{pc_id}/{tag}")
    async def get_pointcloud(pc_id: str, tag: str):
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}📥 [GET /pointcloud/{pc_id}/{tag}] ENDPOINT CALLED{Colors.RESET}")
        print(f"{'='*80}\n")
        
        # dynamically point to latest PLY for the given project
        ply_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / pc_id / "models" / "latest.ply"
        print(f"{Colors.MAGENTA}   Looking for PLY at: {ply_path}{Colors.RESET}")

        if not ply_path.exists():
            print(f"{Colors.RED}❌ ERROR: Point cloud file not found at {ply_path}{Colors.RESET}")
            return {"error": "Point cloud file not found."}

        # Load LAST camera pose from predictions file
        predictions_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / pc_id / "predictions.pt"
        camera_pose = None
        print(f"{Colors.MAGENTA}   Looking for predictions at: {predictions_path}{Colors.RESET}")
        
        if predictions_path.exists():
            load_start = time.time()
            try:
                predictions = torch.load(str(predictions_path), map_location="cpu")
                camera_poses_tensor = predictions.get("camera_poses")
                
                if camera_poses_tensor is not None:
                    # Extract only the LAST camera pose [batch=0, image=-1]
                    # Handle batch dimension [1, N, ...] format
                    if camera_poses_tensor.dim() >= 2 and camera_poses_tensor.shape[0] == 1:
                        camera_pose = camera_poses_tensor[0][-1].numpy().tolist()
                    else:
                        camera_pose = camera_poses_tensor[-1].numpy().tolist()
                    print(f"{Colors.GREEN}✅ Loaded last camera pose (shape: {camera_poses_tensor.shape}){Colors.RESET}")
                else:
                    print(f"{Colors.RED}⚠️  WARNING: No camera_poses found in predictions for {pc_id}{Colors.RESET}")
                log_time("Load camera pose from predictions", load_start)
            except Exception as e:
                print(f"{Colors.RED}⚠️  WARNING: Failed to load camera pose: {e}{Colors.RESET}")
        else:
            print(f"{Colors.RED}⚠️  WARNING: predictions.pt not found at {predictions_path}{Colors.RESET}")

        # Always serve the latest file on disk
        read_start = time.time()
        with open(ply_path, 'rb') as f:
            ply_data = f.read()
        ply_size = len(ply_data)
        print(f"{Colors.MAGENTA}   Read PLY file: {ply_size / 1024:.2f} KB{Colors.RESET}")
        log_time("Read PLY file", read_start)
        
        encode_start = time.time()
        multipart_data = MultipartEncoder(
            fields={
                'pointcloud': (f"{pc_id}_{tag}.ply", ply_data, 'application/octet-stream'),
                'camera_pose': json.dumps(camera_pose) if camera_pose is not None else json.dumps(None)
            }
        )
        log_time("Encode multipart response", encode_start)
        
        print(f"\n{'='*80}")
        print(f"{Colors.GREEN}✅ [GET /pointcloud/{pc_id}/{tag}] ENDPOINT COMPLETE{Colors.RESET}")
        log_time("🎯 TOTAL ENDPOINT TIME", endpoint_start)
        print(f"{'='*80}\n")
        
        return Response(
            content=multipart_data.to_string(),
            media_type=multipart_data.content_type
        )

    return web_app