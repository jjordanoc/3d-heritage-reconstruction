import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time
import asyncio
import os
import uuid
import base64
import json
import queue  # Import standard queue for Empty exception
from PIL import Image
import io

# Define constants
VOL_MOUNT_PATH = Path("/mnt/volume")

# ANSI Color codes for logging
class Colors:
    CYAN = '\033[96m'      # Function entry/exit
    YELLOW = '\033[93m'    # Timing info
    RED = '\033[91m'       # Warnings/Errors
    GREEN = '\033[92m'     # Success
    MAGENTA = '\033[95m'   # Data info
    RESET = '\033[0m'      # Reset color

def log_time(label, start_time):
    """Helper to log elapsed time in yellow without emojis"""
    elapsed = time.time() - start_time
    print(f"{Colors.YELLOW}[TIME] [{label}] took {elapsed:.3f}s{Colors.RESET}")
    return elapsed

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
# Setup Volume
volume = modal.Volume.from_name(name="ut3c-heritage", create_if_missing=True)

# Setup App
app = modal.App("websocket-light", image=image)

# Setup Queue and Dict
reconstruction_queue = modal.Queue.from_name("reconstruction-queue", create_if_missing=True)
notification_queue = modal.Queue.from_name("notification-queue", create_if_missing=True)
reconstruction_state = modal.Dict.from_name("reconstruction-state", create_if_missing=True)

# Helper to process image (resize)
def preprocess_image(image_bytes: bytes, target_size: int = 512) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    return img

# Helper to extract last pointcloud from predictions
def extract_last_image_pointcloud(inf_data_path, id, conf_thres=50):
    """
    Extracts the pointcloud and camera pose for the LAST (most recent) image.
    Returns: tuple (last_pc_ply_path, camera_pose)
    """
    import torch
    import numpy as np
    import open3d as o3d
    
    print(f"Extracting pointcloud from {inf_data_path}")
    
    predictions = torch.load(inf_data_path, map_location="cpu")
    
    # Extract camera pose for the last image
    camera_poses = predictions.get("camera_poses")
    camera_pose = None
    if camera_poses is not None:
        if camera_poses.dim() >= 2:
            camera_pose = camera_poses[0][-1].numpy().tolist()
        else:
            camera_pose = camera_poses[-1].numpy().tolist()
    
    # Extract pointcloud data for last image
    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", torch.ones_like(predictions["points"]))
    images = predictions["images"]
    
    # Get last image data [batch=0, image=-1]
    last_points = pred_world_points[0][-1].numpy()  # [H*W, 3]
    last_conf = pred_world_points_conf[0][-1].numpy()  # [H*W]
    last_colors = images[0][-1].numpy()  # [H*W, 3]
    
    vertices_3d = last_points.reshape(-1, 3)
    colors_rgb = (last_colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf = last_conf.reshape(-1)
    
    conf_threshold = conf_thres / 100.0
    mask = (conf >= conf_threshold) & (conf > 1e-5)
    
    vertices_3d = vertices_3d[mask]
    colors_rgb = colors_rgb[mask]
    
    if len(vertices_3d) == 0:
        print("ERROR: No points kept after filtering!")
        return None, camera_pose
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
    
    # Save temporary PLY
    temp_folder = VOL_MOUNT_PATH / "backend_data" / "reconstructions" / str(id) / "models"
    temp_folder.mkdir(parents=True, exist_ok=True)
    temp_ply_path = temp_folder / "latest.ply" # Saving as latest.ply directly
    
    o3d.io.write_point_cloud(str(temp_ply_path), pcd)
    print(f"Saved pointcloud to {temp_ply_path}")
    
    return str(temp_ply_path), camera_pose

def process_infered_data(inf_data_path, id, conf_thres=50):
    """
    Process full reconstruction (all images combined)
    """
    import torch
    import numpy as np
    import open3d as o3d
    
    print(f"Processing full inference data for {id}")
    
    pty_location = str(VOL_MOUNT_PATH) + f"/backend_data/reconstructions/{id}/models/latest.ply"
    pty_folder = str(VOL_MOUNT_PATH) + f"/backend_data/reconstructions/{id}/models"
    Path(pty_folder).mkdir(parents=True, exist_ok=True)
    
    predictions = torch.load(inf_data_path, weights_only=False, map_location=torch.device('cpu'))

    pred_world_points = predictions["points"].numpy()
    pred_world_points_conf = predictions.get(
        "conf",
        torch.ones_like(predictions["points"])
    ).numpy()
    images = predictions["images"].numpy()
    
    vertices_3d = pred_world_points.reshape(-1, 3)
    colors_rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)
    conf = pred_world_points_conf.reshape(-1)
    
    conf_threshold = conf_thres / 100
    mask = (conf >= conf_threshold) & (conf > 1e-5)
    
    vertices_3d = vertices_3d[mask]
    colors_rgb = colors_rgb[mask]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

    # Save as PLY
    o3d.io.write_point_cloud(pty_location, pcd)
    print(f"Saved full pointcloud to {pty_location}")
    return pty_location

# Worker function
@app.function(
    image=image, 
    volumes={VOL_MOUNT_PATH: volume}, 
    timeout=24*60*60, # 24 hours
    max_containers=10
)
def process_queue():
    import shutil
    
    # Instantiate the model inference class from the other app
    try:
        Pi3Remote = modal.Cls.from_name("pi3-inference", "ModelInference")
        inference_obj = Pi3Remote()
    except Exception as e:
        print(f"Error connecting to inference model: {e}")
        return

    print("Worker started, waiting for tasks...")
    
    # Process items until queue is empty for a short duration
    # This allows the container to stay warm for bursts but scale down when idle
    while True:
        try:
            # Fetch item with timeout (e.g., 10 seconds)
            # If nothing arrives within timeout, assume we can exit (scale to zero)
            item = reconstruction_queue.get(block=True, timeout=10) 
            
            project_id = item["project_id"]
            image_id = item["image_id"]
            
            # --- LOCK CHECK ---
            try:
                current_state = reconstruction_state.get(project_id)
            except KeyError:
                current_state = None
            
            if current_state and current_state.get("status") == "processing":
                print(f"Project {project_id} is busy. Re-queueing {image_id}")
                reconstruction_queue.put(item) 
                time.sleep(1)
                continue
            # unnecessary overhead, this would have to be implemented differently just using the queue order
            # all items in the queue should be processed at once
            # --- OPTIMIZATION: Folder Timestamp Check ---
            # try:
            #     project_root = VOL_MOUNT_PATH / "backend_data" / "reconstructions" / project_id
            #     images_folder = project_root / "images"
            #     outputs_folder = project_root / "models"
                
            #     volume.reload()
                
            #     if images_folder.exists():
            #         folder_mtime = images_folder.stat().st_mtime
                    
            #         # Check if we can skip
            #         last_processed_ts = current_state.get("last_processed_timestamp", 0) if current_state else 0
                    
            #         if folder_mtime <= last_processed_ts:
            #             print(f"Skipping inference for {image_id}: Folder unchanged (mtime {folder_mtime} <= last {last_processed_ts})")
                        
            #             ts = time.time()
            #             reconstruction_state[project_id] = {
            #                 "latest_image_id": image_id,
            #                 "timestamp": ts,
            #                 "status": "updated",
            #                 "last_processed_timestamp": last_processed_ts
            #             }
                        
            #             # Notify
            #             event = {
            #                 "type": "update",
            #                 "image_id": image_id,
            #                 "timestamp": ts,
            #                 "status": "updated",
            #                 "last_processed_timestamp": last_processed_ts
            #             }
            #             notification_queue.put(event, partition=project_id)
            #             continue
            # except Exception as e:
            #     print(f"Timestamp check failed: {e}. Proceeding with inference.")

            print(f"{Colors.CYAN}Processing: Project {project_id}, Image {image_id}{Colors.RESET}")
            process_start_total = time.time()
            
            # Acquire Lock
            lock_start = time.time()
            reconstruction_state[project_id] = {
                "latest_image_id": image_id, 
                "timestamp": time.time(),
                "status": "processing",
                # Keep old timestamp until finished
                "last_processed_timestamp": current_state.get("last_processed_timestamp", 0) if current_state else 0
            }
            log_time("Acquire Lock", lock_start)
            
            images_path_relative = f"/backend_data/reconstructions/{project_id}/images" # Path relative to volume root for inference

            try:
                # 1. Run Inference
                print(f"{Colors.CYAN}Starting remote inference for {project_id}...{Colors.RESET}")
                
                inf_start = time.time()
                inf_res_path = inference_obj.run_inference.remote(images_path_relative)
                log_time("Remote Inference", inf_start)
                
                inference_path = str(VOL_MOUNT_PATH) + inf_res_path + "/predictions.pt"
                print(f"{Colors.GREEN}Inference complete. Results at {inference_path}{Colors.RESET}")

                # 2. Save predictions.pt to project folder
                vol_sync_start = time.time()
                volume.reload()
                log_time("Volume Reload", vol_sync_start)
                
                # DEBUG: Check if file exists
                if not os.path.exists(inference_path):
                    print(f"{Colors.RED}DEBUG: File MISSING at {inference_path}{Colors.RESET}")
                    parent_dir = Path(inference_path).parent.parent
                    if parent_dir.exists():
                        print(f"DEBUG: Contents of {parent_dir}: {os.listdir(parent_dir)}")
                        target_uuid = Path(inference_path).parent.name
                        print(f"DEBUG: Looking for UUID folder: {target_uuid}")
                    else:
                        print(f"DEBUG: Parent dir {parent_dir} does not exist!")
                else:
                    print(f"DEBUG: File FOUND at {inference_path}")
                
                # unnecessary overhead, use the inference path directly
                # copy_start = time.time()
                # standard_predictions_path = project_root / "predictions.pt"
                # shutil.copy2(inference_path, str(standard_predictions_path))
                # log_time("Copy predictions.pt", copy_start)
                
                # commit_start = time.time()
                # volume.commit()
                # log_time("Volume Commit (predictions.pt)", commit_start)
                
                # 3. Process Inference Data (create PLY)
                ply_proc_start = time.time()
                result_ply = process_infered_data(str(inference_path), project_id)
                log_time("Process Inferred Data (PLY)", ply_proc_start)
                
                # Commit volume changes
                commit_ply_start = time.time()
                volume.commit()
                log_time("Volume Commit (PLY)", commit_ply_start)
                
                check_mtime_start = time.time()
                # adds unnecessary overhead, changes inside the current container are already reflected 
                # volume.reload()
                if images_folder.exists():
                    new_folder_mtime = images_folder.stat().st_mtime
                else:
                    new_folder_mtime = time.time()
                log_time("Check Final mtime", check_mtime_start)

                # Update state (Unlock)
                unlock_start = time.time()
                ts = time.time()
                reconstruction_state[project_id] = {
                    "latest_image_id": image_id,
                    "timestamp": ts,
                    "status": "updated",
                    "last_processed_timestamp": new_folder_mtime
                }
                log_time("Unlock State", unlock_start)
                
                # Notify
                event = {
                    "type": "update",
                    "image_id": image_id,
                    "timestamp": ts,
                    "status": "updated",
                    "last_processed_timestamp": new_folder_mtime
                }
                notification_queue.put(event, partition=project_id)
                
                print(f"{Colors.GREEN}State updated for {project_id}{Colors.RESET}")
                log_time("Total Process Queue Item", process_start_total)
                
            except Exception as e:
                print(f"Error in pipeline execution: {e}")
                import traceback
                traceback.print_exc()
                
                # Release lock on error
                ts = time.time()
                reconstruction_state[project_id] = {
                    "latest_image_id": image_id,
                    "timestamp": ts,
                    "status": "error",
                    "error": str(e)
                }
                
                # Notify Error
                event = {
                    "type": "update",
                    "image_id": image_id,
                    "timestamp": ts,
                    "status": "error",
                    "error": str(e)
                }
                notification_queue.put(event, partition=project_id)
        
        except queue.Empty:
            print("Queue empty for timeout duration. Exiting worker.")
            break
        except Exception as e:
            print(f"Error in worker loop: {e}")
            time.sleep(1)

@app.function(
    image=image, 
    volumes={VOL_MOUNT_PATH: volume}, 
    max_containers=100,
    timeout=24*60*60 # 24 hours
)
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI()
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.websocket("/ws/{project_id}")
    async def websocket_endpoint(websocket: WebSocket, project_id: str):
        await websocket.accept()
        print(f"Client connected to {project_id}")
        
        # 1. Initial Sync (Snapshot)
        # try:
        #     # Use .aio to avoid blocking
        #     state = await reconstruction_state.get.aio(project_id)
        #     if state:
        #         msg = {
        #             "type": "update",
        #             "image_id": state.get("latest_image_id"),
        #             "timestamp": state.get("timestamp"),
        #             "status": state.get("status")
        #         }
        #         if state.get("status") == "error":
        #             msg["error"] = state.get("error")
        #         await websocket.send_json(msg)
        # except KeyError:
        #     pass
        # except Exception as e:
        #     print(f"Error fetching initial state: {e}")

        # 2. Concurrency: Listen to Client (Ping) AND Stream Notifications
        async def notification_generator():
            while True:
                # Iterate through notifications for this project
                # item_poll_timeout allows it to wait for new items
                try:
                    async for event in notification_queue.iterate(partition=project_id, item_poll_timeout=10.0):
                        try:
                            await websocket.send_json(event)
                        except Exception as e:
                            print(f"Error sending notification: {e}")
                            return # Stop generator if socket is dead
                except Exception as e:
                    print(f"Notification generator error: {e}")
                    await asyncio.sleep(1)

        async def client_listener():
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                print(f"Client disconnected from {project_id}")
            except Exception as e:
                print(f"Error in client listener: {e}")

        # Run both tasks
        try:
            notify_task = asyncio.create_task(notification_generator())
            listener_task = asyncio.create_task(client_listener())
            
            done, pending = await asyncio.wait(
                [notify_task, listener_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in pending:
                task.cancel()
                
        except Exception as e:
            print(f"WebSocket handler error: {e}")

    return web_app

@app.local_entrypoint()
def main():
    print("Spawning worker process...")
    process_queue.spawn()
    print("To run the server: modal serve backend/websocket_light.py")
