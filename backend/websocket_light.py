import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time
import asyncio
import os

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
    max_containers=1
)
def process_queue(project_id: str):
    import shutil
    
    # Instantiate the model inference class from the other app
    try:
        Pi3Remote = modal.Cls.from_name("pi3-inference", "ModelInference")
        inference_obj = Pi3Remote()
    except Exception as e:
        print(f"Error connecting to inference model: {e}")
        return

    print(f"Worker started for project {project_id}")
    
    # Drain queue logic
    try:
        # Non-blocking drain of the partition
        items = reconstruction_queue.get_many(partition=project_id, n_values=100, block=False)
    except queue.Empty:
        # Should generally not happen if spawned by REST API, but good for robustness
        print(f"No items found for project {project_id}. Exiting.")
        return

    if not items:
        print(f"Empty batch for {project_id}. Exiting.")
        return

    # Deduplicate? No, we just use the project_id. The fact that we have items is the trigger.
    # We just need to process the project ONCE.
    
    print(f"{Colors.CYAN}Processing batch of {len(items)} items for Project {project_id}{Colors.RESET}")
    
    # LOGGING: Extract and print image IDs in this batch
    batch_image_ids = [item.get("image_id", "unknown") for item in items]
    print(f"{Colors.MAGENTA}Batch contains {len(batch_image_ids)} images: {batch_image_ids}{Colors.RESET}")
    
    # Use the LAST item's image_id as the "latest" identifier for status updates
    latest_item = items[-1]
    image_id = latest_item["image_id"]
    
    # --- LOCK CHECK ---
    try:
        current_state = reconstruction_state.get(project_id)
    except KeyError:
        current_state = None
    
    if current_state and current_state.get("status") == "processing":
        print(f"Project {project_id} is currently busy. Re-queueing {len(items)} items.")
        # Re-queue the items to the back of the partition
        reconstruction_queue.put_many(items, partition=project_id)
        time.sleep(1)
        return

    process_start_total = time.time()
    
    # Acquire Lock
    lock_start = time.time()
    reconstruction_state[project_id] = {
        "latest_image_id": image_id, 
        "timestamp": time.time(),
        "status": "processing",
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
        
        # 3. Process Inference Data (create PLY)
        ply_proc_start = time.time()
        result_ply = process_infered_data(str(inference_path), project_id)
        log_time("Process Inferred Data (PLY)", ply_proc_start)
        
        # Commit volume changes
        commit_ply_start = time.time()
        volume.commit()
        log_time("Volume Commit (PLY)", commit_ply_start)
        
        images_folder = Path(VOL_MOUNT_PATH) / "backend_data" / "reconstructions" / project_id / "images"
        check_mtime_start = time.time()
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
            "last_processed_timestamp": new_folder_mtime,
            "project_id": project_id # Add project_id to event so consumer knows who to notify
        }
        # CRITICAL CHANGE: Use DEFAULT partition for notifications so central consumer can drain all
        notification_queue.put(event)
        
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
            "error": str(e),
            "project_id": project_id
        }
        # CRITICAL CHANGE: Use DEFAULT partition
        notification_queue.put(event)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, project_id: str, user_id: str, websocket: WebSocket):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append((user_id, websocket))
        print(f"Client connected to {project_id}. Total: {len(self.active_connections[project_id])}")
        print("Connections: ", self.active_connections)

    def disconnect(self, project_id: str, user_id: str, websocket: WebSocket):
        if project_id in self.active_connections:
            if (user_id, websocket) in self.active_connections[project_id]:
                self.active_connections[project_id].remove((user_id, websocket))
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
        print(f"Client disconnected from {project_id}")

    async def broadcast(self, project_id: str, message: dict):
        if project_id in self.active_connections:
            disconnected = []
            # gather all and send to all
            messages = []
            for user_id, connection in self.active_connections[project_id]:
                messages.append(connection.send_json(message))
            await asyncio.gather(*messages)

manager = ConnectionManager()

@app.function(
    image=image, 
    volumes={VOL_MOUNT_PATH: volume}, 
    max_containers=1, # MUST be 1 to allow centralized ConnectionManager
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

    @web_app.on_event("startup")
    async def startup_event():
        print("Starting notification consumer...")
        # Store task in app state to cancel it later
        web_app.state.consume_task = asyncio.create_task(consume_notifications())

    @web_app.on_event("shutdown")
    async def shutdown_event():
        print("Shutting down notification consumer...")
        if hasattr(web_app.state, "consume_task") and web_app.state.consume_task:
            web_app.state.consume_task.cancel()
            try:
                await web_app.state.consume_task
            except asyncio.CancelledError:
                print("Notification consumer cancelled.")
            except Exception as e:
                print(f"Error during consumer shutdown: {e}")

    async def consume_notifications():
        print("Notification consumer running...")
        try:
            while True:
                try:
                    # Use get_many with blocking to wait efficiently
                    # Draining default partition
                    items = await notification_queue.get_many.aio(block=True, timeout=5.0, n_values=100)
                    
                    if items:
                        print(f"Consumer: Received {len(items)} notifications")
                        for item in items:
                            pid = item.get("project_id")
                            if pid:
                                await manager.broadcast(pid, item)
                            else:
                                print(f"Warning: Notification missing project_id: {item}")

                except queue.Empty:
                    # Timeout reached, loop again
                    pass
                except Exception as e:
                    if "ClientClosed" in str(e):
                        print("Modal Client Closed. Exiting consumer.")
                        break
                    print(f"Error in notification consumer: {e}")
                    await asyncio.sleep(1)
                
                # Tiny sleep not strictly needed with blocking get, but safe
                # await asyncio.sleep(0.01) 
        except asyncio.CancelledError:
            print("Consumer loop cancelled via CancelledError.")
            raise

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.websocket("/ws/{project_id}")
    async def websocket_endpoint(websocket: WebSocket, project_id: str):
        import uuid
        user_id = uuid.uuid4()
        await manager.connect(project_id, websocket, user_id)
        print(f"Connected to {project_id} with UUID {user_id}")
        # Initial Sync
        # try:
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
        # except Exception:
        #     pass

        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            manager.disconnect(project_id, websocket)
        except Exception as e:
            print(f"Error in websocket: {e}")
            manager.disconnect(project_id, websocket)

    return web_app
