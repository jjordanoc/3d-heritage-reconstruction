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
from PIL import Image
import io
import torch
import numpy as np
import open3d as o3d
import shutil

# Define constants
VOL_MOUNT_PATH = Path("/mnt/volume")

# Setup Modal Image
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "git",
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
    "requests-toolbelt"
])

# Setup Volume
volume = modal.Volume.from_name(name="ut3c-heritage", create_if_missing=True)

# Setup App
app = modal.App("websocket-light", image=image)

# Setup Queue and Dict
reconstruction_queue = modal.Queue.from_name("reconstruction-queue", create_if_missing=True)
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
    timeout=1200,
    concurrency_limit=10
)
def process_queue():
    # Instantiate the model inference class from the other app
    try:
        Pi3Remote = modal.Cls.from_name("pi3-inference", "ModelInference")
        inference_obj = Pi3Remote()
    except Exception as e:
        print(f"Error connecting to inference model: {e}")
        return

    print("Worker started, waiting for tasks...")
    while True:
        try:
            # Fetch item (blocking)
            item = reconstruction_queue.get() 
            
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
            
            print(f"Processing: Project {project_id}, Image {image_id}")
            
            # Acquire Lock
            reconstruction_state[project_id] = {
                "latest_image_id": image_id, 
                "timestamp": time.time(),
                "status": "processing"
            }
            
            # Construct paths
            project_root = VOL_MOUNT_PATH / "backend_data" / "reconstructions" / project_id
            images_folder = project_root / "images"
            outputs_folder = project_root / "models"
            images_path_relative = f"/backend_data/reconstructions/{project_id}/images" # Path relative to volume root for inference

            # Reload volume to ensure we have latest files
            volume.reload()

            try:
                # 1. Run Inference
                print(f"Starting remote inference for {project_id}...")
                inf_res_path = inference_obj.run_inference.remote(images_path_relative)
                inference_path = str(VOL_MOUNT_PATH) + inf_res_path + "/predictions.pt"
                print(f"Inference complete. Results at {inference_path}")

                # 2. Save predictions.pt to project folder
                standard_predictions_path = project_root / "predictions.pt"
                shutil.copy2(inference_path, str(standard_predictions_path))
                
                # 3. Process Inference Data (create PLY)
                # We can choose to extract just the last image or process the full cloud.
                # The original code in modal_server.py does both or either depending on flow.
                # Here we want to update the "latest.ply" which the frontend reads.
                # The simpler approach for now is to overwrite latest.ply with the FULL reconstruction 
                # (or just the new points if we were doing incremental, but user said "forward pass")
                # Based on user request "run inference in the same way... we're not really using registration pipeline anymore"
                # We will generate the full cloud from the predictions.
                
                # Use process_infered_data to save the full PLY to models/latest.ply
                result_ply = process_infered_data(str(standard_predictions_path), project_id)
                
                # Commit volume changes
                volume.commit()
                
                # Update state (Unlock)
                reconstruction_state[project_id] = {
                    "latest_image_id": image_id,
                    "timestamp": time.time(),
                    "status": "updated"
                }
                print(f"State updated for {project_id}")
                
            except Exception as e:
                print(f"Error in pipeline execution: {e}")
                import traceback
                traceback.print_exc()
                
                # Release lock on error
                reconstruction_state[project_id] = {
                    "latest_image_id": image_id,
                    "timestamp": time.time(),
                    "status": "error",
                    "error": str(e)
                }
        
        except Exception as e:
            print(f"Error in worker loop: {e}")
            time.sleep(1)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, project_id: str, websocket: WebSocket):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)
        print(f"Client connected to {project_id}. Total: {len(self.active_connections[project_id])}")

    def disconnect(self, project_id: str, websocket: WebSocket):
        if project_id in self.active_connections:
            if websocket in self.active_connections[project_id]:
                self.active_connections[project_id].remove(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
        print(f"Client disconnected from {project_id}")

    async def broadcast(self, project_id: str, message: dict):
        if project_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[project_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            for connection in disconnected:
                self.disconnect(project_id, connection)

manager = ConnectionManager()

def handle_image_upload_sync(project_id: str, image_bytes: bytes):
    """
    Synchronous helper to handle image processing, saving, and queuing.
    """
    # Preprocess (CPU bound)
    img = preprocess_image(image_bytes)
    
    # Save to Volume (I/O bound + Network)
    images_dir = VOL_MOUNT_PATH / "backend_data" / "reconstructions" / project_id / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_uuid = str(uuid.uuid4())
    image_filename = f"{image_uuid}.png"
    save_path = images_dir / image_filename
    
    img.save(save_path, format="PNG")
    volume.commit() # BLOCKING
    
    # Queue (Network bound)
    reconstruction_queue.put({"project_id": project_id, "image_id": image_uuid}) # BLOCKING
    
    return image_uuid

@app.function(
    image=image, 
    volumes={VOL_MOUNT_PATH: volume}, 
    allow_concurrent_inputs=100,
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
        try:
            print("Starting state monitor...")
            asyncio.create_task(monitor_state())
        except Exception as e:
            print(f"Startup error: {e}")

    async def monitor_state():
        last_timestamps = {}
        print("State monitor running")
        while True:
            try:
                # Iterate active projects
                active_projects = list(manager.active_connections.keys())
                
                for project_id in active_projects:
                    try:
                        # Use to_thread to avoid blocking on network call
                        state = await asyncio.to_thread(reconstruction_state.get, project_id)
                    except KeyError:
                        state = None
                    except Exception as e:
                        print(f"Error fetching state: {e}")
                        state = None
                        
                    if state:
                        ts = state.get("timestamp", 0)
                        last_ts = last_timestamps.get(project_id, 0)
                        
                        if ts > last_ts:
                            print(f"Detected update for {project_id}: {state}")
                            last_timestamps[project_id] = ts
                            
                            msg = {
                                "type": "update",
                                "image_id": state.get("latest_image_id"),
                                "timestamp": ts,
                                "status": state.get("status")
                            }
                            if state.get("status") == "error":
                                msg["error"] = state.get("error")
                                
                            await manager.broadcast(project_id, msg)
            except Exception as e:
                print(f"Error in monitor_state: {e}")
            
            await asyncio.sleep(1)

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.websocket("/ws/{project_id}")
    async def websocket_endpoint(websocket: WebSocket, project_id: str):
        await manager.connect(project_id, websocket)
        try:
            while True:
                data = await websocket.receive_json()
                
                if data.get("type") == "upload_image":
                    image_data = data.get("image_data")
                    if image_data:
                        try:
                            # Decode (CPU bound, usually fast enough, but good to offload if large)
                            image_bytes = base64.b64decode(image_data)
                            
                            # Run blocking operations in a thread
                            image_uuid = await asyncio.to_thread(
                                handle_image_upload_sync, 
                                project_id, 
                                image_bytes
                            )
                            
                            await websocket.send_json({
                                "type": "upload_acknowledged",
                                "image_id": image_uuid,
                                "message": "Image queued for processing"
                            })
                        except Exception as e:
                            print(f"Error processing upload: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Upload failed: {str(e)}"
                            })
                        
        except WebSocketDisconnect:
            manager.disconnect(project_id, websocket)
        except Exception as e:
            print(f"Error in websocket: {e}")
            manager.disconnect(project_id, websocket)

    return web_app

@app.local_entrypoint()
def main():
    print("Spawning worker process...")
    process_queue.spawn()
    print("To run the server: modal serve backend/websocket_light.py")
