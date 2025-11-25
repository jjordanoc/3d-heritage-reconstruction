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
]).add_local_file("pipeline.py", "/root/pipeline.py")

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

# Worker function
@app.function(
    image=image, 
    volumes={VOL_MOUNT_PATH: volume}, 
    timeout=1200,
    concurrency_limit=10
)
def process_queue():
    # We assume pipeline.py is available in the working directory or installed
    try:
        import pipeline
    except ImportError:
        # Fallback if pipeline is not in path (e.g. strict sandbox)
        print("WARNING: Could not import pipeline. Ensure pipeline.py is mounted.")
        return

    print("Worker started, waiting for tasks...")
    while True:
        try:
            # Fetch item (blocking)
            # Queue items: {"project_id": str, "image_id": str}
            item = reconstruction_queue.get() 
            
            project_id = item["project_id"]
            image_id = item["image_id"]
            
            # --- LOCK CHECK ---
            # Check if currently processing using modal.Dict as lock
            try:
                current_state = reconstruction_state.get(project_id)
            except KeyError:
                current_state = None
            
            if current_state and current_state.get("status") == "processing":
                # Project is busy! Put it back and wait a bit.
                print(f"Project {project_id} is busy. Re-queueing {image_id}")
                reconstruction_queue.put(item) 
                time.sleep(1) # Prevent hot loop
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
            
            # Check for existing reconstruction for registration
            old_ply_path = outputs_folder / "latest" / "full.ply"
            if not old_ply_path.exists():
                old_ply_path_str = None
                print("No existing reconstruction found. Starting fresh.")
            else:
                old_ply_path_str = str(old_ply_path)
                print(f"Found existing reconstruction at {old_ply_path_str}")

            # Reload volume to ensure we have latest files
            volume.reload()

            # Run pipeline
            try:
                result_path = pipeline.addImageToCollection(
                    inPath=str(images_folder),
                    oldPly=old_ply_path_str,
                    new_id=image_id,
                    outputs_directory=str(outputs_folder),
                    save_all_shards=False 
                )
                
                print(f"Pipeline finished. Result at {result_path}")
                
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
    This isolates blocking I/O and Modal API calls from the async event loop.
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

