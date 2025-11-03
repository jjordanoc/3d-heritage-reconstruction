"""
WebSocket-based Heritage Reconstruction Server

Provides real-time, multi-user 3D reconstruction streaming with:
- Incremental point cloud updates via WebSocket
- Multi-user broadcasting per heritage site
- Periodic full refetch every 10 updates
- Progressive shard streaming for mobile support
"""

import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, List, Optional
import json
import base64
import asyncio
from datetime import datetime

from pipeline import *

# Same Modal image and volume as existing server
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "git",
    "libgl1-mesa-glx",  # Provides libGL.so.1
    "libglib2.0-0",     # Often needed by Open3D
).pip_install([
    "fastapi[standard]",
    "pillow",
    "python-multipart",
    "torch",
    "numpy",
    "open3d",
])

volume = modal.Volume.from_name(name="ut3c-heritage")

# Modal app
app = modal.App("ut3c-heritage-websocket", image=image)
vol_mnt_loc = Path("/mnt/volume")


class ConnectionManager:
    """
    Manages WebSocket connections per heritage site/project.
    Handles broadcasting updates to all clients viewing the same site.
    """
    
    def __init__(self):
        # project_id -> list of active websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # project_id -> metadata about the project state
        self.project_state: Dict[str, dict] = {}
        
    async def connect(self, project_id: str, websocket: WebSocket):
        """Accept a new WebSocket connection for a project"""
        await websocket.accept()
        
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        
        self.active_connections[project_id].append(websocket)
        
        # Initialize project state if needed
        if project_id not in self.project_state:
            self.project_state[project_id] = {
                "update_count": 0,
                "last_full_refetch": None,
                "total_images": 0,
            }
        
        print(f"Client connected to project {project_id}. Total clients: {len(self.active_connections[project_id])}")
        
    def disconnect(self, project_id: str, websocket: WebSocket):
        """Remove a disconnected WebSocket"""
        if project_id in self.active_connections:
            if websocket in self.active_connections[project_id]:
                self.active_connections[project_id].remove(websocket)
                print(f"Client disconnected from project {project_id}. Remaining clients: {len(self.active_connections[project_id])}")
            
            # Clean up empty project lists
            if len(self.active_connections[project_id]) == 0:
                del self.active_connections[project_id]
                print(f"No more clients for project {project_id}, cleaned up connection list")
    
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send a message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message to client: {e}")
    
    async def broadcast_to_project(self, project_id: str, message: dict):
        """Broadcast a message to all clients viewing a specific project"""
        if project_id not in self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections[project_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(project_id, connection)
    
    def get_project_state(self, project_id: str) -> dict:
        """Get the current state for a project"""
        return self.project_state.get(project_id, {
            "update_count": 0,
            "last_full_refetch": None,
            "total_images": 0,
        })
    
    def update_project_state(self, project_id: str, **kwargs):
        """Update project state"""
        if project_id not in self.project_state:
            self.project_state[project_id] = {
                "update_count": 0,
                "last_full_refetch": None,
                "total_images": 0,
            }
        
        self.project_state[project_id].update(kwargs)
    
    def increment_update_count(self, project_id: str) -> int:
        """Increment and return the update count for a project"""
        state = self.get_project_state(project_id)
        new_count = state.get("update_count", 0) + 1
        self.update_project_state(project_id, update_count=new_count)
        return new_count
    
    def reset_update_count(self, project_id: str):
        """Reset update count after full refetch"""
        self.update_project_state(
            project_id, 
            update_count=0,
            last_full_refetch=datetime.now().isoformat()
        )


@app.function(image=image, volumes={vol_mnt_loc: volume})
@modal.asgi_app()
def fastapi_app():
    from PIL import Image
    import io
    import uuid
    import os
    import glob
    
    # Import black_box stub (will be replaced with actual implementation)
    try:
        from black_box_stub import black_box
    except ImportError:
        # Fallback if not available yet
        def black_box(input_img_path: str, old_point_cloud: str, new_img_id: str) -> str:
            raise NotImplementedError("black_box function not implemented yet")
    
    web_app = FastAPI(title="Heritage Reconstruction WebSocket Server")
    
    # CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For testing - restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global connection manager
    manager = ConnectionManager()
    
    @web_app.on_event("startup")
    async def startup():
        print("WebSocket Heritage Reconstruction Server started")
    
    
    # ============================================================================
    # Helper Functions
    # ============================================================================
    
    def preprocess_image(image_bytes: bytes, target_size: int = 512) -> Image.Image:
        """Preprocess an uploaded image (resize while maintaining aspect ratio)"""
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
    
    
    def save_image_to_volume(project_id: str, img: Image.Image) -> tuple[str, str]:
        """
        Save preprocessed image to volume.
        Returns: (image_path, image_id)
        """
        images_folder = vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "images"
        images_folder.mkdir(parents=True, exist_ok=True)
        
        image_id = str(uuid.uuid4())
        image_path = images_folder / f"{image_id}.png"
        
        img.save(image_path, format="PNG")
        volume.commit()
        
        return str(image_path), image_id
    
    
    def get_reference_pointcloud_path(project_id: str) -> Optional[str]:
        """
        Get the path to the current reference point cloud (latest.ply).
        Returns None if no reconstruction exists yet.
        """
        ply_path = vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "models" / "latest.ply"
        
        if ply_path.exists():
            return str(ply_path)
        return None
    
    
    def read_ply_as_base64(ply_path: str) -> str:
        """Read a PLY file and encode it as base64 string"""
        with open(ply_path, 'rb') as f:
            ply_bytes = f.read()
        return base64.b64encode(ply_bytes).decode('utf-8')
    
    
    async def stream_shards_to_project(
        project_id: str, 
        shard_folder: str, 
        is_full_refetch: bool,
        sequence: int
    ):
        """
        Stream all PLY shards from a folder to all clients in a project.
        
        Args:
            project_id: The heritage site ID
            shard_folder: Path to folder containing shard PLY files
            is_full_refetch: Whether this is a full refetch or incremental update
            sequence: Current update sequence number
        """
        # Find all PLY files in the folder
        shard_files = sorted(glob.glob(os.path.join(shard_folder, "*.ply")))
        
        if not shard_files:
            print(f"WARNING: No PLY shards found in {shard_folder}")
            return
        
        total_shards = len(shard_files)
        print(f"Streaming {total_shards} shard(s) to project {project_id} (full_refetch={is_full_refetch})")
        
        # Send refetch notification if this is a full refetch
        if is_full_refetch and total_shards > 1:
            await manager.broadcast_to_project(project_id, {
                "type": "refetch_starting",
                "reason": "10 updates reached - full reconstruction",
                "total_shards": total_shards,
                "sequence": sequence
            })
        
        # Stream each shard
        for idx, shard_path in enumerate(shard_files):
            try:
                shard_data = read_ply_as_base64(shard_path)
                
                message = {
                    "type": "shard_update",
                    "sequence": sequence,
                    "shard_index": idx,
                    "total_shards": total_shards,
                    "is_full_refetch": is_full_refetch,
                    "shard_data": shard_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                await manager.broadcast_to_project(project_id, message)
                print(f"Streamed shard {idx + 1}/{total_shards} to project {project_id}")
                
                # Small delay between shards to avoid overwhelming clients
                if total_shards > 1:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error streaming shard {shard_path}: {e}")
                await manager.broadcast_to_project(project_id, {
                    "type": "error",
                    "message": f"Failed to stream shard {idx}",
                    "error": str(e)
                })
    
    
    async def send_initial_state(project_id: str, websocket: WebSocket):
        """
        Send the current reference set to a newly connected client.
        """
        state = manager.get_project_state(project_id)
        
        # Check if we have a reference set
        models_folder = vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "models"
        
        if not models_folder.exists():
            # No reconstruction yet
            await manager.send_to_client(websocket, {
                "type": "initial_state",
                "message": "No reconstruction available yet. Waiting for first image.",
                "total_images": 0
            })
            return
        
        # Find all shard files from the current reference set
        # These are numbered files like 0000.ply, 0001.ply, etc.
        shard_files = sorted(glob.glob(str(models_folder / "[0-9]*.ply")))
        
        if not shard_files:
            await manager.send_to_client(websocket, {
                "type": "initial_state",
                "message": "Reconstruction in progress. No reference set available yet.",
                "total_images": state.get("total_images", 0)
            })
            return
        
        # Send notification that initial state is being streamed
        await manager.send_to_client(websocket, {
            "type": "initial_state_starting",
            "total_shards": len(shard_files),
            "total_images": state.get("total_images", 0)
        })
        
        # Stream all shards from the reference set
        for idx, shard_path in enumerate(shard_files):
            try:
                shard_data = read_ply_as_base64(shard_path)
                
                await manager.send_to_client(websocket, {
                    "type": "initial_state_shard",
                    "shard_index": idx,
                    "total_shards": len(shard_files),
                    "shard_data": shard_data
                })
                
                # Small delay between shards
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error sending initial shard {shard_path}: {e}")
                await manager.send_to_client(websocket, {
                    "type": "error",
                    "message": f"Failed to send initial shard {idx}",
                    "error": str(e)
                })
        
        # Done sending initial state
        await manager.send_to_client(websocket, {
            "type": "initial_state_complete",
            "total_images": state.get("total_images", 0)
        })
    
    
    # ============================================================================
    # WebSocket Endpoint
    # ============================================================================
    
    @web_app.websocket("/ws/{project_id}")
    async def websocket_endpoint(websocket: WebSocket, project_id: str):
        """
        Main WebSocket endpoint for heritage reconstruction.
        
        Clients connect with a project_id (heritage site identifier).
        They can upload images and receive real-time reconstruction updates.
        """
        await manager.connect(project_id, websocket)
        
        try:
            # Send initial state (current reference set) to the newly connected client
            await send_initial_state(project_id, websocket)
            
            # Main message loop
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                # ============================================================
                # Handle Image Upload
                # ============================================================
                if message_type == "upload_image":
                    try:
                        # Extract image data
                        image_data_b64 = data.get("image_data")
                        if not image_data_b64:
                            await manager.send_to_client(websocket, {
                                "type": "error",
                                "message": "No image data provided"
                            })
                            continue
                        
                        # Decode base64 image
                        image_bytes = base64.b64decode(image_data_b64)
                        
                        # Preprocess and save image
                        img = preprocess_image(image_bytes)
                        image_path, image_id = save_image_to_volume(project_id, img)
                        
                        print(f"Received image for project {project_id}, saved as {image_id}")
                        
                        # Get reference point cloud
                        old_point_cloud_path = get_reference_pointcloud_path(project_id)
                        
                        # Acknowledge receipt
                        await manager.send_to_client(websocket, {
                            "type": "upload_acknowledged",
                            "image_id": image_id,
                            "message": "Image received, processing..."
                        })
                        
                        # Broadcast to all clients that processing started
                        await manager.broadcast_to_project(project_id, {
                            "type": "processing_started",
                            "image_id": image_id
                        })
                        
                        # Call black_box function for reconstruction
                        try:
                            shard_folder = black_box(
                                input_img_path=image_path,
                                old_point_cloud=old_point_cloud_path or "",
                                new_img_id=image_id
                            )
                            
                            print(f"black_box returned shard folder: {shard_folder}")
                            
                        except NotImplementedError:
                            # black_box not implemented yet - send placeholder response
                            await manager.broadcast_to_project(project_id, {
                                "type": "error",
                                "message": "black_box function not implemented yet. This is a stub response.",
                                "image_id": image_id
                            })
                            continue
                        
                        # Increment update counter
                        current_count = manager.increment_update_count(project_id)
                        state = manager.get_project_state(project_id)
                        state["total_images"] = state.get("total_images", 0) + 1
                        manager.update_project_state(project_id, **state)
                        
                        # Determine if this is a full refetch (every 10th update)
                        is_full_refetch = (current_count % 10 == 0)
                        
                        # Stream shards to all connected clients
                        await stream_shards_to_project(
                            project_id=project_id,
                            shard_folder=shard_folder,
                            is_full_refetch=is_full_refetch,
                            sequence=current_count
                        )
                        
                        # Reset counter if full refetch
                        if is_full_refetch:
                            manager.reset_update_count(project_id)
                        
                        # Notify completion
                        await manager.broadcast_to_project(project_id, {
                            "type": "processing_complete",
                            "image_id": image_id,
                            "sequence": current_count,
                            "total_images": state["total_images"]
                        })
                        
                    except Exception as e:
                        print(f"Error processing image upload: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        await manager.broadcast_to_project(project_id, {
                            "type": "error",
                            "message": f"Failed to process image: {str(e)}"
                        })
                
                # ============================================================
                # Handle Request for Resync
                # ============================================================
                elif message_type == "request_resync":
                    print(f"Client requested resync for project {project_id}")
                    await send_initial_state(project_id, websocket)
                
                # ============================================================
                # Handle Ping/Keepalive
                # ============================================================
                elif message_type == "ping":
                    await manager.send_to_client(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # ============================================================
                # Unknown message type
                # ============================================================
                else:
                    await manager.send_to_client(websocket, {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
        
        except WebSocketDisconnect:
            manager.disconnect(project_id, websocket)
            print(f"Client disconnected from project {project_id}")
        
        except Exception as e:
            print(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()
            manager.disconnect(project_id, websocket)
    
    
    # ============================================================================
    # HTTP Health Check Endpoint
    # ============================================================================
    
    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "heritage-reconstruction-websocket",
            "active_projects": len(manager.active_connections),
            "total_connections": sum(len(conns) for conns in manager.active_connections.values())
        }
    
    
    @web_app.get("/projects/{project_id}/status")
    async def project_status(project_id: str):
        """Get status of a specific project"""
        state = manager.get_project_state(project_id)
        active_clients = len(manager.active_connections.get(project_id, []))
        
        return {
            "project_id": project_id,
            "active_clients": active_clients,
            "update_count": state.get("update_count", 0),
            "total_images": state.get("total_images", 0),
            "last_full_refetch": state.get("last_full_refetch")
        }
    
    return web_app

