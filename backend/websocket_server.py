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
import os
import shutil






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
    "torchvision",
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


@app.function(image=image, volumes={vol_mnt_loc: volume}, timeout=60*60*24)
@modal.asgi_app()
def fastapi_app():
    from PIL import Image
    import io
    import uuid
    import glob
    import open3d
    import typing
    import numpy
    import os
    from PIL import Image
    import math
    from torchvision import transforms
    import argparse
    import torch
    
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


    # Pipeline
    def run_inference(tensor_imgs):
        pi3_obj = modal.Cls.from_name("model_inference_ramtensors", "PI3Model")
        pi3Model = pi3_obj() #instantiate the class
        predictions = pi3Model.run_inference.remote(tensor_imgs)
        print("Inference complete. Received results locally.")
        return predictions

    def tensorize_images(images):
        transform = transforms.ToTensor()
        tensors = torch.stack([transform(img) for img in images],dim=0)
        return tensors

    def do_inference(images):
        print(f"[DO_INFERENCE] Tensorizing {len(images)} images...")
        tensor_imgs = tensorize_images(images)
        print(f"[DO_INFERENCE] Tensor shape: {tensor_imgs.shape}")
        print(f"[DO_INFERENCE] Calling remote inference...")
        inference = run_inference(tensor_imgs)
        print(f"[DO_INFERENCE] Remote inference returned")
        return inference

    def unify_pointcloud(pointclouds: list[open3d.geometry.PointCloud]) -> open3d.geometry.PointCloud:
        full_pc = open3d.geometry.PointCloud()
        for pc in pointclouds:
            full_pc += pc
        return full_pc

    def do_registration(new_ply: open3d.geometry.PointCloud,
                        last_ply: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        """

        Computes the appropiate transformation matrix to take the newest inference
        (concat(pointclouds)) to the old one's (last_ply) coordinate system.
        Arguments:
        pointclouds: List of latest pointclouds per image
        last_ply: point cloud of the newest inference 
        """
        NORMAL_EST_KNN = 100
        DOWNSAMPLE = 500
        TANGENT_PLANE_K = 15
        FPFH_KNN = 100
        MAX_CORR_DST_FGR = 0.2
        FGR_ITERS = 64

        # simplify plys
        pc1_simple = new_ply.uniform_down_sample(DOWNSAMPLE)
        pc2_simple = last_ply.uniform_down_sample(DOWNSAMPLE)

        #calculate required RANSAC stuff:

        #Normal estimation
        pc1_simple.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_EST_KNN), 
            fast_normal_computation=True
        )
        pc2_simple.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_EST_KNN), 
            fast_normal_computation=True
        )
        # Normal direction regularization
        pc1_simple.orient_normals_consistent_tangent_plane(k=TANGENT_PLANE_K)
        pc2_simple.orient_normals_consistent_tangent_plane(k=TANGENT_PLANE_K)

        # FPFH Feature Calculation
        pc1_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pc1_simple,
                                        open3d.geometry.KDTreeSearchParamKNN(knn=FPFH_KNN))
        pc2_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pc2_simple,
                                        open3d.geometry.KDTreeSearchParamKNN(knn=FPFH_KNN))

        fgr_option = open3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=MAX_CORR_DST_FGR,
            iteration_number=FGR_ITERS
        )

        rough_reg_res = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            pc1_simple, pc2_simple,
            pc1_fpfh, pc2_fpfh,
            option=fgr_option
        )

        return rough_reg_res.transformation

    def tensor_to_pointcloud(model_output,threshold=0.3):
        pts = model_output["points"].squeeze(0)
        colors = model_output["images"].squeeze(0)
        confidence = model_output["conf"].squeeze(0)
        cameras = model_output["camera_poses"].squeeze(0)


        pcds = []
        campos = []

        
        for i in range(pts.shape[0]):
            # from x*y*3  tensor to (x*y)*3 to numpy
            image_pts_raw = pts[i]
            image_colors_raw = colors[i]
            image_conf_raw = confidence[i]
            flat_pts = image_pts_raw.reshape(-1, 3)
            flat_colors = image_colors_raw.reshape(-1, 3)
            flat_confidence = image_conf_raw.reshape(-1, 1)
            img_pts = flat_pts.numpy()
            img_col = flat_colors.numpy()
            img_conf = flat_confidence.numpy()

            #thresholding - building the threshold
            threshmask = img_conf > threshold
            threshmask = numpy.squeeze(threshmask)

            #actually thresholding
            #print(threshmask)
            img_pts = img_pts[threshmask]
            img_col = img_col[threshmask]

            pcd = open3d.geometry.PointCloud()

            pcd.points = open3d.utility.Vector3dVector(img_pts)
            pcd.colors = open3d.utility.Vector3dVector(img_col)
            
            pcds.append(pcd)
            campos.append(cameras[i].numpy())
        return (pcds,campos)

    def run_pipeline(images,last_ply):
        """
        Arguments:
        Images: PIL.Image list of images to run inference on
        last_ply: An optional last ply file to wich the new infered pointcloud will be registered
        Returns:
        unified,pointclouds
        """
        print(f"\n[RUN_PIPELINE] Starting")
        print(f"  Number of images: {len(images)}")
        print(f"  Has reference PLY: {last_ply is not None}")
        
        print(f"[RUN_PIPELINE] Running inference on {len(images)} images...")
        result_tensor = do_inference(images)
        print(f"[RUN_PIPELINE] Inference complete")
        
        print(f"[RUN_PIPELINE] Converting tensors to point clouds...")
        pointclouds, cameras = tensor_to_pointcloud(result_tensor)
        print(f"[RUN_PIPELINE] Created {len(pointclouds)} point clouds")
        
        print(f"[RUN_PIPELINE] Unifying point clouds...")
        unified = unify_pointcloud(pointclouds)
        print(f"[RUN_PIPELINE] Unified point cloud has {len(unified.points)} points")
        
        if (last_ply is not None):
            print(f"[RUN_PIPELINE] Performing registration to reference frame...")
            print(f"  Reference PLY points: {len(last_ply.points)}")
            tf = do_registration(unified,last_ply)
            print(f"[RUN_PIPELINE] Registration complete, transformation matrix:")
            print(tf)

            print(f"[RUN_PIPELINE] Applying transformation to all point clouds...")
            for i, pcd in enumerate(pointclouds):
                pcd.transform(tf)
            unified.transform(tf)

            for i, cam in enumerate(cameras):
                cameras[i] = tf @ cam
            print(f"[RUN_PIPELINE] Transformation applied to {len(pointclouds)} clouds and {len(cameras)} cameras")
        else:
            print(f"[RUN_PIPELINE] No registration needed (first reconstruction)")
        
        print(f"[RUN_PIPELINE] Complete\n")
        return unified,pointclouds,cameras

    def read_images(path:str,new_id:str,PIXEL_LIMIT=255000) -> list[Image.Image]:
        """
        Reads a directory's worth of images.. Makes sure that the last image in the returned list
        is the image with name new_id. new_id may be the file name or the file name with extension.
        Arguments:
        path: A path to a directory with images. 
        new_id: A file name with or without extension residing in that directory
        PIXEL_LIMIT: a limit for the ammount of pixels sent to the model

        Returns:
        list[Image.Image]: a list of PIL images, where the last image is the image with file name new_id
        """
        print(f"\n[READ_IMAGES] Starting")
        print(f"  path: {path}")
        print(f"  new_id: {new_id}")
        print(f"  PIXEL_LIMIT: {PIXEL_LIMIT}")
        
        sources = []
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"[READ_IMAGES] Found {len(filenames)} image files:")
        for i, fn in enumerate(filenames):
            print(f"    {i}: {fn}")
        
        #Make sure new_id is the last image
        new_id_filename = next((name for name in filenames 
                                if os.path.splitext(name)[0] == new_id or name == new_id), 
                            None)
        
        if new_id_filename:
            print(f"[READ_IMAGES] Found matching file for new_id: {new_id_filename}")
            filenames.remove(new_id_filename)
            filenames.append(new_id_filename)
            print(f"[READ_IMAGES] Reordered so new_id is last")
        else:
            print(f"[READ_IMAGES] ERROR: new_id '{new_id}' not found in directory!")
            print(f"[READ_IMAGES] Searched for files matching:")
            print(f"    - Basename without extension: {new_id}")
            print(f"    - Exact filename: {new_id}")
            raise Exception("new_id no existe en el directorio especificado")
        
        print(f"\n[READ_IMAGES] Final image order:")
        for i, fn in enumerate(filenames):
            print(f"    {i}: {fn}")
        
        shape = 500
        for i in range(0, len(filenames)):
            img_path = os.path.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except:
                print(f"[READ_IMAGES] Failed to load image {filenames[i]}")
        
        print(f"[READ_IMAGES] Successfully loaded {len(sources)} images")
        
        #resize (copied from PI3)
        first_img = sources[0]
        W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
        print(f"[READ_IMAGES] All images will be resized to: ({TARGET_W}, {TARGET_H})")

        image_list = []
        
        for img_pil in sources[0:-1]: #avoid last image, if exception ocurs there the pipeline should fail
            try:
                # Resize to the uniform target size
                resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                # Convert to tensor
                image_list.append(resized_img)
            except Exception as e:
                print(f"[READ_IMAGES] Error processing an image: {e}")
        #process last image and dont catch
        resized_img = sources[-1].resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        image_list.append(resized_img)

        print(f"[READ_IMAGES] Returning {len(image_list)} processed images\n")
        return image_list

    def addImageToCollection(inPath,oldPly,new_id,outputs_directory = "./data/pointclouds",save_all_shards=False):
        """
        Arguments
        inPath: Path to a folder containing the images
        oldPly: (optional) path pointing to a PLY file of the last full scene to perform registration.
                if None is passed registration is ommited.
        new_id: string corresponding to the latest image
        outputs_directory: The directory in wich to save the outputs
        save_all_shards: If True, save individual PLY files for all images (for full refetch)
        
        returns: a path to a directory containing at least the latest infered pointcloud and {i}.ply the pointcloud 
                for the new_id (latest) image.
        
        Internally,
        Performs inference on a given path, 
        Performs conversion to pointcloud
        Serializes the point clouds
        Registers the new point cloud to the old point cloud to allow for online addition
        """
        print(f"\n[ADD_IMAGE_TO_COLLECTION] Starting")
        print(f"  inPath: {inPath}")
        print(f"  oldPly: {oldPly}")
        print(f"  new_id: {new_id}")
        print(f"  outputs_directory: {outputs_directory}")
        print(f"  save_all_shards: {save_all_shards}")
        
        new_path = outputs_directory + "/latest"
        print(f"[ADD_IMAGE_TO_COLLECTION] new_path will be: {new_path}")
        
        #if there is someone currently on the latest path, i must move them
        if os.path.exists(new_path):
            print(f"[ADD_IMAGE_TO_COLLECTION] new_path already exists, archiving it")
            #each old ply bears the name of he who obsoleted them, as an eternal reminder
            #of our mortality and how age comes for us all
            old_path = outputs_directory + f'/{new_id.split(".")[0]}'
            print(f"[ADD_IMAGE_TO_COLLECTION] Renaming {new_path} -> {old_path}")
            os.rename(new_path,old_path)
            print(f"[ADD_IMAGE_TO_COLLECTION] Creating new directory: {new_path}")
            os.mkdir(new_path)
        else:
            print(f"[ADD_IMAGE_TO_COLLECTION] new_path does not exist yet, creating it")
            os.makedirs(new_path, exist_ok=True)

        images = read_images(inPath,new_id)
        print(f"[ADD_IMAGE_TO_COLLECTION] Loaded {len(images)} images from read_images()")

        last_ply = None
        if oldPly is not None:
            print(f"[ADD_IMAGE_TO_COLLECTION] Loading reference point cloud: {oldPly}")
            last_ply = open3d.io.read_point_cloud(oldPly)
            print(f"[ADD_IMAGE_TO_COLLECTION] Reference cloud has {len(last_ply.points)} points")
        else:
            print(f"[ADD_IMAGE_TO_COLLECTION] No reference point cloud (first reconstruction)")

        print(f"[ADD_IMAGE_TO_COLLECTION] Running pipeline (inference + registration)...")
        unified, pcds, cam_estimates = run_pipeline(images,last_ply)
        
        print(f"[ADD_IMAGE_TO_COLLECTION] Pipeline complete:")
        print(f"  Unified point cloud: {len(unified.points)} points")
        print(f"  Individual point clouds: {len(pcds)}")
        for i, pcd in enumerate(pcds):
            print(f"    {i}: {len(pcd.points)} points")
        
        full_ply_path = new_path + "/full.ply"
        print(f"\n[ADD_IMAGE_TO_COLLECTION] Writing full point cloud: {full_ply_path}")
        open3d.io.write_point_cloud(full_ply_path, unified)
        
        # Save the last point cloud with the new_id name
        new_id_basename = new_id.split(".")[0]
        last_ply_path = new_path + f'/{new_id_basename}.ply'
        print(f"[ADD_IMAGE_TO_COLLECTION] Writing last image's point cloud: {last_ply_path}")
        print(f"  new_id: '{new_id}'")
        print(f"  new_id_basename: '{new_id_basename}'")
        print(f"  Last point cloud index: {len(pcds)-1}")
        print(f"  Last point cloud points: {len(pcds[-1].points)}")
        open3d.io.write_point_cloud(last_ply_path, pcds[-1])
        
        # For full refetch, save all individual point clouds as shards
        if save_all_shards:
            print(f"[ADD_IMAGE_TO_COLLECTION] Saving all {len(pcds)} shards (full refetch mode)")
            for i, pcd in enumerate(pcds):
                shard_path = new_path + f"/{i:04d}.ply"
                print(f"  Writing shard {i}: {shard_path} ({len(pcd.points)} points)")
                open3d.io.write_point_cloud(shard_path, pcd)
        else:
            print(f"[ADD_IMAGE_TO_COLLECTION] Not saving individual shards (incremental mode)")
        
        print(f"\n[ADD_IMAGE_TO_COLLECTION] Complete. Returning: {new_path}")
        print(f"[ADD_IMAGE_TO_COLLECTION] Files written:")
        if os.path.exists(new_path):
            for f in os.listdir(new_path):
                fpath = os.path.join(new_path, f)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"    - {f} ({size_kb:.2f} KB)")
        
        return new_path

    
    
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
    
    
    def save_image_to_volume(project_id: str, img: Image.Image) -> tuple[str, str, str]:
        """
        Save preprocessed image to volume.
        Returns: (image_path, image_id_with_ext, image_id_without_ext)
        """
        images_folder = vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "images"
        images_folder.mkdir(parents=True, exist_ok=True)
        
        image_id = str(uuid.uuid4())
        image_path = images_folder / f"{image_id}.png"
        
        img.save(image_path, format="PNG")
        volume.commit()
        
        return str(image_path), f"{image_id}.png", image_id
    
    
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
    
    
    def reconstruction_wrapper(
        project_id: str,
        image_id: str,
        is_full_refetch: bool
    ) -> str:
        """
        Wrapper around pipeline.addImageToCollection for Modal integration.
        
        Args:
            project_id: Heritage site ID
            image_id: UUID of the newly uploaded image (without .png extension)
            is_full_refetch: Whether this is a full refetch (every 10th)
        
        Returns:
            Path to folder containing shard PLY files
        """
        print(f"\n{'='*80}")
        print(f"[RECONSTRUCTION_WRAPPER] Starting reconstruction")
        print(f"  project_id: {project_id}")
        print(f"  image_id: {image_id}")
        print(f"  is_full_refetch: {is_full_refetch}")
        print(f"{'='*80}\n")
        
        # Modal volume paths
        images_folder = str(vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "images")
        outputs_folder = str(vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "models")
        old_ply_path = outputs_folder + "/latest/full.ply"
        
        print(f"[RECONSTRUCTION_WRAPPER] Constructed paths:")
        print(f"  images_folder: {images_folder}")
        print(f"  outputs_folder: {outputs_folder}")
        print(f"  old_ply_path: {old_ply_path}")
        
        # Check if reference PLY exists
        if not os.path.exists(old_ply_path):
            print(f"[RECONSTRUCTION_WRAPPER] Reference PLY does NOT exist, will be first reconstruction")
            old_ply_path = None
        else:
            print(f"[RECONSTRUCTION_WRAPPER] Reference PLY EXISTS, will perform registration")
        
        # List images before calling pipeline
        if os.path.exists(images_folder):
            image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"\n[RECONSTRUCTION_WRAPPER] Images in folder ({len(image_files)} total):")
            for img_file in image_files:
                print(f"    - {img_file}")
            print()
        else:
            print(f"[RECONSTRUCTION_WRAPPER] WARNING: images_folder does not exist!")
        
        # Call pipeline with full_refetch flag
        print(f"[RECONSTRUCTION_WRAPPER] Calling addImageToCollection...")
        result_folder = addImageToCollection(
            inPath=images_folder,
            oldPly=old_ply_path,
            new_id=image_id,  # UUID without extension
            outputs_directory=outputs_folder,
            save_all_shards=is_full_refetch
        )
        print(f"[RECONSTRUCTION_WRAPPER] addImageToCollection returned: {result_folder}")
        
        # List what was created
        if os.path.exists(result_folder):
            created_files = os.listdir(result_folder)
            print(f"\n[RECONSTRUCTION_WRAPPER] Files created in result folder:")
            for f in created_files:
                file_path = os.path.join(result_folder, f)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"    - {f} ({file_size:.2f} KB)")
        else:
            print(f"[RECONSTRUCTION_WRAPPER] WARNING: result_folder does not exist!")
        
        volume.commit()  # Persist to Modal volume
        print(f"[RECONSTRUCTION_WRAPPER] Volume committed")
        
        # Return appropriate folder based on refetch type
        if is_full_refetch:
            print(f"[RECONSTRUCTION_WRAPPER] Full refetch - returning result folder with all shards")
            return result_folder
        else:
            print(f"[RECONSTRUCTION_WRAPPER] Incremental update - creating temp folder with single shard")
            
            # Return folder with just the new image's PLY
            # Create temp folder with only the incremental update
            temp_folder = result_folder + "_incremental"
            print(f"[RECONSTRUCTION_WRAPPER] Creating temp folder: {temp_folder}")
            os.makedirs(temp_folder, exist_ok=True)
            
            # Copy just the new image's PLY
            src = os.path.join(result_folder, f"{image_id}.ply")
            dst = os.path.join(temp_folder, "0000.ply")  # Name as shard 0
            
            print(f"[RECONSTRUCTION_WRAPPER] Copying incremental shard:")
            print(f"  Source: {src}")
            print(f"  Dest: {dst}")
            print(f"  Source exists: {os.path.exists(src)}")
            
            volume.reload()
            
            if not os.path.exists(src):
                print(f"[RECONSTRUCTION_WRAPPER] ERROR: Source file does not exist!")
                print(f"[RECONSTRUCTION_WRAPPER] Files in result_folder:")
                for f in os.listdir(result_folder):
                    print(f"    - {f}")
                raise FileNotFoundError(f"Expected file not found: {src}")
            
            shutil.copy(src, dst)
            print(f"[RECONSTRUCTION_WRAPPER] Copy successful")
            
            dst_size = os.path.getsize(dst) / 1024
            print(f"[RECONSTRUCTION_WRAPPER] Copied file size: {dst_size:.2f} KB")
            
            return temp_folder
    
    
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
                        image_path, image_id_with_ext, image_id = save_image_to_volume(project_id, img)
                        
                        print(f"Received image for project {project_id}, saved as {image_id}")
                        
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
                        
                        # Determine if this is a full refetch BEFORE calling reconstruction
                        state = manager.get_project_state(project_id)
                        next_count = state.get("update_count", 0) + 1
                        is_full_refetch = (next_count % 10 == 0)
                        
                        # Call reconstruction pipeline
                        try:
                            shard_folder = reconstruction_wrapper(
                                project_id=project_id,
                                image_id=image_id,  # UUID without extension
                                is_full_refetch=is_full_refetch
                            )
                            
                            print(f"Reconstruction complete. Shard folder: {shard_folder}")
                            
                        except Exception as e:
                            print(f"Reconstruction failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            await manager.broadcast_to_project(project_id, {
                                "type": "error",
                                "message": f"Reconstruction failed: {str(e)}",
                                "image_id": image_id
                            })
                            continue
                        
                        # Increment update counter
                        current_count = manager.increment_update_count(project_id)
                        state = manager.get_project_state(project_id)
                        state["total_images"] = state.get("total_images", 0) + 1
                        manager.update_project_state(project_id, **state)
                        
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

