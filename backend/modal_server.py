import modal
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
from pathlib import Path

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
        "open3d"
    ])

volume = modal.Volume.from_name(name="ut3c-heritage")


#application stuff
app = modal.App("ut3c-heritage-backend", image=image)
vol_mnt_loc = Path("/mnt/volume")
@app.function(image=image,volumes={vol_mnt_loc:volume})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Response, status
    from fastapi.responses import JSONResponse, FileResponse
    from PIL import Image
    from fastapi import UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import os
    import uuid
    import io
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


    async def upload_image(id: str, file: UploadFile = File(...)):
        """
            Takes a project id and a file and preprocesses and puts the image 
            into the correct path
            id: str an id that matches one of the existing reconstructions
            file: UploadFile a file represnrting an image
            returns: Path, the direction of the stored image
        """
        path = vol_mnt_loc / "backend_data" / "reconstructions" / id / "images"
        if not os.path.exists(path) or not os.path.isdir(path):
            raise HTTPException(status_code=400,
                                 detail="The provided recosntruction id does not exist")
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="The sent file is not an image")
        
        # Load image with Pillow
        img = None
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        finally:
            file.file.seek(0)  # reset pointer if needed

        target_size = 512

        width, height = img.size
        if width >= height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)

        img = img.resize((new_width, new_height), Image.BICUBIC)
        uuidname = uuid.uuid4()
        img.save(path / f"{uuidname}.png", format="PNG")
        volume.commit()
        return path / f"{uuidname}.png"

    async def run_inference(id: str):
        """
        Runs inference on the specified project.
        id: the id of a project
        returns: Path the path of the .pt object
        """
        print("Running inference...")
        images_path = f"/backend_data/reconstructions/{id}/images"
        inf_res_path = inference_obj.run_inference.remote(images_path)
        print("Inference completed")
        return str(vol_mnt_loc) + inf_res_path + "/predictions.pt"
    
    def process_infered_data(inf_data_path,id,conf_thres = 50):
        """
        """
        pty_location = str(vol_mnt_loc) + f"/backend_data/reconstructions/{id}/models/latest.ply"
        pty_folder = str(vol_mnt_loc) + f"/backend_data/reconstructions/{id}/models"
        Path(pty_folder).mkdir(parents=True, exist_ok=True)
        volume.commit()
        volume.reload()
        predictions = torch.load(inf_data_path,weights_only=False,map_location=torch.device('cpu'))

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
        pcd = simplify_cloud(pcd)
        o3d.io.write_point_cloud(pty_location, pcd)
        print(f"Saved PLY point cloud to {pty_location}")
        volume.commit()
        return pty_location

    def process_infered_data_per_image(inf_data_path, id, conf_thres=50):
        """
        Loads inference data and writes one PLY file per image.
        Handles both batched [1, N, ...] and unbatched [N, ...] formats.
        Prints debug info to understand shapes and counts.
        """
        base_folder = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / str(id) / "models"
        base_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Loading inference data from {inf_data_path} ===")
        predictions = torch.load(inf_data_path, map_location="cpu")

        # --- Keys and basic info ---
        print("Keys in predictions:", list(predictions.keys()))
        pred_world_points = predictions["points"]
        print("points dtype:", pred_world_points.dtype)
        print("points shape:", pred_world_points.shape)

        if "conf" in predictions:
            print("conf shape:", predictions["conf"].shape)
        else:
            print("No confidence array found, using default ones")

        images = predictions["images"]
        print("images shape:", images.shape)
        print("Sample min/max (images):", images.min().item(), images.max().item())

        # --- Convert to numpy ---
        pred_world_points = pred_world_points.numpy()
        pred_world_points_conf = predictions.get("conf", torch.ones_like(predictions["points"])).numpy()
        images = images.numpy()

        # --- Remove batch dimension if present ---
        if pred_world_points.shape[0] == 1 and images.shape[0] == 1:
            print("Removing leading batch dimension (shape [1, N, ...])")
            pred_world_points = pred_world_points[0]
            pred_world_points_conf = pred_world_points_conf[0]
            images = images[0]

        # --- Determine number of images ---
        num_images = pred_world_points.shape[0]
        conf_threshold = conf_thres / 100.0
        print(f"Detected {num_images} image(s)\n")

        ply_paths = []

        for i in range(num_images):
            print(f"--- Processing image {i} ---")
            vertices_3d = pred_world_points[i].reshape(-1, 3)
            colors_rgb = (images[i].reshape(-1, 3) * 255).astype(np.uint8)
            conf = pred_world_points_conf[i].reshape(-1)

            print("  vertices_3d.shape:", vertices_3d.shape)
            print("  colors_rgb.shape:", colors_rgb.shape)
            print("  conf.shape:", conf.shape)
            print("  conf range:", conf.min(), conf.max())

            mask = (conf >= conf_threshold) & (conf > 1e-5)
            print(f"  mask kept {mask.sum()} / {len(mask)} points")

            vertices_3d = vertices_3d[mask]
            colors_rgb = colors_rgb[mask]

            if len(vertices_3d) == 0:
                print("  WARNING: no points kept after filtering")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices_3d)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

            ply_path = base_folder / f"{i:04d}.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"  Saved {ply_path}")
            ply_paths.append(ply_path)

        print(f"\n✅ Done. Saved {len(ply_paths)} PLY files to {base_folder}\n")
        return [str(p) for p in ply_paths]


    @web_app.post("/pointcloud/{id}")
    async def new_image(id: str, file: UploadFile = File(...)):
        uploaded = await upload_image(id,file)
        print("Image uploaded succesfully")
        inference_path = await run_inference(id)
        print("Inference results at " + str(inference_path))
        process_infered_data(str(inference_path),id)
        process_infered_data_per_image(str(inference_path),id)
        
        return JSONResponse(
                content={"status": "OK", "message": f"File {uploaded.name}  created successfully"},
                status_code=status.HTTP_200_OK)

    @web_app.get("/pointcloud/{pc_id}/{tag}")
    async def get_pointcloud(pc_id: str, tag: str):
        # dynamically point to latest PLY for the given project
        ply_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / pc_id / "models" / "latest.ply"

        if not ply_path.exists():
            return {"error": "Point cloud file not found."}

        # Always serve the latest file on disk
        return FileResponse(
            path=str(ply_path),
            filename=f"{pc_id}_{tag}.ply",
            media_type="application/octet-stream"
        )

    return web_app