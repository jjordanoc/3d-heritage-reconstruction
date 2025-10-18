import modal
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
from pathlib import Path

# environment stuff
image = modal.Image.debian_slim().pip_install([
    "fastapi[standard]",
    "pillow",
    "python-multipart"
    "torch"
])

volume = modal.Volume.from_name(name="ut3c-heritage")


#application stuff
app = modal.App(image=image)
vol_mnt_loc = Path("/mnt/volume")
@app.function(image=image,volumes={vol_mnt_loc:volume})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Response, status
    from fastapi.responses import JSONResponse
    from PIL import Image
    from fastapi import UploadFile, File, HTTPException
    import os
    import uuid
    import io
    from contextlib import asynccontextmanager
    import shutil
    import torch

    web_app = FastAPI()
    Pi3Remote = modal.Cls.from_name("pi3-inference", "ModelInference")
    inference_obj = Pi3Remote()

    @asynccontextmanager
    def lifespan():
        folder_path = vol_mnt_loc / "backend_data" / "reconstructions" / "auditorio" / "images"
        shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized folder: {folder_path}")
        yield
    

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
        return path / f"{uuidname}.png"

    async def run_inference(id: str):
        """
        Runs inference on the specified project.
        id: the id of a project
        returns: Path the path of the .pt object
        """
        path = vol_mnt_loc / "backend_data" / "reconstructions" / id / "images"
        return inference_obj.run_inference.remote(vol_mnt_loc)
    
    def process_infered_data(inf_data_path,pty_location,conf_thresh = 50):
        """
        """
        data = torch.load(inf_data_path)
        for i in data:
            print(i)


    @web_app.post("/pointcloud/{id}")
    async def new_image(id: str, file: UploadFile = File(...)):
        uploaded = await upload_image(id,file)
        inference_path = run_inference()
        process_infered_data(inference_path,"")
        
        return JSONResponse(
                content={"status": "OK", "message": f"File {uploaded.name}.png  created successfully"},
                status_code=status.HTTP_200_OK)

    @web_app.get("/pointcloud/{pc_id}/{tag}")
    async def get_pointcloud(pc_id: str, tag: str):
        pass

    return web_app