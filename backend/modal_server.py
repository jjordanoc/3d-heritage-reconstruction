import modal
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
from pathlib import Path

# environment stuff
image = modal.Image.debian_slim().pip_install([
    "fastapi[standard]",
    "pillow",
    "python-multipart"
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

    web_app = FastAPI()
    Pi3Remote = modal.Cls.from_name("pi3-inference", "ModelInference")
    inference_obj = Pi3Remote()

    image_hashes = {}
    @asynccontextmanager
    def lifespan():
        folder_path = vol_mnt_loc / "backend_data" / "reconstructions" / "auditorio" / "images"
        shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized folder: {folder_path}")
        yield

    @web_app.post("/pointcloud/{id}")
    async def upload_image(id: str, file: UploadFile = File(...)):
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
        
        return JSONResponse(
                content={"status": "OK", "message": f"File {uuidname}.png  created successfully"},
                status_code=status.HTTP_200_OK)

    @web_app.get("/pointcloud/{pc_id}/{tag}")
    async def get_pointcloud(pc_id: str, tag: str):
        path = vol_mnt_loc / "backend_data" / "reconstructions" / pc_id / "images"
        print(inference_obj.run_inference.remote(vol_mnt_loc))

    return web_app