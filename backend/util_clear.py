import modal
from pathlib import Path
import shutil

image = modal.Image.debian_slim().pip_install([])

volume = modal.Volume.from_name("ut3c-heritage", create_if_missing=True)
vol_mnt_loc = Path("/mnt/volume")

app = modal.App(image=image)

@app.function(image=image, volumes={vol_mnt_loc: volume})
def reset_backend():
    backend_path = vol_mnt_loc / "backend_data"

    if backend_path.exists() and backend_path.is_dir():
        shutil.rmtree(backend_path)
        print(f"Removed existing folder: {backend_path}")

    backend_path.mkdir(parents=True, exist_ok=True)
    print(f"Created empty folder: {backend_path}")
    return str(backend_path)

# Optional local test
@app.local_entrypoint()
def main():
    print("Resetting backend remotely...")
    # Runs in the Modal cloud, affects the mounted volume
    path = reset_backend.remote()
    print(f"Backend reset at: {path}")