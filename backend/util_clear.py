import modal
from pathlib import Path
import shutil

image = modal.Image.debian_slim().pip_install([])

volume = modal.Volume.from_name("ut3c-heritage", create_if_missing=True)
vol_mnt_loc = Path("/mnt/volume")

app = modal.App(image=image, volumes={vol_mnt_loc: volume})

@app.function(image=image, volumes={vol_mnt_loc: volume})
def clear_auditorio_data():
    """
    Clears:
      - /backend_data/reconstructions/auditorio/images
      - /preds/*
    while keeping the overall backend structure.
    """
    # --- Target 1: auditorio images ---
    images_folder = vol_mnt_loc / "backend_data" / "reconstructions" / "auditorio" / "images"
    if images_folder.exists() and images_folder.is_dir():
        shutil.rmtree(images_folder)
        print(f"Cleared folder: {images_folder}")
    images_folder.mkdir(parents=True, exist_ok=True)
    print(f"Recreated empty folder: {images_folder}")

    # --- Target 2: preds ---
    preds_folder = vol_mnt_loc / "preds"
    print(preds_folder)
    if preds_folder.exists() and preds_folder.is_dir():
        shutil.rmtree(preds_folder)
        print(f"Cleared folder: {preds_folder}")
    preds_folder.mkdir(parents=True, exist_ok=True)
    print(f"Recreated empty folder: {preds_folder}")

    return {
        "images_folder": str(images_folder),
        "preds_folder": str(preds_folder)
    }

# Optional local test
@app.local_entrypoint()
def main():
    print("Clearing auditorio images and preds remotely...")
    result = clear_auditorio_data.remote()
    print(f"Folders reset at:\n{result}")