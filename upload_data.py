import modal
import os

vol = modal.Volume.from_name("v0", create_if_missing=True)

LOCAL_DIR = "../data/auditori"
TMP_DIR = "../data/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

import os
from PIL import Image
from pathlib import Path
import shutil

src_dirs = [
    LOCAL_DIR
]

def resize_to_max512(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= 512:
        return img  # no resize needed
    scale = 512 / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)



with vol.batch_upload() as batch:
    for src in src_dirs:
        for idx, path in enumerate(Path(src).rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    img_type = path.suffix.lower()
                    img = Image.open(path)
                    # only landscape images
                    if img.width > img.height:
                        img = resize_to_max512(img)
                        # keep filename but flatten structure
                        out_path = f"{TMP_DIR}/{path.name}"
                        img.save(out_path)
                        batch.put_file(out_path, f"/data/auditorio/iphone13/{idx}{img_type}")
                        img.close()
                        print(f"✅ Saved {out_path}")
                except Exception as e:
                    print(f"❌ Skipped {path}: {e}")


    # batch.put_directory(LOCAL_DIR, "/data/auditorio/iphone13")