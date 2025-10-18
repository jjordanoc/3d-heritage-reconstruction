import modal

vol = modal.Volume.from_name("v0", create_if_missing=True)

LOCAL_DIR = "../data/iphone13"

with vol.batch_upload() as batch:
    batch.put_directory(LOCAL_DIR, "/data/iphone13")