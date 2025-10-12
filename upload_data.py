import modal

vol = modal.Volume.from_name("v0", create_if_missing=True)

with vol.batch_upload() as batch:
    batch.put_directory("../data/iphone13", "/data/iphone13")