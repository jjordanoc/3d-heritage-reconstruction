import os
import subprocess
import time

import modal

# GPU = "A100-80GB"
GPU = "A100-40GB"

cuda_version = "12.1.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
HF_CACHE_PATH = "/cache"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git", "build-essential", "curl", "unzip", "wget", "git-lfs", "clang","libgl1", "libglib2.0-0","libgomp1")
    .pip_install("jupyter")
    .pip_install(
        "torch",
        "torchvision",
        "roma",
        "gradio",
        "matplotlib",
        "tqdm",
        "opencv-python-headless",
        "scipy",
        "einops",
        "gdown",
        "trimesh",
        "pyglet<2",
        "huggingface-hub[torch]>=0.22",
        "evo",
    )
    .pip_install("imageio", "pillow")
    .run_commands("git clone --recursive https://github.com/junyi42/monst3r", "sed -i 's/opencv-python/opencv-python-headless/g' monst3r/requirements.txt", "cd monst3r && pip install -r requirements.txt")
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/torch/lib"})
)


app = modal.App(
    image=image
)

volume = modal.Volume.from_name(
    "v0", create_if_missing=True
)

JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!

HOURS = 3600


@app.function(max_containers=1, volumes={"/root/jupyter": volume}, timeout=24 * HOURS, gpu=GPU)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    viewer_port = 8080
    with modal.forward(jupyter_port) as tunnel, modal.forward(viewer_port) as viewer_tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")
        print(f"Viewer tunnel available at => {viewer_tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 24 * HOURS):
    run_jupyter.remote(timeout=timeout)