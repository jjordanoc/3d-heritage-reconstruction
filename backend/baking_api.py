import modal
# from fastapi import FastAPI, BackgroundTasks, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import subprocess
import time
import json
import os

# ANSI Color codes for logging
class Colors:
    CYAN = '\033[96m'      # Function entry/exit
    YELLOW = '\033[93m'    # Timing info
    RED = '\033[91m'       # Warnings/Errors
    GREEN = '\033[92m'     # Success
    MAGENTA = '\033[95m'   # Data info
    RESET = '\033[0m'      # Reset color

def log_time(label, start_time):
    """Helper to log elapsed time in yellow"""
    elapsed = time.time() - start_time
    print(f"{Colors.YELLOW}⏱️  [{label}] took {elapsed:.3f}s{Colors.RESET}")
    return elapsed

# Image definition - Placeholder as per user instructions
# User will handle the actual setup and installation of MapAnything
# image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10").apt_install(
#     "git",
#     "libgl1-mesa-glx",
#     "libglib2.0-0"
# ).pip_install(
#     "fastapi[standard]",
#     "shutil", # Standard lib but listed for clarity
#     "requests-toolbelt"
# )

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands(
        # Clone the Pi3 repository
        "git clone https://github.com/facebookresearch/map-anything.git /root/map-anything",
        # Install from requirements.txt
        "cd /root/map-anything && pip install -e .",
        # build the curope extension for faster inference
        #"cd /root/Pi3/croco/models/curope && python setup.py build_ext --inplace", 
        # use opencv headless version
        # "pip uninstall -y opencv-python", 
    )
    .run_commands(
      'cd /root/map-anything && pip install -e ".[colmap]"'
    )
    # .run_commands(
    #   'cd /root/map-anything && pre-commit install'
    # )
    # .pip_install("opencv-python-headless")
    .pip_install("requests")
    .env({"PYTHONPATH": "/root/map-anything"})
)

volume = modal.Volume.from_name(name="ut3c-heritage", create_if_missing=True)
vol_mnt_loc = Path("/mnt/volume")

app = modal.App("ut3c-heritage-baking-api", image=image)

@app.function(
    image=image,
    volumes={vol_mnt_loc: volume},
    timeout=3600, # 1 hour timeout
    gpu="A100-80GB", # GPU required for MapAnything
    scaledown_window=300
)
def process_map_anything_task(scene_id: str, memory_efficient: bool = True, use_ba: bool = False):
    """
    Background task to run MapAnything COLMAP pipeline.
    """
    func_start = time.time()
    print(f"\n{'='*80}")
    print(f"{Colors.CYAN}🔧 [BACKGROUND: process_map_anything_task] STARTING for scene {scene_id}{Colors.RESET}")
    print(f"   Config: memory_efficient={memory_efficient}, use_ba={use_ba}")
    print(f"{'='*80}\n")

    try:
        # 1. Reload volume
        reload_start = time.time()
        volume.reload()
        log_time("Volume reload", reload_start)

        # 2. Define paths
        reconstructions_dir = vol_mnt_loc / "backend_data" / "reconstructions"
        scene_dir = reconstructions_dir / scene_id
        source_images_dir = scene_dir / "images"
        
        colmap_dir = scene_dir / "colmap"
        colmap_images_dir = colmap_dir / "images"
        
        status_file = colmap_dir / "status.json"
        log_file = colmap_dir / "map_anything.log"

        print(f"{Colors.MAGENTA}   Source Images: {source_images_dir}{Colors.RESET}")
        print(f"{Colors.MAGENTA}   Target COLMAP Dir: {colmap_dir}{Colors.RESET}")

        if not source_images_dir.exists():
             raise FileNotFoundError(f"Source images directory not found: {source_images_dir}")

        # Initialize status
        colmap_dir.mkdir(parents=True, exist_ok=True)
        initial_status = {
            "status": "processing",
            "stage": "initializing",
            "timestamp": time.time(),
            "config": {"memory_efficient": memory_efficient, "use_ba": use_ba}
        }
        with open(status_file, 'w') as f:
            json.dump(initial_status, f, indent=2)

        # 3. Copy images
        print(f"{Colors.CYAN}Step 1: Copying images to COLMAP directory...{Colors.RESET}")
        copy_start = time.time()
        
        if colmap_images_dir.exists():
            shutil.rmtree(colmap_images_dir)
        shutil.copytree(source_images_dir, colmap_images_dir)
        
        log_time("Copy images", copy_start)
        print(f"{Colors.GREEN}✅ Copied images to {colmap_images_dir}{Colors.RESET}")

        # 4. Commit volume before external process
        # Even though we are in the same container, good practice if we were splitting tasks
        # volume.commit() 

        # 5. Run MapAnything
        print(f"\n{Colors.CYAN}Step 2: Running MapAnything demo_colmap.py...{Colors.RESET}")
        map_anything_start = time.time()

        # Construct command
        # python scripts/demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --memory_efficient_inference --use_ba
        cmd = [
            "python",
            "./map-anything/scripts/demo_colmap.py",
            f"--scene_dir={colmap_dir}",
        ]

        if memory_efficient:
            cmd.append("--memory_efficient_inference")
        
        if use_ba:
            cmd.append("--use_ba")
            # Optional: add default BA params if needed, but keeping it simple as requested
            # cmd.extend(["--max_query_pts=2048", "--query_frame_num=5"])

        print(f"{Colors.YELLOW}Executing: {' '.join(cmd)}{Colors.RESET}")
        print(f"CWD: {os.getcwd()}")

        print(f"{Colors.CYAN}--- Directory Listing ---{Colors.RESET}")
        subprocess.run(["ls", "-R"], check=False) # -F adds trailing / to dirs
        print(f"{Colors.CYAN}-------------------------{Colors.RESET}")

        # Open log file
        with open(log_file, 'w') as log_f:
            log_f.write(f"Command: {' '.join(cmd)}\n\n")
            log_f.flush()

            # Update status to running
            current_status = initial_status.copy()
            current_status["stage"] = "running_map_anything"
            with open(status_file, 'w') as f:
                json.dump(current_status, f, indent=2)


            
            # Run subprocess
            process = subprocess.Popen(
                cmd,
                # cwd="/map-anything",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output
            for line in process.stdout:
                print(line, end='')
                log_f.write(line)
            
            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"MapAnything script failed with return code {process.returncode}")

        log_time("MapAnything execution", map_anything_start)
        print(f"{Colors.GREEN}✅ MapAnything completed successfully{Colors.RESET}")

        # 6. Final Commit
        print(f"\n{Colors.CYAN}Step 3: Committing results...{Colors.RESET}")
        
        # Update status to complete
        final_status = {
            "status": "complete",
            "timestamp": time.time(),
            "config": {"memory_efficient": memory_efficient, "use_ba": use_ba},
            "output_dir": str(colmap_dir)
        }
        with open(status_file, 'w') as f:
            json.dump(final_status, f, indent=2)

        commit_start = time.time()
        volume.commit()
        log_time("Volume commit", commit_start)

        print(f"\n{'='*80}")
        print(f"{Colors.GREEN}✅ [BACKGROUND: process_map_anything_task] COMPLETE{Colors.RESET}")
        log_time("Total duration", func_start)
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"{Colors.RED}❌ [BACKGROUND: process_map_anything_task] FAILED{Colors.RESET}")
        print(f"{Colors.RED}   Error: {str(e)}{Colors.RESET}")
        print(f"{'='*80}\n")
        
        # Try to write error status
        try:
            err_status = {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
            with open(status_file, 'w') as f:
                json.dump(err_status, f, indent=2)
            volume.commit()
        except:
            pass
        
        import traceback
        traceback.print_exc()

# @app.function(
#     image=image,
#     volumes={vol_mnt_loc: volume},
#     timeout=60,
#     scaledown_window=300
# )
# @modal.asgi_app()
# def fastapi_app():
#     web_app = FastAPI()
    
#     web_app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     @web_app.post("/scene/{id}/colmap_ba")
#     async def run_colmap_ba(
#         id: str,
#         background_tasks: BackgroundTasks,
#         memory_efficient: bool = True,
#         use_ba: bool = True
#     ):
#         """
#         Trigger MapAnything COLMAP pipeline.
#         """
#         print(f"\n{Colors.CYAN}🚀 [POST] Triggering MapAnything for scene {id}{Colors.RESET}")
        
#         # Verify source exists locally (requires volume reload if checking existence strictly, 
#         # but we delegate actual heavy check to background task to return fast)
#         # For now, assume valid if ID is provided.
        
#         background_tasks.add_task(process_map_anything_task, id, memory_efficient, use_ba)
        
#         return {
#             "status": "processing",
#             "scene_id": id,
#             "message": "MapAnything COLMAP pipeline started in background",
#             "config": {
#                 "memory_efficient": memory_efficient,
#                 "use_ba": use_ba
#             },
#             "check_status_at": f"/scene/{id}/colmap_ba/status"
#         }

#     @web_app.get("/scene/{id}/colmap_ba/status")
#     async def get_status(id: str):
#         """
#         Check status of the MapAnything pipeline.
#         """
#         volume.reload() # Ensure we see latest file updates
        
#         colmap_dir = vol_mnt_loc / "backend_data" / "reconstructions" / id / "colmap"
#         status_file = colmap_dir / "status.json"
        
#         if not status_file.exists():
#             return {"status": "not_started", "scene_id": id}
            
#         try:
#             with open(status_file, 'r') as f:
#                 status_data = json.load(f)
#             return status_data
#         except Exception as e:
#             return {
#                 "status": "error_reading_status", 
#                 "error": str(e),
#                 "scene_id": id
#             }

#     return web_app


@app.local_entrypoint()
def main():
    handle = process_map_anything_task.spawn("L507", True, False)
    handle.get()

if __name__ == "__main__":
    main()


