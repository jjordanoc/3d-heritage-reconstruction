import modal
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import time
import base64

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


cuda_version = "12.6.3"  # should be no greater than host CUDA version
flavor = "cudnn-devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# environment stuff
image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10").apt_install(
        "git",  # if you need it
        "libgl1-mesa-glx",  # Provides libGL.so.1
        "libglib2.0-0",     # Often needed by Open3D
    ).pip_install([
        "fastapi[standard]",
        "pillow",
        "python-multipart",
        "torch",
        "torchvision",
        "numpy",
        "open3d",
        "requests-toolbelt",
        "gsplat"
    ]).apt_install( # need clang for gsplat's (examples) dependencies
        "build-essential",
        "curl",
        "unzip",
        "wget",
        "clang",
        "libglib2.0-0",
        "libgomp1",
    ).run_commands( # need to install gsplat (examples) dependencies manually, says nowhere in docs
      "git clone https://github.com/nerfstudio-project/gsplat.git",
      "pip install torch torchvision wheel setuptools",
      "cd /gsplat/examples && pip install --no-build-isolation -r requirements.txt"
    )

volume = modal.Volume.from_name(name="ut3c-heritage", create_if_missing=True)

app = modal.App("ut3c-heritage-splat", image=image)
vol_mnt_loc = Path("/mnt/volume")


@app.function(
    image=image,
    volumes={vol_mnt_loc: volume},
    gpu="A10G",  # GPU required for gsplat training
    timeout=3600,  # 1 hour timeout for training
)
def train_gsplat_modal(scene_id: str, data_factor: int = 1, max_steps: int = 30000):
    """
    Modal function that can be called from baking_server.
    Wraps process_gsplat_training for cross-app calling.
    
    Args:
        scene_id: Scene/reconstruction ID
        data_factor: Image downscale factor
        max_steps: Training iterations
    """
    import asyncio
    asyncio.run(process_gsplat_training(scene_id, data_factor, max_steps))


@app.function(
    image=image, 
    volumes={vol_mnt_loc: volume},
    gpu="A10G",  # GPU required for gsplat training
    timeout=3600,  # 1 hour timeout for training
    scaledown_window=240,  # keep warm for 5 minutes
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Response, status, BackgroundTasks
    from fastapi.responses import JSONResponse, FileResponse
    from PIL import Image
    from fastapi import UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from requests_toolbelt.multipart.encoder import MultipartEncoder
    import os
    import uuid
    import io
    import json
    import torch
    import numpy as np
    import open3d as o3d
    import pycolmap
    import subprocess

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

    @web_app.on_event("startup")
    async def clone_gsplat():
        folder_path = vol_mnt_loc / "backend_data" / "gsplat"
        
        # Create parent directory if it doesn't exist
        folder_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clone only if the folder doesn't exist
        if not folder_path.exists():
            subprocess.run([
                "git", "clone", 
                "https://github.com/nerfstudio-project/gsplat.git",
                str(folder_path)
            ], check=True)
            print(f"Cloned gsplat to: {folder_path}")
        else:
            print(f"Repository already exists at: {folder_path}")

    def run_gsplat_training(
        scene_id: str,
        data_dir: Path,
        result_dir: Path,
        data_factor: int = 1,
        max_steps: int = 30000
    ) -> str:
        """
        Run gsplat simple_trainer.py script on COLMAP reconstruction.
        
        Args:
            scene_id: Scene ID for logging
            data_dir: Directory containing images/ and sparse/0/
            result_dir: Output directory for gsplat results
            data_factor: Image downscale factor (1=full, 2=half, 4=quarter)
            max_steps: Training iterations
        
        Returns:
            Path to final .ply file
        """
        import subprocess
        import sys
        
        print(f"{Colors.CYAN}Running gsplat training for scene {scene_id}...{Colors.RESET}")
        print(f"  Data dir: {data_dir}")
        print(f"  Result dir: {result_dir}")
        print(f"  Data factor: {data_factor}")
        print(f"  Max steps: {max_steps}")
        gsplat_install_path = vol_mnt_loc / "backend_data" / "gsplat"
        # Prepare command
        # Note: gsplat's simple_trainer.py should be in Python path or we use python -m
        cmd = [
            sys.executable, f"{gsplat_install_path}/examples/simple_trainer.py", "default",
            "--data_dir", str(data_dir),
            "--data_factor", str(data_factor),
            "--result_dir", str(result_dir),
            "--max_steps", str(max_steps)
        ]
        
        # Create log file
        log_file = result_dir / "training_log.txt"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{Colors.YELLOW}Executing: {' '.join(cmd)}{Colors.RESET}")
        
        # Run training
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to log file and console
            for line in process.stdout:
                f.write(line)
                f.flush()
                # Print important lines to avoid spam
                if "Iteration" in line or "Error" in line or "Saving" in line:
                    print(f"  {line.strip()}")
            
            process.wait()
        
        if process.returncode != 0:
            error_msg = f"gsplat training failed with return code {process.returncode}"
            # Try to read log for more context
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    lines = log_content.split('\n')
                    last_lines = '\n'.join(lines[-50:])
                    error_msg += f"\n\nLast lines from log:\n{last_lines[-2000:]}"
            except:
                error_msg += f"\n\nCould not read log file at {log_file}"
            
            print(f"{Colors.RED}Training failed! Log content:{Colors.RESET}")
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Find the final .ply file
        ply_files = list(result_dir.glob("**/*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No .ply file generated in {result_dir}")
        
        # Return the most recent .ply file
        final_ply = max(ply_files, key=lambda p: p.stat().st_mtime)
        print(f"{Colors.GREEN}✅ Training complete! Final model: {final_ply}{Colors.RESET}")
        
        return str(final_ply)

    async def process_gsplat_training(
        scene_id: str,
        data_factor: int = 1,
        max_steps: int = 30000
    ):
        """
        Background task to run gsplat training.
        
        Args:
            scene_id: Scene/reconstruction ID
            data_factor: Image downscale factor
            max_steps: Training iterations
        """
        func_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}🎨 [BACKGROUND: process_gsplat_training] STARTING for scene {scene_id}{Colors.RESET}")
        print(f"{'='*80}\n")
        
        try:
            # Reload volume
            reload_start = time.time()
            volume.reload()
            log_time("Volume reload", reload_start)
            
            # Define paths
            scene_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / scene_id
            images_dir = scene_path / "images"
            sparse_initial_dir = scene_path / "colmap" / "sparse_initial"
            gsplat_dir = scene_path / "gsplat"
            result_dir = gsplat_dir / "results"
            
            print(f"{Colors.MAGENTA}   Images dir: {images_dir}{Colors.RESET}")
            print(f"{Colors.MAGENTA}   COLMAP dir: {sparse_initial_dir}{Colors.RESET}")
            print(f"{Colors.MAGENTA}   Output dir: {gsplat_dir}{Colors.RESET}")
            
            # Validate inputs
            if not sparse_initial_dir.exists():
                raise FileNotFoundError(
                    f"COLMAP reconstruction not found at {sparse_initial_dir}. "
                    f"Run COLMAP conversion first."
                )
            
            if not images_dir.exists() or not any(images_dir.iterdir()):
                raise FileNotFoundError(f"No images found in {images_dir}")
            
            # Prepare data directory structure for gsplat
            # gsplat expects: data_dir/images/ and data_dir/sparse/0/
            # We have: scene_path/images/ and scene_path/colmap/sparse_initial/
            
            # Create sparse/0 symlink pointing to sparse_initial
            sparse_dir = scene_path / "sparse"
            sparse_0_dir = sparse_dir / "0"
            
            sparse_dir.mkdir(exist_ok=True)
            
            # Remove existing symlink if it exists
            if sparse_0_dir.exists() or sparse_0_dir.is_symlink():
                if sparse_0_dir.is_symlink():
                    sparse_0_dir.unlink()
                else:
                    import shutil
                    shutil.rmtree(sparse_0_dir)
            
            # Create symlink
            sparse_0_dir.symlink_to(sparse_initial_dir.resolve(), target_is_directory=True)
            print(f"{Colors.GREEN}✅ Created symlink: {sparse_0_dir} → {sparse_initial_dir}{Colors.RESET}")
            
            # Save configuration
            config = {
                "scene_id": scene_id,
                "data_factor": data_factor,
                "max_steps": max_steps,
                "timestamp": time.time()
            }
            config_file = gsplat_dir / "config.json"
            gsplat_dir.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Run gsplat training
            print(f"\n{Colors.CYAN}Running gsplat training...{Colors.RESET}")
            train_start = time.time()
            
            final_ply_path = run_gsplat_training(
                scene_id=scene_id,
                data_dir=scene_path,  # Points to directory with images/ and sparse/
                result_dir=result_dir,
                data_factor=data_factor,
                max_steps=max_steps
            )
            
            log_time("gsplat training", train_start)
            
            # Copy final .ply to standard location
            results_ply = gsplat_dir / "results.ply"
            if Path(final_ply_path).exists():
                import shutil
                shutil.copy2(final_ply_path, results_ply)
                print(f"{Colors.GREEN}✅ Final model saved to {results_ply}{Colors.RESET}")
            
            # Commit volume
            commit_start = time.time()
            volume.commit()
            log_time("Volume commit", commit_start)
            
            print(f"\n{'='*80}")
            print(f"{Colors.GREEN}✅ [BACKGROUND: process_gsplat_training] COMPLETE for scene {scene_id}{Colors.RESET}")
            print(f"   Final model: {results_ply}")
            log_time("🎯 TOTAL BACKGROUND TASK TIME", func_start)
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"{Colors.RED}❌ [BACKGROUND: process_gsplat_training] FAILED for scene {scene_id}{Colors.RESET}")
            print(f"{Colors.RED}   Error: {str(e)}{Colors.RESET}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            
            # Save error log
            try:
                error_log = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / scene_id / "gsplat" / "error.log"
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log, 'w') as f:
                    f.write(f"Error: {str(e)}\n\n")
                    f.write(traceback.format_exc())
                volume.commit()
            except:
                pass
            
            # Re-raise exception so Modal function call fails
            raise

    @web_app.post("/scene/{id}/train_gsplat")
    async def train_gaussian_splatting(
        id: str,
        background_tasks: BackgroundTasks,
        data_factor: int = 1,
        max_steps: int = 30000
    ):
        """
        Train 3D Gaussian Splatting model from COLMAP reconstruction.
        
        Prerequisites:
        - COLMAP sparse reconstruction exists (run colmap_ba endpoint first)
        - Images directory exists
        
        Args:
            id: Scene ID
            data_factor: Image downscale (1=full, 2=half, 4=quarter)
            max_steps: Training iterations (default: 30000)
        
        Returns:
            JSON with job status
        """
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}🚀 [POST /scene/{id}/train_gsplat] ENDPOINT CALLED{Colors.RESET}")
        print(f"  data_factor: {data_factor}")
        print(f"  max_steps: {max_steps}")
        print(f"{'='*80}\n")
        
        # Validate prerequisites
        scene_path = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / id
        sparse_initial_dir = scene_path / "colmap" / "sparse_initial"
        images_dir = scene_path / "images"
        
        if not sparse_initial_dir.exists():
            print(f"{Colors.RED}❌ ERROR: COLMAP reconstruction not found at {sparse_initial_dir}{Colors.RESET}")
            raise HTTPException(
                status_code=404,
                detail=f"COLMAP reconstruction not found for scene {id}. Run COLMAP conversion first."
            )
        
        if not images_dir.exists() or not any(images_dir.iterdir()):
            print(f"{Colors.RED}❌ ERROR: No images found in {images_dir}{Colors.RESET}")
            raise HTTPException(
                status_code=400,
                detail=f"No images found for scene {id}. Upload images first."
            )
        
        # Schedule background task
        background_tasks.add_task(process_gsplat_training, id, data_factor, max_steps)
        
        print(f"{Colors.GREEN}✅ gsplat training scheduled for scene {id}{Colors.RESET}")
        log_time("Endpoint response time", endpoint_start)
        print(f"{'='*80}\n")
        
        return {
            "status": "processing",
            "scene_id": id,
            "message": "Gaussian Splatting training started in background",
            "config": {
                "data_factor": data_factor,
                "max_steps": max_steps
            },
            "check_status_at": f"/scene/{id}/train_gsplat/status"
        }

    @web_app.get("/scene/{id}/train_gsplat/status")
    async def get_gsplat_training_status(id: str):
        """
        Check status of gsplat training.
        
        Args:
            id: Scene ID
        
        Returns:
            JSON with processing status and results (if complete)
        """
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}📊 [GET /scene/{id}/train_gsplat/status] ENDPOINT CALLED{Colors.RESET}")
        print(f"{'='*80}\n")
        
        gsplat_dir = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / id / "gsplat"
        results_ply = gsplat_dir / "results.ply"
        config_file = gsplat_dir / "config.json"
        log_file = gsplat_dir / "results" / "training_log.txt"
        error_log = gsplat_dir / "error.log"
        
        # Check for errors
        if error_log.exists():
            try:
                with open(error_log, 'r') as f:
                    error_msg = f.read()
                
                print(f"{Colors.RED}❌ gsplat training failed for scene {id}{Colors.RESET}")
                log_time("Status check", endpoint_start)
                print(f"{'='*80}\n")
                
                return {
                    "status": "failed",
                    "scene_id": id,
                    "error": error_msg[:500]  # Truncate long errors
                }
            except:
                pass
        
        # Check if complete
        if results_ply.exists():
            config = {}
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                except:
                    pass
            
            # Get file size
            ply_size_mb = results_ply.stat().st_size / (1024 * 1024)
            
            print(f"{Colors.GREEN}✅ gsplat training complete for scene {id}{Colors.RESET}")
            log_time("Status check", endpoint_start)
            print(f"{'='*80}\n")
            
            return {
                "status": "complete",
                "scene_id": id,
                "model_path": str(results_ply),
                "model_size_mb": round(ply_size_mb, 2),
                "config": config,
                "download_url": f"/scene/{id}/gsplat/model"
            }
        
        # Check if processing
        elif log_file.exists() or config_file.exists():
            print(f"{Colors.YELLOW}⏳ gsplat training in progress for scene {id}{Colors.RESET}")
            log_time("Status check", endpoint_start)
            print(f"{'='*80}\n")
            
            # Try to get last log line for progress
            progress_info = None
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Find last line with "Iteration"
                        for line in reversed(lines[-50:]):  # Check last 50 lines
                            if "Iteration" in line or "Step" in line:
                                progress_info = line.strip()
                                break
                except:
                    pass
            
            return {
                "status": "processing",
                "scene_id": id,
                "message": "Gaussian Splatting training in progress",
                "progress": progress_info
            }
        
        # Not started
        else:
            print(f"{Colors.YELLOW}⏳ gsplat training not started for scene {id}{Colors.RESET}")
            log_time("Status check", endpoint_start)
            print(f"{'='*80}\n")
            
            return {
                "status": "not_started",
                "scene_id": id,
                "message": "Gaussian Splatting training not started"
            }

    @web_app.get("/scene/{id}/gsplat/model")
    async def download_gsplat_model(id: str):
        """
        Download the trained .ply file.
        
        Args:
            id: Scene ID
        
        Returns:
            FileResponse with results.ply
        """
        endpoint_start = time.time()
        print(f"\n{'='*80}")
        print(f"{Colors.CYAN}📥 [GET /scene/{id}/gsplat/model] ENDPOINT CALLED{Colors.RESET}")
        print(f"{'='*80}\n")
        
        results_ply = Path(vol_mnt_loc) / "backend_data" / "reconstructions" / id / "gsplat" / "results.ply"
        
        if not results_ply.exists():
            print(f"{Colors.RED}❌ ERROR: Model file not found at {results_ply}{Colors.RESET}")
            raise HTTPException(
                status_code=404,
                detail=f"Trained model not found for scene {id}. Run training first."
            )
        
        print(f"{Colors.GREEN}✅ Serving model file: {results_ply}{Colors.RESET}")
        log_time("Endpoint response time", endpoint_start)
        print(f"{'='*80}\n")
        
        return FileResponse(
            path=results_ply,
            filename=f"{id}_gaussian_splatting.ply",
            media_type="application/octet-stream"
        )

    return web_app

