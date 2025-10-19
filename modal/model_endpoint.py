import modal



# Define your image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        # Clone the Pi3 repository
        "git clone https://github.com/yyfz/Pi3.git /root/Pi3",
        # Install from requirements.txt
        "pip install -r /root/Pi3/requirements.txt",
        # build the curope extension for faster inference
        #"cd /root/Pi3/croco/models/curope && python setup.py build_ext --inplace", 
        # use opencv headless version
        "pip uninstall -y opencv-python", 
    )
    .pip_install("opencv-python-headless")
    .env({"PYTHONPATH": "/root/Pi3"})
)

app = modal.App("pi3-inference")

# Create volume for predictions
volume = modal.Volume.from_name("ut3c-heritage", create_if_missing=True)


MNT_DIR = "/mnt/vol"

@app.cls(
    image=image,
    gpu="a10g",  # or "a100", "t4", etc.
    volumes={MNT_DIR: volume},
    timeout=600,  # 10 minutes
    scaledown_window=60,  # keep warm for 5 minutes
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Load model once when container starts"""
        print("Loading model...")
        from pi3.models.pi3 import Pi3
        import torch
        
         # Device check
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")
        self.device = "cuda"
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    @modal.method()
    def run_inference(self, target_dir: str) -> str:
        """
        Run inference and save predictions to volume.
        
        Args:
            target_dir: Remote path to directory containing images in volume
            
        Returns:
            Remote path to saved predictions in volume
        """
        from pi3.utils.geometry import se3_inverse, homogenize_points, depth_edge
        from pi3.utils.basic import load_images_as_tensor
        import os
        import uuid
        from pathlib import Path
        import glob
        import torch
        
        
        # Generate unique ID for this prediction
        pred_uuid = str(uuid.uuid4())
        output_abs_ref = f"/preds/{pred_uuid}" # absolute reference to the output directory
        output_dir = Path(f"{MNT_DIR}{output_abs_ref}")
        output_dir.mkdir(parents=True, exist_ok=True)

        input_dir = Path(f"{MNT_DIR}{target_dir}")
        print(f"Processing images from {input_dir}")
        
        # Load and preprocess images
        image_names = glob.glob(os.path.join(input_dir, "*"))
        image_names = sorted(image_names)
        print(f"Found {len(image_names)} images")
        
        if len(image_names) == 0:
            raise ValueError("No images found. Check your upload.")
        
        interval = 1
        # Replace with your actual image loading function
        imgs = load_images_as_tensor(
            input_dir,
            interval=interval
        ).to(self.device)
        
        # Run inference
        print("Running model inference...")
        dtype = torch.bfloat16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = self.model(imgs[None])  # Add batch dimension
        
        # Your post-processing
        predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
        predictions['conf'] = torch.sigmoid(predictions['conf'])
        
        # Replace with your actual depth_edge function
        edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
        predictions['conf'][edge] = 0.0
        del predictions['local_points']
        
        # Much simpler and fast enough
        predictions_path = output_dir / "predictions.pt"
        torch.save(predictions, predictions_path)

        # Commit to volume
        volume.commit()

        torch.cuda.empty_cache()

        return str(output_abs_ref)


# Deploy with: modal deploy your_script.py
# Or run ephemeral with: modal serve your_script.py

@app.local_entrypoint()
def test():
    """Test the endpoint locally"""
    model = ModelInference()
    result_path = model.run_inference.remote("/data/auditorio/iphone13")
    print(f"Predictions saved to: {result_path}")
