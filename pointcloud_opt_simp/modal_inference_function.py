import modal
import os
import uuid
import glob
import torch
import math
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms


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
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4", "pillow", "plyfile", "huggingface_hub", "safetensors")
    .env({"PYTHONPATH": "/root/Pi3"})
)

#image = modal.Image.debian_slim(python_version="3.10")
app = modal.App("model_inference_ramtensors")

@app.cls(gpu="a10g",image=image,timeout=600,scaledown_window=240)
class PI3Model:
    @modal.enter()
    def __enter__(self):
        # This code runs ONCE when the container starts.
        # All your model loading and setup goes here.
        print("Imports...")
        import torch
        from pi3.models.pi3 import Pi3
        from pi3.utils.geometry import depth_edge
        
        # Store imports we'll need in the inference method
        self.torch = torch
        self.depth_edge = depth_edge

        #load model
        print("Loading model...")
        self.device = "cuda"
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device)
        self.model.eval()
        print("Model loaded succesfully and is persistent.")


    @modal.method()
    def run_inference(self,tensorized_images):
        tensorized_images = tensorized_images.to(self.device)
        #run forward pass
        print("Running model inference...")
        dtype = torch.bfloat16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = self.model(tensorized_images[None])  # Add batch dimension
        print("Model done")
        #postprocessing
        predictions['images'] = tensorized_images[None].permute(0, 1, 3, 4, 2)
        predictions['conf'] = torch.sigmoid(predictions['conf'])
        
        # Replace with your actual depth_edge function
        edge = self.depth_edge(predictions['local_points'][..., 2], rtol=0.03)
        predictions['conf'][edge] = 0.0
        del predictions['local_points']

        predictions['points'] = predictions['points'].cpu()
        predictions['camera_poses'] = predictions['camera_poses'].cpu()
        predictions['conf'] = predictions['conf'].cpu()
        predictions['images'] = predictions['images'].cpu()

        return predictions
    
def load_images(path,PIXEL_LIMIT=255000):
    import os
    from PIL import Image
    import math
    from torchvision import transforms
    import argparse
    import torch
    sources = []
    filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
    shape = 500
    for i in range(0, len(filenames)):
        img_path = os.path.join(path, filenames[i])
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except:
            print("Failed to load image {filenames[i]}")
    #resize (copied from PI3)
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")
    # resize/tensorize
    tensor_tf = transforms.ToTensor()
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")
    return torch.stack(tensor_list, dim=0)

def runall(img_path, results_path):
    # This function runs locally
    print(f"Loading images from {img_path}...")
    imgs = load_images(img_path)
    
    if imgs is None:
        print("Image loading failed, aborting run.")
        return

    # Create a local "stub" for the remote class
    model_stub = PI3Model()
    
    print("Calling remote inference method...")
    # .call() is synchronous: it runs the remote function and waits for the result
    pt_results = model_stub.run_inference.remote(imgs)
    
    print("Inference complete. Saving results locally...")
    
    # pt_results is the dictionary returned from run_inference,
    # which already contains CPU tensors. We can save it directly.
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(pt_results, results_path)
    print(f"Results saved to {results_path}")

@app.local_entrypoint()
def main():
    runall("./data/sample1","./data/predictions/test_run.pt")
    #runall("./data/sample2","./data/predictions/sample2_results.pt")
    #runall("./data/sample3","./data/predictions/sample3_results.pt")