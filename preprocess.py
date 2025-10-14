import modal



image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision", 
        "numpy",
        "pillow",
        "opencv-python-headless",  # Full OpenCV without GUI dependencies
        "imageio",
        "matplotlib",
    )
    .apt_install("git", "curl")
)

app = modal.App("preprocess", image=image)

volume = modal.Volume.from_name(
    "v0", create_if_missing=True
)

JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!

HOURS = 3600


@app.function(max_containers=1, volumes={"/root/jupyter": volume}, timeout=24 * HOURS)
def preprocess(image_dir: str, output_file: str, model_resolution: int):
    """
    Preprocesses a directory of images for the DUSt3R model.
    This involves gathering all image files, running the model-specific
    'load_images' function, and saving the processed data to a file.
    """
    import os
    import torch
    from glob import glob
    import sys

    # Add the project root to the Python path to allow for correct module imports
    # This assumes the script is run from the root of the 3d-heritage-reconstruction project
    current_directory = os.getcwd()
    print(f"--> The script is running from: {current_directory}")
    if not os.path.exists("/root/jupyter/monst3r/"):
      os.system("git clone --recursive https://github.com/junyi42/monst3r /root/jupyter/monst3r")
      print(f"--> cloning monst3r")
    else:
      print(f"--> monst3r already exists")

    sys.path.insert(0, "/root/jupyter/monst3r")
    sys.path.insert(0, "/root/jupyter/monst3r/dust3r")

    # Now we can import the project's functions
    from dust3r.utils.image import load_images
    # 1. Gather all image files recursively and sort them
    print(f"--> Searching for images in {image_dir}")
    image_paths = glob(os.path.join(image_dir, '**/*.jpg'), recursive=True) + \
                  glob(os.path.join(image_dir, '**/*.jpeg'), recursive=True) + \
                  glob(os.path.join(image_dir, '**/*.png'), recursive=True) + \
                  glob(os.path.join(image_dir, '**/*.JPG'), recursive=True)
    
    if not image_paths:
        print("Error: No images found. Check the directory and file extensions.")
        return
        
    image_paths.sort()
    print(f"--> Found {len(image_paths)} images.")

    # 2. Run the model's preprocessing pipeline
    # The 'size' parameter should match the one used during inference in demo.py
    print(f"--> Preprocessing images to size {model_resolution}...")
    processed_imgs = load_images(image_paths, size=model_resolution, verbose=True)

    # 3. Save the processed data using torch.save for efficiency
    print(f"--> Saving preprocessed data to {output_file}")
    torch.save(processed_imgs, output_file)
    print("--> Preprocessing complete.")


@app.local_entrypoint()
def main(image_dir: str = "/root/jupyter/data/",
    output_file: str = "/root/jupyter/data/preprocessed_data.pth",
    model_resolution: int = 512):
    preprocess.remote(image_dir=image_dir, output_file=output_file, model_resolution=model_resolution)