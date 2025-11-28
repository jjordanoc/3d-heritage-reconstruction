import modal
import os
import glob
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

# 1. Define the Container Image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scikit-learn",
        "tqdm",
        "pillow"
    )
)
volume = modal.Volume.from_name(name="ut3c-heritage", create_if_missing=True)

app = modal.App("resnet-keyframer", image=image)
vol_mnt_loc = Path("/mnt/volume")


@app.cls(gpu="T4", timeout=600,volumes={vol_mnt_loc:volume})
class ResNetKeyframer:
    """
    Implements the ad-hoc keyframe selection algorithm
    using ResNet50 embeddings and a similarity graph.
    """
    # We move model loading to __enter__ so it happens once when the container starts
    # rather than every time we call a function.
    @modal.enter()
    def load_model(self):
        import torch
        from torchvision import models
        from torch.nn import Sequential
        import shutil
        from torchvision import transforms

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Pre-trained ResNet50
        print("Loading ResNet50 model...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final classification layer to get embeddings (2048-dim)
        self.model = Sequential(*list(self.model.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define Transformations
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        print("Model loaded successfully.")

    def _extract_embedding(self, image_path):
        import torch
        from PIL import Image

        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0)
            batch_t = batch_t.to(self.device)
            
            with torch.no_grad():
                embedding = self.model(batch_t)
            
            return embedding.squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"Warning: Failed to process {image_path}. Error: {e}")
            return None

    # This is the main remote function we will call
    @modal.method()
    def do_selection_in_project(self, project_id, max_exceedence, desired_keyframes, too_similar_threshold=0.85):
        import shutil
        from pathlib import Path
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Ensure vol_mnt_loc is defined (assuming it's a global or class var, or define it here)
        # vol_mnt_loc = Path("/data") 
        
        full_images_path = vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "all_images"
        selected_images_path = vol_mnt_loc / "backend_data" / "reconstructions" / project_id / "images"

        # Check existing count
        current_image_count = len(glob.glob(str(selected_images_path / "*.png")))
        
        # Note: Logic changed here slightly. Usually you check the source count, not destination count?
        # Assuming you want to check if the SOURCE has too many images. 
        # If you meant checking if we already ran selection, checking destination is fine.
        source_images = glob.glob(str(full_images_path / "*.png"))
        
        if len(source_images) <= desired_keyframes + max_exceedence:
            print(f"Source count {len(source_images)} is within limit. Doing nothing")
            return

        # --- Extract Embeddings ---
        image_embeddings = dict()
        for img_path in tqdm(source_images, desc="Extracting embeddings"):
            # FIX: Use self._extract_embedding
            embedding = self._extract_embedding(img_path) 
            if embedding is not None:
                image_embeddings[img_path] = embedding

        # --- Build Matrix ---
        img_paths = list(image_embeddings.keys())
        embeddings_matrix = np.array([image_embeddings[img] for img in img_paths])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        np.fill_diagonal(similarity_matrix, 0)
        similarity_matrix[similarity_matrix > too_similar_threshold] = 0

        # --- Selection Loop ---
        selected_images = set()
        
        while len(selected_images) < desired_keyframes:
            # FIX: Infinite Loop Protection
            if np.max(similarity_matrix) <= 0:
                print("No more valid edges found in graph.")
                break

            max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            img1_idx, img2_idx = max_sim_idx
            
            # Add to set (using the full path string)
            selected_images.add(img_paths[img1_idx])
            selected_images.add(img_paths[img2_idx])
            
            # Zero out rows/cols to avoid re-picking
            similarity_matrix[img1_idx, :] = 0
            similarity_matrix[:, img1_idx] = 0
            similarity_matrix[img2_idx, :] = 0
            similarity_matrix[:, img2_idx] = 0

        # FIX: Random Backfill (if graph didn't find enough)
        if len(selected_images) < desired_keyframes:
            print(f"Graph selected {len(selected_images)}. Filling remainder randomly.")
            remaining = [p for p in img_paths if p not in selected_images]
            np.random.shuffle(remaining)
            needed = desired_keyframes - len(selected_images)
            selected_images.update(remaining[:needed])

        # --- Copy Files (Preserving Names) ---
        # Clear destination
        if os.path.exists(selected_images_path):
            shutil.rmtree(selected_images_path)
        os.makedirs(selected_images_path, exist_ok=True)

        for img_path in selected_images:
            img_name = os.path.basename(img_path) # <--- Name preserved here
            dest_path = selected_images_path / img_name
            shutil.copy(img_path, dest_path)
            
        print(f"Selection complete. Copied {len(selected_images)} images.")

    @modal.method()
    def clear_folder(self,path_to_folder):
        import shutil
        from pathlib import Path

        folder_path = Path(path_to_folder)
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)


#local entry point for testing
@app.local_entrypoint()
def main():
    keyframer_stub = ResNetKeyframer()
    #keyframer_stub.clear_folder.remote("/mnt/volume/backend_data/reconstructions/CUARTO PROCHAZKA/all_images")
    #keyframer_stub.clear_folder.remote("/mnt/volume/backend_data/reconstructions/CUARTO PROCHAZKA/images")
    keyframer_stub.do_selection_in_project.remote("CUARTO PROCHAZKA", max_exceedence=0, desired_keyframes=7)


    