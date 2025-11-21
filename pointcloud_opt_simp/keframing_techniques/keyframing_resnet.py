import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import Sequential
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import shutil

class ResNetKeyframer:
    """
    Implements the ad-hoc keyframe selection algorithm described by the user,
    using ResNet50 embeddings and a similarity graph.
    """

    def __init__(self):
        # --- 1. Load Pre-trained ResNet50 ---
        # Check if CUDA is available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # We use 'weights=models.ResNet50_Weights.DEFAULT' for the latest weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We'll use the output of the 'avgpool' layer, which is a 
        # 2048-dimensional vector before the final classifier.
        self.model = Sequential(*list(self.model.children())[:-1])
        
        # Move model to the selected device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # --- 2. Set up Image Transformations ---
        # The model expects 224x224 images, normalized
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # --- 3. Internal Storage ---
        self.image_paths = []
        self.embeddings = None

    def _extract_embedding(self, image_path):
        """
        Loads a single image, preprocesses it, and extracts its
        ResNet50 embedding.
        """
        try:
            # Open with PIL.Image, convert to RGB
            img = Image.open(image_path).convert('RGB')
            # Apply preprocessing
            img_t = self.preprocess(img)
            # Add a 'batch' dimension (e.g., [3, 224, 224] -> [1, 3, 224, 224])
            batch_t = torch.unsqueeze(img_t, 0)
            
            # Move the tensor to the selected device
            batch_t = batch_t.to(self.device)
            
            # Get the embedding
            with torch.no_grad(): # No need to calculate gradients
                embedding = self.model(batch_t)
            
            # Squeeze to remove batch and spatial dims,
            # move to CPU, and convert to a flat numpy array
            return embedding.squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"Warning: Failed to process {image_path}. Error: {e}")
            return None

    def process_folder(self, folder_path):
        """
        Extracts embeddings for all images in a folder.
        """
        print(f"Processing folder: {folder_path}")
        image_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
        image_paths.extend(sorted(glob.glob(os.path.join(folder_path, '*.png'))))
        
        self.image_paths = []
        embeddings_list = []
        
        for path in tqdm(image_paths, desc="Extracting Features"):
            emb = self._extract_embedding(path)
            if emb is not None:
                self.image_paths.append(path)
                embeddings_list.append(emb)
        
        # Store as a single large numpy array
        self.embeddings = np.array(embeddings_list)
        print(f"Successfully processed {len(self.image_paths)} images.")
        
    def select_keyframes(self, k=31, threshold=0.85):
        """
        Selects keyframes based on the described graph algorithm.
        
        Args:
            k (int): The required number of keyframes.
            threshold (float): The similarity threshold (T). Edges ABOVE
                               this value are removed.
        
        Returns:
            list: A list of image paths for the selected keyframes.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            print("Error: No embeddings found. Run process_folder() first.")
            return []

        print("Calculating similarity graph...")
        # --- 1. Compute all-pairs cosine similarity ---
        # This is our "fully connected graph"
        sim_matrix = cosine_similarity(self.embeddings)
        
        # --- 2. Build and filter the edge list ---
        # We will store edges as (similarity, index_i, index_j)
        edges = []
        num_images = len(self.image_paths)
        
        for i in range(num_images):
            for j in range(i + 1, num_images):
                similarity = sim_matrix[i, j]
                
                # "All edges with similarities above a threshold T are removed"
                if similarity < threshold:
                    edges.append((similarity, i, j))
        
        # --- 3. Sort edges in descending weight order ---
        edges.sort(key=lambda x: x[0], reverse=True)
        
        print(f"Graph has {len(edges)} valid edges (similarity < {threshold}).")

        # --- 4. Pop edges and add images to keyframe set ---
        keyframe_indices = set()
        
        for (sim, i, j) in tqdm(edges, desc="Selecting Keyframes"):
            # Check if we've already reached our target
            if len(keyframe_indices) >= k:
                break
                
            # Add 'i' if it's not already a keyframe
            if i not in keyframe_indices:
                keyframe_indices.add(i)
                
            # Check *again* in case adding 'i' was the last one
            if len(keyframe_indices) >= k:
                break
            
            # Add 'j' if it's not already a keyframe
            if j not in keyframe_indices:
                keyframe_indices.add(j)

        # --- Handle edge case: not enough edges to fill K ---
        # If the graph-based selection didn't yield K images,
        # we fill the remainder with the first images that aren't in the set.
        if len(keyframe_indices) < k:
            print(f"Warning: Only {len(keyframe_indices)} selected by graph. Adding more to reach {k}.")
            for i in range(num_images):
                if i not in keyframe_indices:
                    keyframe_indices.add(i)
                if len(keyframe_indices) >= k:
                    break

        # --- 5. Return the full paths of the selected keyframes ---
        selected_paths = [self.image_paths[i] for i in keyframe_indices]
        
        return selected_paths


# --- This is the runnable part of the script ---
if __name__ == "__main__":
    
    # --- 1. Setup ---
    # <<< CONFIGURE YOUR PATHS HERE >>>
    SOURCE_FOLDER = "./data/pq_snmrtin/"
    KEYFRAME_FOLDER = "./data/keyframes/resnet_pq_snmrtin/"
    
    # <<< CONFIGURE YOUR PARAMETERS HERE >>>
    NUM_KEYFRAMES = 19
    SIMILARITY_THRESHOLD = 0.85 # T=0.85

    # --- 2. Create Output Directory ---
    os.makedirs(KEYFRAME_FOLDER, exist_ok=True)


    # --- 3. Initialize and Process ---
    keyframer = ResNetKeyframer()
    
    # This will process all images and create embeddings
    keyframer.process_folder(SOURCE_FOLDER) 


    # --- 4. Get Keyframes ---
    # This implements the graph logic
    keyframe_paths = keyframer.select_keyframes(
        k=NUM_KEYFRAMES, 
        threshold=SIMILARITY_THRESHOLD
    )


    # --- 5. Copy Files ---
    print(f"\n--- Selected {len(keyframe_paths)} Keyframes ---")

    for img_path in keyframe_paths:
        # img_path is the full source path from the class
        img_filename = os.path.basename(img_path)
        dest_path = os.path.join(KEYFRAME_FOLDER, img_filename)
        
        # Check if file already exists, just in case
        if not os.path.exists(dest_path):
            print(f"Copying {img_filename} to {KEYFRAME_FOLDER}")
            shutil.copy2(img_path, dest_path)
        else:
            print(f"Skipping {img_filename} (already exists)")

    print("\nDone copying keyframes.")