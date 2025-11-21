import cv2
import numpy as np
import os
import glob
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from tqdm import tqdm
import logging
import shutil
import os

# Suppress divide-by-zero warnings that can happen
# when the vocabulary is very small (e.g., size 1)
np.seterr(divide='ignore', invalid='ignore')

class OnlineSaliencyBoW:
    """
    Implements the online saliency algorithm from the provided paper
    (akim-2013a.pdf) [cite: 1717], including the correct normalization.
    """

    def __init__(self, distance_threshold=75):
        """
        Initializes the state.
        
        Args:
            distance_threshold (int): The Hamming distance threshold for 
                                      considering a feature a "new word".
        """
        # Using ORB as a patent-free alternative to SURF [cite: 1488]
        self.detector = cv2.ORB_create(nfeatures=500)
        
        # --- Vocabulary and Statistics ---
        # W(t): Vocabulary (list of feature descriptors) [cite: 1488]
        self.W = np.empty((0, 32), dtype=np.uint8) 
        # N(t): Total number of non-overlapping images [cite: 1488]
        self.N = 0  
        # nw(t): Occurrence of each word k in the database [cite: 1488]
        self.nw = np.array([], dtype=np.int32) 
        
        # --- Internal Data Storage ---
        self.image_data = [] 
        
        # --- Parameters ---
        self.distance_threshold = distance_threshold

        # --- Saliency Results ---
        self.saliency_scores = {}
        
        # --- Normalization trackers ---
        # R_max: Maximum rarity score encountered 
        self.R_max = 0.0


    def _extract_features(self, image_path):
        """
        Loads an image, pre-blurs, and extracts ORB features.
        The pre-blurring step is mentioned in the paper[cite: 1488, 1368].
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 0
        
        # Preblur [cite: 1488]
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Extract features
        keypoints, descriptors = self.detector.detectAndCompute(img_blur, None)
        
        if descriptors is None or len(descriptors) == 0:
            return None, 0
            
        return descriptors, len(descriptors)

    def _update_all_saliency_scores(self):
        """
        Re-calculates saliency for ALL images based on the paper's
        normalized equations[cite: 1396, 1446].
        """
        if self.N == 0:
            return

        vocab_size = len(self.W)
        if vocab_size == 0:
            return

        # Pre-calculate all IDF scores
        # idf(w) = log(N(t) / nw(t))
        # Add 1 to N and nw to avoid division by zero (Laplace smoothing)
        idf_scores = np.log((self.N + 1) / (self.nw + 1))
        
        # We need to do one pass to find the new R_max
        new_R_max = 0.0
        temp_rarity_scores = []
        
        all_pi = []
        all_histograms = []
        all_feature_counts = []

        for item in self.image_data:
            Hi = item['histogram']
            nf = item['feature_count']
            all_histograms.append(Hi)
            all_feature_counts.append(nf)
            
            if nf == 0:
                all_pi.append(np.array([]))
                temp_rarity_scores.append(0.0)
                continue
                
            # Compute BoW distribution: pi(w) = Hi(w) / nf [cite: 1489]
            pi = Hi / nf
            all_pi.append(pi)

            # --- Calculate Rarity R_i(t) (Eqn. 4 - implicit) ---
            # R_i(t) is the sum of (pi(w) * idf(w))
            active_idf_scores = idf_scores[Hi > 0]
            active_pi = pi[Hi > 0]
            Ri = np.sum(active_pi * active_idf_scores)
            temp_rarity_scores.append(Ri)

        # Find the new maximum rarity score from this batch
        if temp_rarity_scores:
            new_R_max = np.max(temp_rarity_scores)
            
        # Update the global maximum rarity 
        self.R_max = max(self.R_max, new_R_max)

        # --- Second pass to calculate normalized scores ---
        for i in range(len(self.image_data)):
            img_path = self.image_data[i]['path']
            Hi = all_histograms[i]
            nf = all_feature_counts[i]
            pi = all_pi[i]

            if nf == 0 or len(pi) == 0:
                self.saliency_scores[img_path] = {'local': 0, 'global': 0}
                continue

            # --- Local Saliency (Eqn. 2 & 3) ---
            pi_nz = pi[pi > 0]
            # Eqn (2): H_i = Shannon Entropy 
            HLi = entropy(pi_nz, base=2)
            
            # Eqn (3): S_Li = H_i / log2(W(t)) 
            # We check if vocab_size > 1 to avoid log2(1) = 0
            max_entropy = np.log2(vocab_size)
            SLi = 0.0
            if max_entropy > 0:
                SLi = HLi / max_entropy
            
            # --- Global Saliency (Eqn. 5) ---
            Ri = temp_rarity_scores[i]
            
            # Eqn (5): S_Gi(t) = R_i(t) / R_max 
            SGi = 0.0
            if self.R_max > 0:
                SGi = Ri / self.R_max

            self.saliency_scores[img_path] = {'local': SLi, 'global': SGi}
            
            # Store calculated values
            self.image_data[i]['pi'] = pi
            self.image_data[i]['SLi'] = SLi
            self.image_data[i]['SGi'] = SGi


    def add_image(self, image_path):
        """
        Processes a single image and updates the model according to Algorithm 1[cite: 1488].
        """
        
        # Preblur and extract SURF features [cite: 1488]
        Fi, nf = self._extract_features(image_path)
        
        if Fi is None:
            logging.warning(f"Could not extract features from {image_path}. Skipping.")
            return

        # --- Compute intra-image BoW statistics ---
        vocab_size = len(self.W)
        Hi = np.zeros(vocab_size, dtype=np.int32) # [cite: 1488]
        
        vocab_updated = False
        new_word_indices = []

        for fj in Fi:
            best_match_idx = -1
            min_dist = float('inf')
            
            if len(self.W) > 0:
                # Find best vocabulary match wk [cite: 1488]
                fj_r = fj.reshape(1, -1)
                distances = cdist(self.W, fj_r, 'hamming') * 256
                min_dist = np.min(distances)
                best_match_idx = np.argmin(distances)

            # if projection fj · wk > threshold then {augment vocab.} [cite: 1488]
            # (Interpreting dot product as similarity, so low distance
            # is a match. High distance > threshold is a new word)
            if min_dist > self.distance_threshold:
                # W(t) ← [W(t), fj ] [cite: 1488]
                self.W = np.vstack([self.W, fj])
                best_match_idx = len(self.W) - 1
                # nwk (t) ← 1 (will be set to 1 later) [cite: 1488]
                self.nw = np.append(self.nw, 0)
                
                vocab_updated = True
                new_word_indices.append(best_match_idx)
                
                # Resize all existing histograms
                Hi = np.append(Hi, 0)
                for item in self.image_data:
                    item['histogram'] = np.append(item['histogram'], 0)
            
            # increment histogram: Hi(wk) [cite: 1488]
            Hi[best_match_idx] += 1

        # --- Update inter-image idf statistics ---
        stats_updated = False
        # (Assuming no overlap [cite: 1489])
        # Increment the document database: N(t) [cite: 1489]
        self.N += 1
        stats_updated = True
        
        present_word_indices = np.where(Hi > 0)[0]
        all_updated_indices = np.unique(np.concatenate((present_word_indices, new_word_indices)))
        
        for wk_idx in all_updated_indices:
            if Hi[wk_idx] > 0:
                # increment word occurrence: nwk(t) [cite: 1489]
                self.nw[wk_idx] += 1

        self.image_data.append({
            'path': image_path,
            'histogram': Hi,
            'feature_count': nf
        })
        
        # --- Saliency Calculation ---
        # if W(t) was updated or stats changed, update all [cite: 1490, 1492]
        if vocab_updated or stats_updated:
            self.saliency_scores = {} # Clear old scores
            self._update_all_saliency_scores()


    def process_folder(self, folder_path):
        """
        Iterates over a folder of images and processes them one by one.
        """
        image_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
        image_paths.extend(sorted(glob.glob(os.path.join(folder_path, '*.png'))))
        
        print(f"Found {len(image_paths)} images. Processing...")
        
        for img_path in tqdm(image_paths, desc="Processing Images"):
            self.add_image(img_path)
            
        print("Processing complete.")
        print(f"  Final vocabulary size: {len(self.W)} words")
        print(f"  Total images processed: {self.N}")


    def get_keyframes(self, top_k=10, strategy='global'):
        """
        Sorts the images by their saliency scores and returns the top K.

        Args:
            top_k (int): The number of keyframes to return.
            strategy (str): 'global', 'local', or 'combined'
        
        Returns:
            list: A list of (image_path, score) tuples.
        """
        if not self.saliency_scores:
            print("No saliency scores calculated. Run process_folder() first.")
            return []

        def get_sort_key(item):
            scores = item[1] # (path, {'local': SLi, 'global': SGi})
            if strategy == 'global':
                return scores.get('global', 0)
            elif strategy == 'local':
                return scores.get('local', 0)
            elif strategy == 'combined':
                return scores.get('local', 0) * scores.get('global', 0)
            else:
                return scores.get('global', 0)

        sorted_images = sorted(self.saliency_scores.items(), 
                               key=get_sort_key, 
                               reverse=True)
        
        return sorted_images[:top_k]
    
# --- Setup ---
SOURCE_FOLDER = "./data/pq_snmrtin/"
KEYFRAME_FOLDER = "./data/keyframes/saliency_pq_snmrtin/"
NUM_KEYFRAMES = 19

# Create the output folder if it doesn't exist
os.makedirs(KEYFRAME_FOLDER, exist_ok=True)

# --- Process ---
saliency = OnlineSaliencyBoW()
saliency.process_folder(SOURCE_FOLDER)

# --- Get Keyframes ---
# 'imgs' will be a list of tuples: [(path1, scores1), (path2, scores2), ...]
imgs = saliency.get_keyframes(NUM_KEYFRAMES, "combined")

print(f"--- Top {NUM_KEYFRAMES} Keyframes (Combined Saliency) ---")
print(imgs) # This will print the full list of tuples

# --- Corrected Loop ---
# We need to get the path (item[0]) from each tuple in the list
for i, (image_path, scores) in enumerate(imgs):
    # image_path is the full path, so we just need its base filename
    img_filename = os.path.basename(image_path)
    
    # Construct the full source and destination paths
    source_path = os.path.join(SOURCE_FOLDER, img_filename)
    dest_path = os.path.join(KEYFRAME_FOLDER, img_filename)
    
    print(f"Copying {img_filename} to {KEYFRAME_FOLDER}")
    shutil.copy2(source_path, dest_path)

print("\nDone copying keyframes.")