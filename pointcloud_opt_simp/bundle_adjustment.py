import open3d as o3d
import numpy as np
import os
from collections import defaultdict
import heapq # For the Priority Queue
import time # [EDIT] Import time for auto-close

# Define the data directory
# ---!!!--- IMPORTANT ---!!!---
# !! Make sure this path is correct for your system !!
# !! You may need to unzip 'pointclouds.zip' first !!
# ---!!!--- IMPORTANT ---!!!---
data_dir = './data/pointclouds/sample1_results_ply_only' 

# --- Parameters ---
ROTATION_WEIGHT = 1.0 
POSITION_WEIGHT = 30.0
MAX_NEIGHBOR_SCORE = 150.0

# [REGISTRATION] New parameters for ICP
# This is the most important parameter to tune.
# It depends on the scale of your point clouds.
# A good starting point is 2x-5x the resolution of your cloud.
VOXEL_SIZE = 0.02 # 2cm. Was 0.05. A smaller value = less simplification.
ICP_THRESHOLD = VOXEL_SIZE * 1.5

# [REGISTRATION] We will use a fast Point-to-Plane ICP
icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6, 
    relative_rmse=1e-6, 
    max_iteration=30
)
icp_estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()


# --- Rotation Helper ---
def get_rotation_angle(R1, R2):
    """Calculates the angle (in degrees) between two rotation matrices."""
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle_rad)

# --- [REGISTRATION] Helper function for preprocessing ---
def preprocess_cloud(pcd):
    """Downsample, estimate normals, and return a copy."""
    pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
    
    # Estimate normals for Point-to-Plane ICP
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
    )
    return pcd_down

# --- [EDIT] New helper function for non-blocking visualization ---
def show_geometry_autoclose(geometry_list, window_name, duration_ms=1000):
    """Create a window, show geometry, and auto-close after duration_ms."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for geom in geometry_list:
        vis.add_geometry(geom)
    
    # Get the view control and auto-set the camera
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.7) # Zoom out a bit to see everything
    
    start_time = time.time()
    while time.time() - start_time < (duration_ms / 1000.0):
        vis.poll_events()
        vis.update_renderer()
    
    vis.destroy_window()

# --- Load all data ---
pointclouds_local = []
camera_poses = []

print(f"Loading all point clouds and ground-truth poses...")

# Check if data directory exists
# ... existing code ...
if not os.path.isdir(data_dir):
    print(f"Error: Data directory not found at '{data_dir}'")
    print("Please make sure the directory exists and contains the .ply and .npy files.")
    print("You might need to unzip the 'pointclouds.zip' file.")
    exit()

pointcloud_files = sorted([f for f in os.listdir(data_dir) if f.startswith('pc_') and f.endswith('.ply')])
# ... existing code ...
pose_files = sorted([f for f in os.listdir(data_dir) if f.startswith('cam_') and f.endswith('.npy')])

if not pointcloud_files or not pose_files:
# ... existing code ...
    print(f"Error: No .ply or .npy files found in '{data_dir}'")
    print("Please check the directory.")
    exit()

for pc_file, pose_file in zip(pointcloud_files, pose_files):
    # [REGISTRATION] We load the cloud and assume it is ALREADY TRANSFORMED
    pcd_global = o3d.io.read_point_cloud(os.path.join(data_dir, pc_file))
    if not pcd_global.has_colors():
        pcd_global.paint_uniform_color([0.5, 0.5, 0.5])
    
    # We store the *original color* version (assumed global)
    pointclouds_local.append(pcd_global)
    
    pose = np.load(os.path.join(data_dir, pose_file))
# ... existing code ...
    camera_poses.append(pose)

n_pcds = len(pointclouds_local)
print(f"Loaded {n_pcds} clouds (assumed to be in global space).")


# --- 1. Find ALL potential edges and group by node ---
print("Scoring all pairs to find best seed and build graph...")
node_edges = defaultdict(list) 
all_pairs_sorted = [] 

for source_id in range(n_pcds):
# ... existing code ...
    for target_id in range(source_id + 1, n_pcds):
        pose1 = camera_poses[source_id]
        pose2 = camera_poses[target_id]
        
        pos1 = pose1[0:3, 3]; pos2 = pose2[0:3, 3]
# ... existing code ...
        pos_dist = np.linalg.norm(pos1 - pos2)
        R1 = pose1[0:3, 0:3]; R2 = pose2[0:3, 0:3]
        rot_deg = get_rotation_angle(R1, R2)
        score = (pos_dist * POSITION_WEIGHT) + (rot_deg * ROTATION_WEIGHT)
        
        if score < MAX_NEIGHBOR_SCORE:
# ... existing code ...
            all_pairs_sorted.append((score, source_id, target_id))
            node_edges[source_id].append((score, target_id))
            node_edges[target_id].append((score, source_id))

if not all_pairs_sorted:
# ... existing code ...
    print("FATAL: No pairs found. Your MAX_NEIGHBOR_SCORE is too strict.")
    exit()

all_pairs_sorted.sort(key=lambda x: x[0])
# ... existing code ...
best_score, node_x, node_y = all_pairs_sorted[0]

# --- 2. Find the "Best Seed Node" ---
def get_avg_neighbor_strength(node_id):
    """Gets the average score of a node's Top 5 neighbors."""
    neighbors = node_edges[node_id]
    if not neighbors:
        return float('inf') 
    
    neighbors.sort(key=lambda x: x[0]) 
# ... existing code ...
    top_5_scores = [score for score, _ in neighbors[:5]]
    return sum(top_5_scores) / len(top_5_scores)

strength_x = get_avg_neighbor_strength(node_x)
# ... existing code ...
strength_y = get_avg_neighbor_strength(node_y)

start_node = node_x if strength_x <= strength_y else node_y

print(f"\n--- Seeding Algorithm ---")
# ... existing code ...
print(f"Best Pair: ({node_x}, {node_y}) with score {best_score:.4f}")
print(f"Node {node_x} avg neighbor score: {strength_x:.4f}")
print(f"Node {node_y} avg neighbor score: {strength_y:.4f}")
print(f"Starting registration from Node {start_node}.")

# --- 3. Run the Priority Queue (Prim's) Traversal ---

# --- Data Structures ---
visited = set() 
pq = [] # [REGISTRATION] PQ will store (score, node_to_add, node_in_tree)
master_cloud_viz = o3d.geometry.PointCloud() 
step_count = 0

# [REGISTRATION] This dictionary will store our *refined* clouds
# We store copies so we don't modify the originals
refined_clouds = {}

# [REGISTRATION] Modified helper to add source_node_id
# ... existing code ...
def add_neighbors_to_pq(source_node_id):
    """Finds all unvisited neighbors of source_node_id and adds them to PQ."""
    for score, neighbor_id in node_edges[source_node_id]:
        if neighbor_id not in visited:
            # Push (score, node_to_add, node_that_added_it)
            heapq.heappush(pq, (score, neighbor_id, source_node_id))

# --- 4. Initialization ---
print(f"\n--- Starting traversal from Node {start_node} ---")
visited.add(start_node)

# [REGISTRATION] Our new world starts with the first cloud, as-is.
# We create a copy to store in our refined map.
pcd_start = o3d.geometry.PointCloud(pointclouds_local[start_node])
refined_clouds[start_node] = pcd_start
master_cloud_viz += pcd_start

# Show Step 0
print(f"Showing Step 0: Cloud {start_node} (Reference). Close window to begin...")
# [EDIT] Use new auto-close function
show_geometry_autoclose(
    [pcd_start], 
    window_name=f"Step 0: Cloud {start_node}",
    duration_ms=1000
)

# [REGISTRATION] Seed the PQ with the first node's neighbors
add_neighbors_to_pq(start_node)
step_count = 1

# --- 5. Run the Main Loop ---
while pq:
    # 5a. Pop the *best* connection from the PQ
    # [REGISTRATION] We now get the source_node_id as well
    (score, new_node_id, source_node_id) = heapq.heappop(pq)
    
    if new_node_id in visited:
# ... existing code ...
        continue
        
    # 5b. Process this new node
    visited.add(new_node_id)
    
    print(f"\n--- Step {step_count}: Refining Cloud {new_node_id} TO Master Cloud ---")
    print(f"  (Neighbor pair score: {score:.4f})")
    
    # --- [REGISTRATION] This is the new ICP refinement block ---
    
    # Source: The new cloud, which is already in (noisy) global space
    # We make a copy so we can transform it.
    pcd_new_source = o3d.geometry.PointCloud(pointclouds_local[new_node_id])
    pcd_new_source_down = preprocess_cloud(pcd_new_source)

    # Target: The entire master cloud built so far
    # We downsample the *whole thing* for speed.
    pcd_master_target_down = preprocess_cloud(master_cloud_viz)
    
    # Initial Guess: Identity, since we assume they are already close
    T_init = np.identity(4)
    
    print(f"  Running Point-to-Plane ICP...")
    reg_result = o3d.pipelines.registration.registration_icp(
        pcd_new_source_down,         # Source
        pcd_master_target_down,      # Target
        ICP_THRESHOLD,               # Max correspondence distance
        T_init,                      # Initial guess (Identity)
        icp_estimation_method,
        icp_criteria
    )
    
    # This is the small refinement transformation
    refinement_transform = reg_result.transformation
    print(f"  ICP Fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.4f}")
    
    # 5c. Visualize this step
    # Apply the refinement to the *original resolution* new cloud
    pcd_new_refined = pcd_new_source # This is already a copy
    pcd_new_refined.transform(refinement_transform)
    #pcd_new_refined.paint_uniform_color([1, 0, 0]) # Tint new cloud RED
    
    # Create a *copy* of the master cloud and tint it GRAY
    master_cloud_gray = o3d.geometry.PointCloud(master_cloud_viz)
    #master_cloud_gray.paint_uniform_color([0.5, 0.5, 0.5]) 
    
    print(f"  Showing 'master cloud' (GRAY) + refined cloud (RED). Close window...")
    # [EDIT] Use new auto-close function
    show_geometry_autoclose(
        [master_cloud_gray, pcd_new_refined], 
        window_name=f"Step {step_count}: Refined Cloud {new_node_id} (Fitness: {reg_result.fitness:.4f})",
        duration_ms=1000
    )
    
    # 5d. Update: Add the *refined* cloud (with original color) to the master
    pcd_new_final = o3d.geometry.PointCloud(pointclouds_local[new_node_id])
    pcd_new_final.transform(refinement_transform)
    
    master_cloud_viz += pcd_new_final
    refined_clouds[new_node_id] = pcd_new_final # Save it
    
    # 5e. Expand: Add all of the *new node's* neighbors to the PQ
    add_neighbors_to_pq(new_node_id)
    step_count += 1

print("\n--- Priority Queue Refinement complete ---")

# --- 6. Final Visualization ---
print(f"Displaying all {len(visited)} successfully refined point clouds.")
# [EDIT] This one is left as-is, so you can inspect the final result.
o3d.visualization.draw_geometries(
    [master_cloud_viz], 
    window_name="Final Refined Model (from ICP)"
)

if len(visited) < n_pcds:
    print(f"\nWARNING: Graph was not fully connected.")
    print(f"Only {len(visited)}/{n_pcds} clouds were aligned.")

