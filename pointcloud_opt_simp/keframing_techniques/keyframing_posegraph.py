import typing
import cv2
import os
import numpy as np
import collections
import networkx as nx
import matplotlib.pyplot as plt
import math
from ortools.graph.python import min_cost_flow

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def build_similarity_graph(imgs: list[np.ndarray], min_inliers: int = 15):
    """
    Builds a similarity graph (view-graph) from a list of images using SIFT.
    ...
    """
    detector = cv2.SIFT_create() 
    all_features = {} 
    
    for i, img in enumerate(imgs):
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- MODIFIED FOR SIFT (2) ---
        # Detect with SIFT
        kp, des = detector.detectAndCompute(gray, None) 
        
        if des is not None and len(kp) > 0:
            # SIFT descriptors must be float32 for FLANN
            all_features[i] = {'kp': kp, 'des': des.astype(np.float32)}

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # or more checks for better accuracy
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    all_good_matches = {}
    image_indices = list(all_features.keys())

    for idx1, i in enumerate(image_indices):
        for j in image_indices[idx1 + 1:]:
            # Use .get() for safety
            feat1 = all_features.get(i)
            feat2 = all_features.get(j)

            # Check for valid features
            if not feat1 or not feat2 or feat1['des'] is None or feat2['des'] is None or \
               len(feat1['des']) < 2 or len(feat2['des']) < 2:
                continue

            try:
                matches = matcher.knnMatch(feat1['des'], feat2['des'], k=2)
            except cv2.error as e:
                print(f"Error matching {i}-{j}: {e}")
                continue

            good_matches_for_pair = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches_for_pair.append(m)
            
            if len(good_matches_for_pair) > min_inliers:
                all_good_matches[(i, j)] = good_matches_for_pair
    graph_nodes = image_indices
    graph_edges = []
    graph_pair_costs = {}
    graph_image_degrees = collections.defaultdict(int)
    graph_image_costs = collections.defaultdict(float)
    
    for (i, j), good_matches in all_good_matches.items():
        
        feat1 = all_features[i]
        feat2 = all_features[j]
        src_pts = np.float32([feat1['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([feat2['kp'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, 
                                         method=cv2.FM_RANSAC, 
                                         ransacReprojThreshold=1.5, 
                                         confidence=0.99)
        
        if mask is None:
            continue
        
        num_inliers = np.sum(mask)

        if num_inliers > min_inliers:
            graph_edges.append((i, j))
            graph_image_degrees[i] += 1
            graph_image_degrees[j] += 1
            graph_pair_costs[(i, j)] = -int(num_inliers)
            graph_image_costs[i] -= num_inliers
            graph_image_costs[j] -= num_inliers

    final_image_costs = {img_id: int(cost) for img_id, cost in graph_image_costs.items()}
    
    for node in graph_nodes:
        if node not in graph_image_degrees:
            graph_image_degrees[node] = 0


    print(f"  Built graph with {len(graph_nodes)} nodes and {len(graph_edges)} edges.")

    return {
        "images": graph_nodes,
        "pairs": graph_edges,
        "pair_costs": graph_pair_costs,
        "image_costs": final_image_costs,
        "image_degrees": graph_image_degrees
    }

def select_keyframes_mcnf(sfm_graph: dict, total_flow: int) -> (set, set):
    """
    Selects an optimal subgraph using the MCNF formulation from 
     A Unified View-Graph Selection Framework for Structure from Motion
     (Rajvi Shah, Visesh Chari, P J Narayanan)
    """
    if total_flow == 0:
        print("  Total flow is 0, selecting 0 keyframes.")
        return set(), set()
        
    # --- THIS IS THE CORRECTED INSTANTIATION ---
    solver = min_cost_flow.SimpleMinCostFlow()

    # --- 1. Node Construction (Figure 2) ---
    node_map_in = {}
    node_map_out = {}
    current_node_index = 0
    
    for img_id in sfm_graph['images']:
        node_map_in[img_id] = current_node_index
        node_map_out[img_id] = current_node_index + 1
        current_node_index += 2

    source_node = current_node_index
    sink_node = current_node_index + 1
    
    # --- 2. Arc Construction (Figure 2) ---
    pair_arc_indices = [] 

    # Add arcs for each image (Source, Image Arc, Sink)
    for img_id in sfm_graph['images']:
        node_in = node_map_in[img_id]
        node_out = node_map_out[img_id]
        cost = int(sfm_graph['image_costs'].get(img_id, 0))
        capacity = sfm_graph['image_degrees'].get(img_id, 0)
        
        # Arc: Source -> v_in
        solver.add_arc_with_capacity_and_unit_cost(
            tail=source_node, head=node_in, capacity=capacity, unit_cost=0
        )
        # Arc: v_in -> v_out (The "image arc" e_i)
        solver.add_arc_with_capacity_and_unit_cost(
            tail=node_in, head=node_out, capacity=capacity, unit_cost=cost
        )
        # Arc: v_out -> Sink
        solver.add_arc_with_capacity_and_unit_cost(
            tail=node_out, head=sink_node, capacity=capacity, unit_cost=0
        )

    # Add arcs for each pair (The "pairwise arc" e_ij)
    for (i, j) in sfm_graph['pairs']:
        cost = int(sfm_graph['pair_costs'].get((i, j), 0))
        arc_index = solver.add_arc_with_capacity_and_unit_cost(
            tail=node_map_out[i], head=node_map_in[j], 
            capacity=1, unit_cost=cost
        )
        pair_arc_indices.append((arc_index, (i, j)))

    # --- 3. Set Supply and Demand ---
    solver.set_node_supply(source_node, total_flow)
    solver.set_node_supply(sink_node, -total_flow)

    # --- 4. Solve ---
    status = solver.solve()

    selected_pairs = set()
    selected_images = set()

    if status == solver.OPTIMAL:
        for arc_index, (i, j) in pair_arc_indices:
            if solver.flow(arc_index) > 0:
                selected_pairs.add((i, j))
                selected_images.add(i)
                selected_images.add(j)
    else:
        # --- MORE INFORMATIVE STATUS ---
        status_name = "UNKNOWN"
        if status == solver.INFEASIBLE: status_name = "INFEASIBLE"
        elif status == solver.UNBALANCED: status_name = "UNBALANCED"
        elif status == solver.NOT_SOLVED: status_name = "NOT_SOLVED"
        
    return selected_images, selected_pairs

def find_keyframes(imgs,flow):
    graph = build_similarity_graph(imgs)
    print(f"Find keyframes with flow {flow}")
    selected, _ = select_keyframes_mcnf(graph,flow)
    return [imgs[i] for i in selected]

def search_flow_for_target(imgs, target_kf):
    low, high = 1, len(imgs)
    
    best_imgs = None
    best_diff = float("inf")
    best_flow = None

    while low <= high:
        flow = (low + high) // 2
        kf = find_keyframes(imgs, flow)
        count = len(kf)
        diff = abs(count - target_kf)

        print(f"[search] flow={flow}  →  {count} keyframes")

        # Update best match
        if diff < best_diff:
            best_diff = diff
            best_imgs = kf
            best_flow = flow

        if count > target_kf:
            high = flow - 1
        elif count < target_kf:
            low = flow + 1
        else:
            # perfect match
            break

    print(f"Best flow = {best_flow} → {len(best_imgs)} keyframes (target={target_kf})")
    return best_imgs

def find_keyframes_with_graph(graph, flow):
    """Use precomputed graph to select keyframes for a given flow."""
    print(f"Find keyframes with flow {flow}")
    selected, _ = select_keyframes_mcnf(graph, flow)
    return selected

def search_flow_for_target_with_graph(graph, imgs, target_kf):
    """Binary search for best flow while reusing the graph."""
    low, high = 1, len(imgs)
    best_selected = None
    best_diff = float("inf")
    best_flow = None

    while low <= high:
        flow = (low + high) // 2
        selected = find_keyframes_with_graph(graph, flow)
        count = len(selected)
        diff = abs(count - target_kf)

        print(f"[search] flow={flow} → {count} keyframes")

        # update best match
        if diff < best_diff:
            best_diff = diff
            best_selected = selected
            best_flow = flow

        if count > target_kf:
            high = flow - 1
        elif count < target_kf:
            low = flow + 1
        else:
            break  # perfect match

    print(f"Best flow = {best_flow} → {len(best_selected)} keyframes (target={target_kf})")
    return [imgs[i] for i in best_selected]


if __name__ == "__main__":
    dir = "./data/pq_snmrtin/"
    imgs = [cv2.imread(dir + p) for p in sorted(os.listdir(dir))]

    print("Building similarity graph once...")
    graph = build_similarity_graph(imgs)  # compute SIFT + matches only once

    print("Finding keyframes targeting EXACTLY 19...")
    keyframes = search_flow_for_target_with_graph(graph, imgs, target_kf=19)
    print(f"{len(keyframes)} keyframes returned")

    outdir = "./data/keyframes/mcnf_pq_snmrtin/"
    os.makedirs(outdir, exist_ok=True)

    for i, img in enumerate(keyframes):
        cv2.imwrite(f"{outdir}/img_{i}.png", img)