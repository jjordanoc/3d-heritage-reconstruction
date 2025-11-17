#from .. import pipeline
import open3d
import numpy as np

def chamfers(to_compare,ground_truth):
    gt_pcd = open3d.io.read_point_cloud(ground_truth)
    to_compare = open3d.io.read_point_cloud(to_compare)

    dsts1_o3d = gt_pcd.compute_point_cloud_distance(to_compare)
    dsts2_o3d = to_compare.compute_point_cloud_distance(gt_pcd)

    dsts1 = np.asarray(dsts1_o3d)
    dsts2 = np.asarray(dsts2_o3d)

    m1 = np.mean(dsts1**2)
    m2 = np.mean(dsts2**2)

    return m1 + m2

    
a = chamfers("./data/pointclouds/keyframes_saliency/full.ply","./data/pointclouds/keyframes_full/full.ply")
print(a)