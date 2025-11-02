import open3d

import numpy as np #for copying pointclouds

def main():
    base = "/home/juan-prochazka/Desktop/Grafica-Proyecto/3d-heritage-reconstruction/pointcloud_opt_simp/data/pointclouds"
    pc1_path = base +"/sample1_results/pc_full.ply"
    pc2_path = base +"/sample3_results/pc_full.ply"

    print("Reading...")
    pc1 = open3d.io.read_point_cloud(pc1_path)
    pc2 = open3d.io.read_point_cloud(pc2_path)
    print("Pre-Viz...")
    open3d.visualization.draw_geometries([pc1,pc2])
    print("Pre-Simplifying...")
    pc1_simple = pc1.uniform_down_sample(500)
    pc2_simple = pc2.uniform_down_sample(500)

    print("Normals and Features...")
    #pre-registration with features
    pc1_simple.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamKNN(knn=100), 
        fast_normal_computation=True
    )

    # Estimate normals for the second point cloud
    pc2_simple.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamKNN(knn=100), 
        fast_normal_computation=True
    )

    #because normals can be 2 different vectors
    pc1_simple.orient_normals_consistent_tangent_plane(k=15)
    pc2_simple.orient_normals_consistent_tangent_plane(k=15)

    #alas, the features
    pc1_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pc1_simple,
                                    open3d.geometry.KDTreeSearchParamKNN(knn=100))
    pc2_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pc2_simple,
                                    open3d.geometry.KDTreeSearchParamKNN(knn=100))
    print("Coarse Registration")

    fgr_option = open3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=0.2,
        iteration_number=64
    )
    rough_reg_res = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pc1_simple, pc2_simple,
        pc1_fpfh, pc2_fpfh,
        option=fgr_option
    )

    print("Rough Registration Viz - Full PC Transformed (Manual Copy)")

    # 1. Manually Copy the whole **FULL-RESOLUTION** point cloud (pc1) data
    #    This ensures you visualize the complete data and respects your manual copy requirement.
    pc1_points = np.asarray(pc1.points)
    pc1_normals = np.asarray(pc1.normals)
    pc1_colors = np.asarray(pc1.colors)

    # 2. Create a new, independent PointCloud object (the deep copy)
    pc1_transformed_full = open3d.geometry.PointCloud()
    pc1_transformed_full.points = open3d.utility.Vector3dVector(pc1_points)
    pc1_transformed_full.normals = open3d.utility.Vector3dVector(pc1_normals)
    pc1_transformed_full.colors = open3d.utility.Vector3dVector(pc1_colors)

    # 3. Apply the transformation from the coarse registration (FGR)
    pc1_transformed_full.transform(rough_reg_res.transformation)

    # 4. Visualize the results: Transformed FULL pc1 vs. Original FULL pc2
    print("Visualizing Coarse Registration Result (Full Resolution)...")
    open3d.visualization.draw_geometries([pc1_transformed_full, pc2],
                                        window_name="Fast Global Registration Result (Full PC)")

main()
