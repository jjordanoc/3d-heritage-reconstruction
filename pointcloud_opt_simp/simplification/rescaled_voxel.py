import open3d as o3d
import numpy as np

def rescaled_voxel_simplify(og_voxel_size: float, pointcloud: o3d.geometry.PointCloud):
    """
    og_voxel_size: float in [0,1] representing percentage of chord length
    pointcloud: Open3D point cloud (CPU or CUDA)

    Returns:
        downsampled point cloud
    """
    # Create a copy manually for chord computation
    pcd_for_chord = o3d.geometry.PointCloud()
    pcd_for_chord.points = o3d.utility.Vector3dVector(np.asarray(pointcloud.points))
    if pointcloud.has_colors():
        pcd_for_chord.colors = o3d.utility.Vector3dVector(np.asarray(pointcloud.colors))
    if pointcloud.has_normals():
        pcd_for_chord.normals = o3d.utility.Vector3dVector(np.asarray(pointcloud.normals))

    # Remove outliers ONLY for chord computation
    pcd_for_chord, _ = pcd_for_chord.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    # Oriented bounding box
    obb = pcd_for_chord.get_oriented_bounding_box()
    corners = np.asarray(obb.get_box_points())

    # Compute chord line (longest diagonal)
    max_dist = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            dist = np.linalg.norm(corners[i] - corners[j])
            if dist > max_dist:
                max_dist = dist

    # Compute voxel size relative to chord length
    voxel_size = max_dist * og_voxel_size

    # Voxel-based downsampling on the ORIGINAL pointcloud
    downsampled_pcd = pointcloud.voxel_down_sample(voxel_size=voxel_size)

    return downsampled_pcd


if __name__ == "__main__":
    # Path to your input point cloud
    input_path = "../data/pointclouds/keyframes_full/full.ply"
    pcd = o3d.io.read_point_cloud(input_path)
    print("Original points:", len(pcd.points))

    # Downsample the point cloud using our function
    down_pcd = rescaled_voxel_simplify(0.00065, pcd)
    print("Downsampled points:", len(down_pcd.points))

    # Save the downsampled point cloud
    output_path = "../data/pointclouds/keyframes_full/full_downsampled.ply"
    o3d.io.write_point_cloud(output_path, down_pcd)
    print(f"Saved downsampled PLY to: {output_path}")
