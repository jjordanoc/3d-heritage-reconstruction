import open3d
import numpy as np

def random_percent_simplify(pointcloud: open3d.geometry.PointCloud, percentage: float):
    """
    Randomly keep a percentage of points from the point cloud, preserving colors and normals if present.

    Args:
        pointcloud: Open3D point cloud
        percentage: float in [0,1] representing the fraction of points to keep

    Returns:
        downsampled Open3D point cloud
    """
    points = np.asarray(pointcloud.points)
    num_points_to_keep = max(1, int(len(points) * percentage))  # at least 1 point
    sampled_indices = np.random.choice(len(points), num_points_to_keep, replace=False)

    downsampled_pcd = open3d.geometry.PointCloud()
    downsampled_pcd.points = open3d.utility.Vector3dVector(points[sampled_indices])

    # Preserve colors if they exist
    if pointcloud.has_colors():
        colors = np.asarray(pointcloud.colors)
        downsampled_pcd.colors = open3d.utility.Vector3dVector(colors[sampled_indices])

    # Preserve normals if they exist
    if pointcloud.has_normals():
        normals = np.asarray(pointcloud.normals)
        downsampled_pcd.normals = open3d.utility.Vector3dVector(normals[sampled_indices])

    return downsampled_pcd


if __name__ == "__main__":
    path = "../data/pointclouds/keyframes_full/full.ply"
    pcd = open3d.io.read_point_cloud(path)

    # Randomly keep 5% of points
    down_pcd = random_percent_simplify(pcd, 0.5)

    print("Original points:", len(pcd.points))
    print("Downsampled points:", len(down_pcd.points))

    output_path = "../data/pointclouds/keyframes_full/full_downsampled_random.ply"
    open3d.io.write_point_cloud(output_path, down_pcd)
    print(f"Saved downsampled PLY to: {output_path}")
