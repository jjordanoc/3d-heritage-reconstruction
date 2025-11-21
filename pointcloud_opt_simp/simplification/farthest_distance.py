import open3d as o3d
import numpy as np

def farthest_point_simplify(pointcloud: o3d.geometry.PointCloud, percent: float):
    """
    Simplifies a point cloud using Open3D's native Farthest Point Sampling.

    Args:
        pointcloud: Open3D point cloud
        percent: Percentage of points to keep (0-1)

    Returns:
        downsampled_pcd: Simplified Open3D point cloud
    """
    # Calculate the target number of points
    N = len(pointcloud.points)
    n_points = max(1, int(percent * N))
    
    # Use the built-in C++ implementation
    downsampled_pcd = pointcloud.farthest_point_down_sample(num_samples=n_points)
    
    return downsampled_pcd


if __name__ == "__main__":
    # Ruta de ejemplo
    input_path = "../data/pointclouds/keyframes_full/full.ply"
    pcd = o3d.io.read_point_cloud(input_path)
    print("Original points:", len(pcd.points))

    # Porcentaje de puntos a mantener
    percent = 0.11  # por ejemplo 1% de la nube
    down_pcd = farthest_point_simplify(pcd, percent)
    print("Downsampled points:", len(down_pcd.points))

    # Guardar
    output_path = "../data/pointclouds/keyframes_full/full_downsampled_fps.ply"
    o3d.io.write_point_cloud(output_path, down_pcd)
    print(f"Saved downsampled PLY to: {output_path}")
