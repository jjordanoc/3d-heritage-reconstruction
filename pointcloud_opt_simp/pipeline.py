import open3d
import typing


### TEST FUNCTIONS 

def do_inference(images):
    pass

def to_simplified_pc(pointclouds: list[open3d.geometry.PointCloud]) -> open3d.geometry.PointCloud:
    full_pc = open3d.geometry.PointCloud()
    for pc in pointclouds:
        full_pc += pc
    return full_pc

def do_registration(pointcloudspointclouds: list[open3d.geometry.PointCloud],
                    last_ply: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
    """

    Computes the appropiate transformation matrix to take the newest inference
    (pc1) to the old one's coordinate system
    """
    NORMAL_EST_KNN = 100
    DOWNSAMPLE = 500
    TANGENT_PLANE_K = 15
    FPFH_KNN = 100
    MAX_CORR_DST_FGR = 0.2
    FGR_ITERS = 64

    # simplify plys
    pc1_simple = to_simplified_pc(pointclouds)
    pc2_simple = last_ply.uniform_down_sample(DOWNSAMPLE)

    #calculate required RANSAC stuff:

    #Normal estimation
    pc1_simple.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_EST_KNN), 
        fast_normal_computation=True
    )
    pc2_simple.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_EST_KNN), 
        fast_normal_computation=True
    )
    # Normal direction regularization
    pc1_simple.orient_normals_consistent_tangent_plane(k=TANGENT_PLANE_K)
    pc2_simple.orient_normals_consistent_tangent_plane(k=TANGENT_PLANE_K)

    # FPFH Feature Calculation
    pc1_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pc1_simple,
                                    open3d.geometry.KDTreeSearchParamKNN(knn=FPFH_KNN))
    pc2_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pc2_simple,
                                    open3d.geometry.KDTreeSearchParamKNN(knn=FPFH_KNN))

    fgr_option = open3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=MAX_CORR_DST_FGR,
        iteration_number=FGR_ITERS
    )

    rough_reg_res = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pc1_simple, pc2_simple,
        pc1_fpfh, pc2_fpfh,
        option=fgr_option
    )

    return 

def tensor_to_pointcloud(result_tensor):
    return result_tensor

def run_pipeline(images,last_ply):
    result_tensor = do_inference(images)
    pointclouds = tensor_to_pointcloud(result_tensor)
    if (last_ply is not None):
        transform = do_registration(pointclouds,last_ply)
    return None,None

def read_images(inPath,new_id):
    return []

def addImageToCollection(inPath,oldPly,new_id):
    """
    Arguments
    inPath: Path to a folder containing the images
    oldPly: (optional) path pointing to a PLY file of the last full scene to perform registration.
            if None is passed registration is ommited.
    new_id: string corresponding to the latest image

    returns: a path to a directory containing at least the latest infered pointcloud and {i}.ply the pointcloud 
            for the new_id (latest) image.
    
    Internally,
    Performs inference on a given path, 
    Performs conversion to pointcloud
    Serializes the point clouds
    Registers the new point cloud to the old point cloud to allow for online addition
    """
    new_path = ""
    images = read_images(inPath,new_id)
    last_ply = open3d.io.read_point_cloud(oldPly)
    point_estimates, cam_estimates = run_pipeline(images,last_ply)
    open3d.io.save_point_cloud(new_path + "/latest.ply")
    open3d.io.save_point_cloud(new_path + "/{i}.ply")


if __name__ == "__main__":
    import os
    import shutil
    for i