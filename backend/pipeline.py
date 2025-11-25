
#from modal_inference_function import PI3Model


### TEST FUNCTIONS 

def run_inference(tensor_imgs):
    pi3_obj = modal.Cls.from_name("model_inference_ramtensors", "PI3Model")
    pi3Model = pi3_obj() #instantiate the class
    predictions = pi3Model.run_inference.remote(tensor_imgs)
    print("Inference complete. Received results locally.")
    return predictions

def tensorize_images(images):
    transform = transforms.ToTensor()
    tensors = torch.stack([transform(img) for img in images],dim=0)
    return tensors

def do_inference(images):
    tensor_imgs = tensorize_images(images)
    inference = run_inference(tensor_imgs)
    return inference

def unify_pointcloud(pointclouds: list[open3d.geometry.PointCloud]) -> open3d.geometry.PointCloud:
    full_pc = open3d.geometry.PointCloud()
    for pc in pointclouds:
        full_pc += pc
    return full_pc

def do_registration(new_ply: open3d.geometry.PointCloud,
                    last_ply: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
    """

    Computes the appropiate transformation matrix to take the newest inference
    (concat(pointclouds)) to the old one's (last_ply) coordinate system.
    Arguments:
    pointclouds: List of latest pointclouds per image
    last_ply: point cloud of the newest inference 
    """
    NORMAL_EST_KNN = 100
    DOWNSAMPLE = 500
    TANGENT_PLANE_K = 15
    FPFH_KNN = 100
    MAX_CORR_DST_FGR = 0.2
    FGR_ITERS = 64

    # simplify plys
    pc1_simple = new_ply.uniform_down_sample(DOWNSAMPLE)
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

    return rough_reg_res.transformation

def tensor_to_pointcloud(model_output,threshold=0.3):
    pts = model_output["points"].squeeze(0)
    colors = model_output["images"].squeeze(0)
    confidence = model_output["conf"].squeeze(0)
    cameras = model_output["camera_poses"].squeeze(0)


    pcds = []
    campos = []

       
    for i in range(pts.shape[0]):
        # from x*y*3  tensor to (x*y)*3 to numpy
        image_pts_raw = pts[i]
        image_colors_raw = colors[i]
        image_conf_raw = confidence[i]
        flat_pts = image_pts_raw.reshape(-1, 3)
        flat_colors = image_colors_raw.reshape(-1, 3)
        flat_confidence = image_conf_raw.reshape(-1, 1)
        img_pts = flat_pts.numpy()
        img_col = flat_colors.numpy()
        img_conf = flat_confidence.numpy()

        #thresholding - building the threshold
        threshmask = img_conf > threshold
        threshmask = numpy.squeeze(threshmask)

        #actually thresholding
        #print(threshmask)
        img_pts = img_pts[threshmask]
        img_col = img_col[threshmask]

        pcd = open3d.geometry.PointCloud()

        pcd.points = open3d.utility.Vector3dVector(img_pts)
        pcd.colors = open3d.utility.Vector3dVector(img_col)
        
        pcds.append(pcd)
        campos.append(cameras[i].numpy())
    return (pcds,campos)

def run_pipeline(images,last_ply):
    """
    Arguments:
    Images: PIL.Image list of images to run inference on
    last_ply: An optional last ply file to wich the new infered pointcloud will be registered
    Returns:
    unified,pointclouds
    """
    result_tensor = do_inference(images)
    pointclouds, cameras = tensor_to_pointcloud(result_tensor)
    unified = unify_pointcloud(pointclouds)
    if (last_ply is not None):
        tf = do_registration(unified,last_ply)

        for pcd in pointclouds:
            pcd.transform(tf)
        unified.transform(tf)

        for i, cam in enumerate(cameras):
            cameras[i] = tf @ cam
    
    return unified,pointclouds,cameras

def read_images(path:str,new_id:str,PIXEL_LIMIT=255000) -> list[Image.Image]:
    """
    Reads a directory's worth of images.. Makes sure that the last image in the returned list
    is the image with name new_id. new_id may be the file name or the file name with extension.
    Arguments:
    path: A path to a directory with images. 
    new_id: A file name with or without extension residing in that directory
    PIXEL_LIMIT: a limit for the ammount of pixels sent to the model

    Returns:
    list[Image.Image]: a list of PIL images, where the last image is the image with file name new_id
    """
    sources = []
    filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
    #Make sure new_id is the last image
    new_id_filename = next((name for name in filenames 
                            if os.path.splitext(name)[0] == new_id or name == new_id), 
                        None)
    if new_id_filename:
        filenames.remove(new_id_filename)
        filenames.append(new_id_filename)
    else:
        raise Exception("new_id no existe en el directorio especificado")
    shape = 500
    for i in range(0, len(filenames)):
        img_path = os.path.join(path, filenames[i])
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except:
            print("Failed to load image {filenames[i]}")
    #resize (copied from PI3)
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    image_list = []
    
    for img_pil in sources[0:-1]: #avoid last image, if exception ocurs there the pipeline should fail
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            image_list.append(resized_img)
        except Exception as e:
            print(f"Error processing an image: {e}")
    #process last image and dont catch
    resized_img = sources[-1].resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    image_list.append(resized_img)

    return image_list

def addImageToCollection(inPath,oldPly,new_id,outputs_directory = "./data/pointclouds",save_all_shards=False):
    """
    Arguments
    inPath: Path to a folder containing the images
    oldPly: (optional) path pointing to a PLY file of the last full scene to perform registration.
            if None is passed registration is ommited.
    new_id: string corresponding to the latest image
    outputs_directory: The directory in wich to save the outputs
    save_all_shards: If True, save individual PLY files for all images (for full refetch)
    
    returns: a path to a directory containing at least the latest infered pointcloud and {i}.ply the pointcloud 
            for the new_id (latest) image.
    
    Internally,
    Performs inference on a given path, 
    Performs conversion to pointcloud
    Serializes the point clouds
    Registers the new point cloud to the old point cloud to allow for online addition
    """
    new_path = outputs_directory + "/latest"
    #if there is someone currently on the latest path, i must move them
    if os.path.exists(new_path):
        #each old ply bears the name of he who obsoleted them, as an eternal reminder
        #of our mortality and how age comes for us all
        old_path = outputs_directory + f'/{new_id.split(".")[0]}'
        os.rename(new_path,old_path)
        os.mkdir(new_path)

    images = read_images(inPath,new_id)

    last_ply = None
    if oldPly is not None:
        last_ply = open3d.io.read_point_cloud(oldPly)

    unified, pcds, cam_estimates = run_pipeline(images,last_ply)
    open3d.io.write_point_cloud(new_path + "/full.ply",unified)
    open3d.io.write_point_cloud(new_path + f'/{new_id.split(".")[0]}.ply',pcds[-1])
    
    # For full refetch, save all individual point clouds as shards
    if save_all_shards:
        for i, pcd in enumerate(pcds):
            shard_path = new_path + f"/{i:04d}.ply"
            open3d.io.write_point_cloud(shard_path, pcd)
    
    return new_path


# def main():
#     addImageToCollection("./data/sample_ip1/","./data/pointclouds/pipeline_results1/latest.ply","0078")

# if __name__ == "__main__":
#     main()