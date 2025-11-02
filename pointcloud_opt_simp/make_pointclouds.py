import open3d as o3d
import torch
import numpy
import os

# pointclouds, cam poses
def make_pointclouds(model_output,threshold=0.3):
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

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(img_pts)
        pcd.colors = o3d.utility.Vector3dVector(img_col)
        
        pcds.append(pcd)
        campos.append(cameras[i].numpy())
    return (pcds,campos)





if __name__ == "__main__":
    who = "sample3_results"
    who = who.split(".")[0]
    model_preds = torch.load(f"./data/predictions/{who}.pt")
    os.mkdir(f"./data/pointclouds/{who}")
    pcds, cams = make_pointclouds(model_preds)
    full_pcd = o3d.geometry.PointCloud()
    for i in range(len(pcds)):
        full_pcd = full_pcd + pcds[i]
        o3d.io.write_point_cloud(f"./data/pointclouds/{who}/pc_{i}.ply",pcds[i])
        numpy.save(f"./data/pointclouds/{who}/cam_{i}.npy", cams[i])
    o3d.io.write_point_cloud(f"./data/pointclouds/{who}/pc_full.ply",full_pcd)
