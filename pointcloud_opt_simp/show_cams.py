import os
import glob
import numpy as np
import open3d as o3d
import cv2


def read_cam_list(folder):
    order_file = os.path.join(folder, "cam_order.txt")
    if os.path.exists(order_file):
        with open(order_file, "r") as f:
            cams = [line.strip() for line in f.readlines() if line.strip()]
        print(f"[INFO] Loaded {len(cams)} image names from cam_order.txt")
        return cams
    else:
        print("[WARN] cam_order.txt not found")
        return None


def create_camera_frustum(scale=0.1, color=[1, 0, 0]):
    pts = np.array([
        [0, 0, 0],
        [-1, -0.75, 2],
        [ 1, -0.75, 2],
        [ 1,  0.75, 2],
        [-1,  0.75, 2],
    ]) * scale

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    fr = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    fr.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return fr


def visualize_with_images(recon_folder, images_folder):
    # ---------------------------------------------------
    # Load point cloud
    # ---------------------------------------------------
    ply_path = os.path.join(recon_folder, "full.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"full.ply not found: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[INFO] Loaded PLY: {ply_path}")

    # ---------------------------------------------------
    # Load extrinsics
    # ---------------------------------------------------
    extrinsic_files = sorted(glob.glob(os.path.join(recon_folder, "*.npy")))
    extrinsic_files = [f for f in extrinsic_files if "cam_order" not in f]

    if not extrinsic_files:
        raise RuntimeError("No *.npy extrinsics found")

    print(f"[INFO] Found {len(extrinsic_files)} camera extrinsics")

    # ---------------------------------------------------
    # Load cam_order.txt
    # ---------------------------------------------------
    cam_names = read_cam_list(recon_folder)
    if cam_names is None:
        cam_names = [f"{i:04d}.png" for i in range(len(extrinsic_files))]

    # ---------------------------------------------------
    # Create frustums
    # ---------------------------------------------------
    base_colors = [
        [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [1,0,1], [0,1,1],
        [1,0.5,0]
    ]

    geoms = [pcd]
    frustums = []

    for i, Tfile in enumerate(extrinsic_files):
        T = np.load(Tfile)
        color = base_colors[i % len(base_colors)]
        fr = create_camera_frustum(scale=0.12, color=color)
        fr.transform(T)
        frustums.append(fr)
        geoms.append(fr)

    # ---------------------------------------------------
    # Key navigation for images
    # ---------------------------------------------------
    current_cam = {"idx": 0}

    def show_image():
        idx = current_cam["idx"]
        img_name = cam_names[idx]

        img_path = os.path.join(images_folder, img_name)
        print(f"[INFO] Trying: {img_path}")

        if not os.path.exists(img_path):
            print(f"[ERR] Image not found: {img_path}")
            return

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERR] Could not load image: {img_path}")
            return

        cv2.imshow("camera image", img)
        cv2.waitKey(1)

    def on_key(vis, key, mod):
        if key == ord("n"):
            current_cam["idx"] = (current_cam["idx"] + 1) % len(frustums)
            print(f"[INFO] Camera {current_cam['idx']}")
            show_image()
        elif key == ord("p"):
            current_cam["idx"] = (current_cam["idx"] - 1) % len(frustums)
            print(f"[INFO] Camera {current_cam['idx']}")
            show_image()
        return False

    print("\n[INFO] Controls:")
    print("  'n' → next camera + show corresponding image")
    print("  'p' → previous camera + show corresponding image")
    print("-----------------------------------------------------\n")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Reconstruction Viewer")

    for g in geoms:
        vis.add_geometry(g)

    vis.register_key_callback(ord("n"), on_key)
    vis.register_key_callback(ord("p"), on_key)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # EDIT THESE TWO PATHS
    reconstruction_folder = "./data/pointclouds/keyframes_random"
    images_folder = "./data/clean_data/"

    visualize_with_images(reconstruction_folder, images_folder)
