import numpy as np
import plotly.graph_objects as go
import imageio.v2 as iio
import glob
import random

# --------------------------
# Paths
# --------------------------
depth_files = sorted(glob.glob("reconstruction/rec/depth/*.npy"))
conf_files  = sorted(glob.glob("reconstruction/rec/conf/*.npy"))
color_files = sorted(glob.glob("reconstruction/rec/color/*.png"))
camera_files = sorted(glob.glob("reconstruction/rec/camera/*.npz"))

# Pick 20 random frames
indices = random.sample(range(len(depth_files)), min(5, len(depth_files)))

all_points = []
all_colors = []
camera_positions = []
camera_poses = []

for i in indices:
    # Load files
    depth = np.load(depth_files[i])
    conf = np.load(conf_files[i])
    color = iio.imread(color_files[i]) / 255.0
    with np.load(camera_files[i]) as cam:
        pose = cam['pose']
        intrinsics = cam['intrinsics']

    H, W = depth.shape
    mask = conf > 0.5
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))

    # Pixels -> camera coordinates
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]

    Xc = (xs - cx) * depth / fx
    Yc = (ys - cy) * depth / fy
    Zc = depth
    points_cam = np.stack([Xc, Yc, Zc], axis=-1).reshape(-1,3)
    cols = color.reshape(-1,3)

    # Keep only confident points
    points_cam = points_cam[mask.reshape(-1)]
    cols = cols[mask.reshape(-1)]

    # Camera -> world coordinates
    ones = np.ones((points_cam.shape[0],1))
    points_cam_h = np.hstack([points_cam, ones])
    points_world_h = points_cam_h @ pose.T
    points_world = points_world_h[:,:3]

    all_points.append(points_world)
    all_colors.append(cols)
    camera_positions.append(pose[:3,3])
    camera_poses.append(pose)

# Concatenate all points
all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)

# --------------------------
# Plot point cloud
# --------------------------
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=all_points[:,0],
    y=all_points[:,1],
    z=all_points[:,2],
    mode='markers',
    marker=dict(size=1, color=all_colors, opacity=0.8),
    name='Points'
))

# --------------------------
# Plot camera positions
# --------------------------
for cam_pos, cam_pose in zip(camera_positions, camera_poses):
    fig.add_trace(go.Scatter3d(
        x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Camera'
    ))

    # Optional: draw a simple camera frustum (lines from camera center along z-axis)
    scale = 0.1  # adjust for visualization
    z_axis = cam_pose[:3,2]  # camera forward
    line_end = cam_pos + z_axis * scale
    fig.add_trace(go.Scatter3d(
        x=[cam_pos[0], line_end[0]],
        y=[cam_pos[1], line_end[1]],
        z=[cam_pos[2], line_end[2]],
        mode='lines',
        line=dict(color='red', width=2),
        name='Camera Frustum'
    ))

# --------------------------
# Layout
# --------------------------
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    title="3D Point Cloud (20 Random Frames)"
)

fig.show()
