import modal
import sys

# Modal App Configuration
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "roma",
        "gradio",
        "matplotlib",
        "tqdm",
        "opencv-python-headless",
        "scipy",
        "einops",
        "gdown",
        "trimesh",
        "pyglet<2",
        "huggingface-hub[torch]>=0.22",
        "evo",
    )
    .pip_install("imageio", "pillow")
    .apt_install("git")
    .run_commands("git clone --recursive https://github.com/junyi42/monst3r", "sed -i 's/opencv-python/opencv-python-headless/g' monst3r/requirements.txt", "cd monst3r && pip install -r requirements.txt")

)

app = modal.App("monst3r-demo", image=image)

volume = modal.Volume.from_name(
    "v0", create_if_missing=True
)

# This is the path inside the container where we will clone the repo and store data.
REMOTE_PROJECT_ROOT = "/root/monst3r_project"



def bootstrap_monst3r_repo():
    """Clones the MonST3R repo into the container if it doesn't exist."""
    import os
    if not os.path.exists(REMOTE_PROJECT_ROOT):
        print("--> Cloning MonST3R repository...")
        os.system(f"git clone --recursive https://github.com/junyi42/monst3r.git {REMOTE_PROJECT_ROOT}")
        print("--> Clone complete.")
    else:
        print("--> MonST3R repository already exists.")

    # Add the cloned repo to Python's path
    if REMOTE_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_ROOT)

    # We also need to add the dust3r submodule to the path
    dust3r_path = os.path.join(REMOTE_PROJECT_ROOT, "dust3r")
    if dust3r_path not in sys.path:
        sys.path.insert(0, dust3r_path)


def update_scenegraph_ui(scenegraph_type):
    """Updates the visibility of sliders based on the selected scenegraph type."""
    import gradio
    if scenegraph_type in ["swin", "swinstride", "swin2stride"]:
        return gradio.update(visible=True), gradio.update(visible=False)
    elif scenegraph_type == "oneref":
        return gradio.update(visible=False), gradio.update(visible=True)
    else:  # 'complete'
        return gradio.update(visible=False), gradio.update(visible=False)


@app.function(gpu="A10G", volumes={REMOTE_PROJECT_ROOT: volume}, timeout=1800)
def run_inference_on_gpu(
    preprocessed_data_path: str,
    scene_output_path: str,
    image_size: int,
    weights: str,
    scenegraph_type: str,
    winsize: int,
    refid: int,
    niter: int,
    schedule: str,
    shared_focal: bool,
    temporal_smoothing_weight: float,
    translation_weight: float,
    flow_loss_weight: float,
    flow_loss_start_iter: float,
    flow_loss_threshold: float,
    use_gt_mask: bool,
    not_batchify: bool,
    window_wise: bool,
    window_size: int,
    window_overlap_ratio: float,
    batch_size: int,
):
    """
    This function runs the entire 3D reconstruction pipeline on a remote GPU.
    It loads the model, processes the data, and saves the final scene object.
    """
    import torch
    import copy

    # Set torch performance flags
    torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

    # Ensure the remote environment is set up
    bootstrap_monst3r_repo()

    import sys
    sys.path = ['/root/monst3r_project/monst3r/third_party/RAFT/core'] + sys.path
    from utils.utils import bilinear_sampler, coords_grid

    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = "cuda"
    print("--> Loading model on GPU...")
    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)
    model.eval()

    print(f"--> Loading preprocessed data from {preprocessed_data_path}")
    imgs = torch.load(preprocessed_data_path)
    
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    if scenegraph_type in ["swin", "swinstride", "swin2stride"]:
        scenegraph_type = f"{scenegraph_type}-{winsize}-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = f"{scenegraph_type}-{refid}"

    print("--> Creating image pairs...")
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

    print("--> Running model inference...")
    output = inference(pairs, model, device, batch_size=batch_size, verbose=True)

    print("--> Running global alignment...")
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(
            output, device=device, mode=mode, verbose=True, shared_focal=shared_focal,
            temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=float(translation_weight),
            flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter,
            flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
            num_total_iter=niter, empty_cache=len(pairs) > 72,
            batchify=not (not_batchify or window_wise),
            window_wise=window_wise, window_size=window_size,
            window_overlap_ratio=window_overlap_ratio
        )
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        lr = 0.01
        if window_wise:
            scene.compute_window_wise_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        else:
            scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    print(f"--> Saving reconstructed scene to {scene_output_path}")
    torch.save(scene, scene_output_path)
    volume.commit() # Ensure the data is saved to the volume
    print("--> Inference and alignment complete.")
    return scene_output_path


@app.function(volumes={REMOTE_PROJECT_ROOT: volume}, timeout=600)
def generate_visualizations(
    scene_path: str,
    output_dir: str, # This is a path inside the volume, e.g., /root/monst.../outputs/scene_hash/
    min_conf_thr: float,
    as_pointcloud: bool,
    mask_sky: bool,
    clean_depth: bool,
    transparent_cams: bool,
    cam_size: float,
    show_cam: bool,
):
    """
    Loads a saved scene object and generates all visual artifacts,
    such as the .glb file and gallery images. Runs on a CPU container.
    """
    import torch
    import os
    import matplotlib.pyplot as pl

    bootstrap_monst3r_repo()
    from dust3r.utils.image import rgb
    from dust3r.utils.device import to_numpy
    from dust3r.utils.viz_demo import convert_scene_output_to_glb

    # This function is now a nested helper, only available in the remote environment.
    def get_3D_model_from_scene_remote(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                                clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
        if scene is None:
            return None
        if clean_depth:
            scene = scene.clean_pointcloud()
        if mask_sky:
            scene = scene.mask_sky()
        rgbimg = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()
        pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
        scene.min_conf_thr = min_conf_thr
        scene.thr_for_init_conf = thr_for_init_conf
        msk = to_numpy(scene.get_masks())
        cmap = pl.get_cmap('viridis')
        cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
        cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
        return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                            transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                            cam_color=cam_color)

    # Create output dir for visuals
    os.makedirs(output_dir, exist_ok=True)
    pl.ion()

    print(f"--> Loading scene for visualization from {scene_path}")
    scene = torch.load(scene_path)
    
    # 1. Generate 3D model (.glb file) by reusing the get_3D_model_from_scene logic
    print("--> Generating 3D model...")
    glb_path = get_3D_model_from_scene_remote(
        output_dir, silent=False, scene=scene, min_conf_thr=min_conf_thr, as_pointcloud=as_pointcloud,
        mask_sky=mask_sky, clean_depth=clean_depth, transparent_cams=transparent_cams,
        cam_size=cam_size, show_cam=show_cam
    )
    
    # 2. Generate gallery images
    print("--> Generating gallery images...")
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths] if len(depths) > 0 else [1.0])
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs] if len(confs) > 0 else [1.0])
    confs = [cmap(d/confs_max) for d in confs]
    
    gallery_paths = []
    for i in range(len(rgbimg)):
        # Save RGB image
        rgb_path = os.path.join(output_dir, f"gallery_{i}_rgb.png")
        pl.imsave(rgb_path, rgbimg[i])
        gallery_paths.append(rgb_path)

        # Save Depth image
        depth_path = os.path.join(output_dir, f"gallery_{i}_depth.png")
        pl.imsave(depth_path, rgb(depths[i]))
        gallery_paths.append(depth_path)

        # Save Confidence image
        conf_path = os.path.join(output_dir, f"gallery_{i}_conf.png")
        pl.imsave(conf_path, rgb(confs[i]))
        gallery_paths.append(conf_path)

    volume.commit()
    return glb_path, gallery_paths


def main_demo(tmpdirname, image_size, server_name, server_port, silent=False, args=None):
    import gradio

    def run_reconstruction_and_get_scene(*args_list):
        """
        This function is called when the Gradio 'Run' button is clicked.
        It calls the remote GPU function to perform inference and alignment,
        then triggers the remote visualization function.
        """
        # 1. Unpack arguments from Gradio UI
        (
            schedule, niter, min_conf_thr, as_pointcloud,
            mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
            scenegraph_type, winsize, refid, seq_name, new_model_weights,
            temporal_smoothing_weight, translation_weight, shared_focal,
            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold,
            use_davis_gt_mask, fps, num_frames
        ) = args_list
        
        # 2. Define path for the output scene file based on the preprocessed data path
        import hashlib
        import os
        file_hash = hashlib.md5(args.preprocessed_data.encode()).hexdigest()
        scene_output_path = os.path.join(REMOTE_PROJECT_ROOT, "outputs", f"scene_{file_hash}.pth")
        
        # 3. Call the remote GPU function to get the scene
        print(f"--> Calling remote inference function. Scene will be saved to {scene_output_path}")
        run_inference_on_gpu.remote(
            preprocessed_data_path=args.preprocessed_data,
            scene_output_path=scene_output_path,
            image_size=image_size,
            weights=new_model_weights,
            scenegraph_type=scenegraph_type,
            winsize=winsize,
            refid=refid,
            niter=niter,
            schedule=schedule,
            shared_focal=shared_focal,
            temporal_smoothing_weight=temporal_smoothing_weight,
            translation_weight=translation_weight,
            flow_loss_weight=flow_loss_weight,
            flow_loss_start_iter=flow_loss_start_iter,
            flow_loss_threshold=flow_loss_threshold,
            use_gt_mask=use_davis_gt_mask,
            not_batchify=args.not_batchify,
            window_wise=args.window_wise,
            window_size=args.window_size,
            window_overlap_ratio=args.window_overlap_ratio,
            batch_size=args.batch_size,
        )

        # 4. Now, trigger the visualization with the default parameters
        return re_render_visuals(
            scene_output_path, min_conf_thr, as_pointcloud, mask_sky,
            clean_depth, transparent_cams, cam_size, show_cam
        )

    def re_render_visuals(scene_path, min_conf_thr, as_pointcloud, mask_sky,
                           clean_depth, transparent_cams, cam_size, show_cam):
        """
        Calls the remote visualization function with new parameters.
        """
        if not scene_path:
            # This can happen if the sliders are changed before a scene is generated.
            return None, None, None

        import os
        import hashlib
        # Use a hash of the scene path and viz params for the output dir
        params_str = f"{scene_path}{min_conf_thr}{as_pointcloud}{mask_sky}{clean_depth}{transparent_cams}{cam_size}{show_cam}"
        viz_hash = hashlib.md5(params_str.encode()).hexdigest()
        visuals_output_dir = os.path.join(REMOTE_PROJECT_ROOT, "outputs", f"visuals_{viz_hash}")

        print(f"--> Calling remote visualization function. Visuals will be saved to {visuals_output_dir}")
        glb_path, gallery_paths = generate_visualizations.remote(
            scene_path=scene_path,
            output_dir=visuals_output_dir,
            min_conf_thr=min_conf_thr,
            as_pointcloud=as_pointcloud,
            mask_sky=mask_sky,
            clean_depth=clean_depth,
            transparent_cams=transparent_cams,
            cam_size=cam_size,
            show_cam=show_cam,
        )
        print("--> Visualization complete. Returning file paths to UI.")
        return scene_path, glb_path, gallery_paths


    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="MonST3R Demo") as demo:
        # scene_path_state stores the path to the last computed scene file
        scene_path_state = gradio.State(None)

        gradio.HTML(f'<h2 style="text-align: center;">MonST3R Demo</h2>')
        with gradio.Column():
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=50, step=1, visible=True)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=100, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence thresholdx
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.01)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                # adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.001)
                # add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # change to another model
                new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=0, maximum=100, step=1)
                # for video processing
                fps = gradio.Slider(label="FPS", value=0, minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100, minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf', columns=4, height="100%")

            # events
            scenegraph_type.change(fn=update_scenegraph_ui,
                                   inputs=[scenegraph_type],
                                   outputs=[winsize, refid])

            run_btn.click(fn=run_reconstruction_and_get_scene,
                          inputs=[schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, new_model_weights, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_davis_gt_mask,
                                  fps, num_frames],
                      outputs=[scene_path_state, outmodel, outgallery])

        # Event handlers for re-rendering the 3D model with different viz parameters
        viz_inputs = [scene_path_state, min_conf_thr, as_pointcloud, mask_sky,
                      clean_depth, transparent_cams, cam_size, show_cam]
        viz_outputs = [scene_path_state, outmodel, outgallery]

        min_conf_thr.release(fn=re_render_visuals, inputs=viz_inputs, outputs=viz_outputs)
        cam_size.change(fn=re_render_visuals, inputs=viz_inputs, outputs=viz_outputs)
        as_pointcloud.change(fn=re_render_visuals, inputs=viz_inputs, outputs=viz_outputs)
        mask_sky.change(fn=re_render_visuals, inputs=viz_inputs, outputs=viz_outputs)
        clean_depth.change(fn=re_render_visuals, inputs=viz_inputs, outputs=viz_outputs)
        transparent_cams.change(fn=re_render_visuals, inputs=viz_inputs, outputs=viz_outputs)

    demo.launch(share=args.share, server_name=server_name, server_port=server_port)


@app.local_entrypoint()
def main(
    server_name: str = None,
    local_network: bool = False,
    image_size: int = 512,
    preprocessed_data: str = "/root/monst3r_project/data/preprocessed_data.pth",
    server_port: int = None,
    output_dir: str = './demo_tmp',
    silent: bool = False,
    share: bool = False,
    # These are needed for the args object passed to main_demo
    not_batchify: bool = False,
    window_wise: bool = False,
    window_size: int = 100,
    window_overlap_ratio: float = 0.5,
    batch_size: int = 16,
    seq_name: str = "sequence",
    weights: str = 'checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth',
):
    import os
    import tempfile
    from argparse import Namespace

    # Create a lightweight args object for compatibility with main_demo
    # This avoids a larger refactor of the Gradio UI code
    args = Namespace(
        server_name=server_name, local_network=local_network, image_size=image_size,
        preprocessed_data=preprocessed_data, server_port=server_port, output_dir=output_dir,
        silent=silent, share=share, not_batchify=not_batchify, window_wise=window_wise,
        window_size=window_size, window_overlap_ratio=window_overlap_ratio, batch_size=batch_size, seq_name=seq_name,
        weights=weights
    )

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='monst3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    main_demo(tmpdirname, args.image_size, server_name, args.server_port, silent=args.silent, args=args)
