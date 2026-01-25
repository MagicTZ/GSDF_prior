#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
from gaussian_splatting.scene.cameras import Camera
import numpy as np
from gaussian_splatting.utils.general_utils import PILtoTorch
from gaussian_splatting.utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # === Load external depth prior ===
    invdepthmap = None
    depth_path = getattr(cam_info, 'depth_path', "")
    if depth_path and os.path.exists(depth_path):
        invdepthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if invdepthmap is not None:
            # Check depth type from args
            depths_type = getattr(args, 'depths', '')
            if depths_type in ['metric3dv2', 'depths_metric3dv2']:
                # Metric depth in mm -> meters
                invdepthmap = invdepthmap.astype(np.float32) / 1000.0
            else:
                # Relative depth normalized to 16-bit
                invdepthmap = invdepthmap.astype(np.float32) / 65535.0
    
    # === Load external normal prior ===
    normalmap = None
    normal_path = getattr(cam_info, 'normal_path', "")
    if normal_path and os.path.exists(normal_path):
        try:
            normalmap = np.load(normal_path)['arr_0']  # .npz format
        except:
            pass  # normal file not found or invalid format

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  # External depth/normal prior
                  depth_params=getattr(cam_info, 'depth_params', None),
                  invdepthmap=invdepthmap,
                  normalmap=normalmap,
                  resolution=resolution)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
