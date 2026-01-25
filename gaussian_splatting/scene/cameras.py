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

import torch
from torch import nn
import numpy as np
import cv2
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 # External depth/normal prior support
                 depth_params=None, invdepthmap=None, normalmap=None, resolution=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # === External depth prior processing ===
        self.invdepthmap = None
        self.depth_reliable = False
        self.depth_mask = None
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.original_image)
            if resolution is not None:
                invdepthmap = cv2.resize(invdepthmap, resolution)
            invdepthmap[invdepthmap < 0] = 0
            self.depth_reliable = True
            
            # Apply depth scale alignment
            if depth_params is not None:
                if depth_params.get("scale", 1.0) < 0.2 * depth_params.get("med_scale", 1.0) or \
                   depth_params.get("scale", 1.0) > 5 * depth_params.get("med_scale", 1.0):
                    self.depth_reliable = False
                    self.depth_mask *= 0
                    print(f'Depth of {self.image_name} is unreliable')
                
                if depth_params.get("scale", 0) > 0:
                    invdepthmap = invdepthmap * depth_params["scale"] + depth_params.get("offset", 0)
            
            if invdepthmap.ndim != 2:
                invdepthmap = invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(invdepthmap[None]).float().to(self.data_device)
        
        # === External normal prior processing ===
        self.normalmap = None
        if normalmap is not None:
            if resolution is not None:
                normalmap = cv2.resize(normalmap, resolution)
            normalmap_ = torch.from_numpy(normalmap).float().permute(2, 0, 1)
            self.normalmap = torch.nn.functional.normalize(normalmap_, dim=0).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

