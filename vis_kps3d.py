from mmhuman3d.data.data_structures.human_data import HumanData

human = HumanData.fromfile("inference_result.npz")

smpl = human["smpl"]

from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

body_model_load_dir = '../mmhuman3d-0.10.0/data/body_models/smpl'
extra_joints_regressor = '../mmhuman3d-0.10.0/data/body_models/J_regressor_extra.npy'

from mmhuman3d.models.body_models.builder import build_body_model

smpl_m = build_body_model(
    dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_54',
        model_path=body_model_load_dir,
        extra_joints_regressor=extra_joints_regressor
    )
)

# print(smpl['global_orient'].shape, smpl['body_pose'].shape)
smpl['global_orient'] = smpl['global_orient'].reshape(smpl['global_orient'].shape[0], 1, -1)
print(smpl['global_orient'].shape)

import torch
import numpy as np

for key in smpl:
    smpl[key] = torch.Tensor(smpl[key])

smpl_output = smpl_m(**smpl)
keypoints = smpl_output['joints']

cam = torch.Tensor(human["pred_cams"])

# focal_length=5000
# det_width=224
# det_height=224

from mmhuman3d.utils.demo_utils import (
    get_default_hmr_intrinsic,
    convert_bbox_to_intrinsic,
    convert_crop_cam_to_orig_img
)

resolution = (1920, 1080)

# K = torch.Tensor(
#     get_default_hmr_intrinsic(
#         focal_length=focal_length,
#         det_height=det_height,
#         det_width=det_width))
# K = K.view(-1, K.shape[-2], K.shape[-1])
# T = torch.cat([cam[..., [1]], cam[..., [2]], 2 * focal_length /
#         (det_width * cam[..., [0]] + 1e-9)
#         ], -1)
# T = T.view(-1, 3)

# from mmhuman3d.core.conventions.cameras.convert_convention import \
#     convert_camera_matrix  # prevent yapf isort conflict

# Ks = convert_bbox_to_intrinsic(human['bboxes_xyxy'], bbox_format='xyxy')

# def prepare_cameras(Ks, K, T, R):
#     num_person = 1
#     num_frames = 40
#     # prepare camera matrixs
#     if Ks is not None:
#         if isinstance(Ks, np.ndarray):
#             Ks = torch.Tensor(Ks)
#         Ks = Ks.view(-1, num_person, 3, 3)
#         Ks = Ks.view(-1, 3, 3)
#         K = K.repeat(num_frames * num_person, 1, 1)

#         Ks = K.inverse() @ Ks @ K
#         T = None
#     return K, T, R

# K, R, T = prepare_cameras(Ks, K, T, None)

# K, R, T = convert_camera_matrix(
#     convention_dst='pytorch3d',
#     K=K,
#     R=None,
#     T=T,
#     is_perspective=True,
#     convention_src='opencv',
#     resolution_src=resolution,
#     in_ndc_src=in_ndc,
#     in_ndc_dst=in_ndc)

from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.cameras.cameras import WeakPerspectiveCameras
from mmhuman3d.core.conventions.cameras.convert_convention import convert_camera_matrix

r = resolution[1] / resolution[0]

orig_cams = convert_crop_cam_to_orig_img(
    cam, 
    human['bboxes_xyxy'], 
    resolution[1], 
    resolution[0], 
    aspect_ratio=1.0,
    bbox_scale_factor=1.25, 
    bbox_format='xyxy'
)

vertices = torch.Tensor(human["verts"])

K, R, T = WeakPerspectiveCameras.convert_orig_cam_to_matrix(
    orig_cam=torch.Tensor(orig_cams), 
    aspect_ratio=r,
    znear=torch.min(vertices[..., 2] - 1)
)
in_ndc = True
K, R, T = convert_camera_matrix(
    K=K,
    R=R,
    T=T,
    is_perspective=False,
    in_ndc_src=in_ndc,
    in_ndc_dst=in_ndc,
    resolution_src=resolution,
    convention_src='opencv',
    convention_dst='pytorch3d')

T = T.view(-1, 3)
pers_cam = build_cameras(
        dict(
            type='weakperspective',
            K=K,
            R=R,
            T=T,
            in_ndc=in_ndc,
            resolution=resolution))

# visualize_kp3d(
#     keypoints.detach().numpy(),
#     output_path='temp',
#     resolution=resolution,
#     mask=None,
#     orbit_speed=0.5,
#     # disable_limbs=True,
#     data_source='smpl'
# )

kps2d = pers_cam.transform_points_screen(keypoints, image_size=resolution)[..., :2].cpu()

visualize_kp2d(
    kps2d.detach().numpy(),
    output_path='temp',
    resolution=resolution,
    mask=None,
    disable_limbs=False,
    data_source='smpl',
    overwrite=True,
    draw_bbox=True,
    origin_frames="outputs/B_single_person_result/images"
)