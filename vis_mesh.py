from mmhuman3d.data.data_structures.human_data import HumanData

human = HumanData.fromfile("inference_result.npz")

smpl = human["smpl"]

frame_id = 0

import torch
if torch.cuda.is_available():
    device_name = 'cuda:0'
else:
    device_name = 'cpu'

body_model_config = {
    'use_pca': False,
    'use_face_contour': True,
    'model_path': '../mmhuman3d-0.10.0/data/body_models'
}

from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
    visualize_smpl_hmr,
    visualize_smpl_pose,
    visualize_smpl_vibe,
    visualize_T_pose,
)

import numpy as np
body_model_config.update(type='smpl')
print(smpl["body_pose"].shape, smpl["global_orient"].shape)
smpl_poses = np.concatenate((smpl["global_orient"].reshape(40, -1), smpl["body_pose"].reshape(40, -1)), axis=1)
print(smpl_poses.shape)

resolution = (1920, 1080)

# output_path: .mmp4, .gif or a image directory
# visualize_smpl_pose(
#     poses=smpl_poses,
#     body_model_config=body_model_config,
#     output_path='temp',
#     resolution=(1048, 576),
#     overwrite=True,
#     plot_kps=True,
#     device=device_name
# )

visualize_smpl_hmr(
    poses=smpl_poses[:1],
    betas=smpl["betas"][:1],
    cam_transl=human["pred_cams"][:1],
    bbox=human["bboxes_xyxy"][:1],
    output_path='temp1',
    resolution=resolution,
    body_model_config=body_model_config,
    overwrite=True,
    read_frames_batch=True,
    origin_frames="outputs/B_single_person_result/images",
    plot_kps=True,
    device=device_name)