from mmhuman3d.data.data_structures.human_data import HumanData
import _pickle as pkl
import torch
import numpy as np

human = HumanData.fromfile("compare/inference_result.npz")
smpl = human["smpl"]

gt = pkl.load(open("compare/courtyard_basketball_01.pkl", "rb"), encoding="bytes")
gt_poses = torch.Tensor(np.array(gt[b'poses']))
gt_betas = torch.Tensor(np.array(gt[b'betas']))
gt_keypoints = torch.Tensor(np.array(gt[b'jointPositions'])).reshape(-1, 24, 3)[:954, ...]

print(gt_keypoints.shape)



body_model_load_dir = '../mmhuman3d-0.10.0/data/body_models/smpl'
extra_joints_regressor = '../mmhuman3d-0.10.0/data/body_models/J_regressor_extra.npy'

from mmhuman3d.models.body_models.builder import build_body_model

smpl_m = build_body_model(
    dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_24',
        model_path=body_model_load_dir,
        extra_joints_regressor=extra_joints_regressor
    )
)

# print(smpl['global_orient'].shape, smpl['body_pose'].shape)
smpl['global_orient'] = smpl['global_orient'].reshape(smpl['global_orient'].shape[0], 1, -1)
# print(smpl['global_orient'].shape)
# print("betas", smpl['betas'].shape)

import torch
import numpy as np

for key in smpl:
    smpl[key] = torch.Tensor(smpl[key])

smpl_output = smpl_m(**smpl)
keypoints = smpl_output['joints']
loss = []

from mmhuman3d.core.evaluation import keypoint_mpjpe
for i in range(954):
    kp = keypoints[i].reshape(1, 24, -1).detach().numpy()
    gtkp = gt_keypoints[i].reshape(1, 24, -1).numpy()
    loss.append(keypoint_mpjpe(kp, gtkp, np.ones(kp.shape[:-1], dtype=bool), alignment='procrustes'))

loss = np.array(loss)
max_idx = np.argmax(loss)
print(max_idx, loss[max_idx])

resolution = (1920, 1080)
smpl_poses = np.concatenate((smpl["global_orient"].reshape(954, -1), smpl["body_pose"].reshape(954, -1)), axis=1)

from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
    visualize_smpl_hmr,
    visualize_smpl_pose,
    visualize_smpl_vibe,
    visualize_T_pose,
)
body_model_config = {
    'use_pca': False,
    'use_face_contour': True,
    'model_path': '../mmhuman3d-0.10.0/data/body_models'
}
body_model_config.update(type='smpl')
visualize_smpl_hmr(
    poses=smpl_poses[max_idx].reshape(1, -1),
    betas=smpl["betas"][max_idx].reshape(1, -1),
    cam_transl=human["pred_cams"][max_idx].reshape(1, -1),
    bbox=human["bboxes_xyxy"][max_idx].reshape(1, -1),
    output_path='compare_out',
    resolution=resolution,
    body_model_config=body_model_config,
    overwrite=True,
    read_frames_batch=True,
    origin_frames="compare",
    plot_kps=True,
    device='cuda:0')
