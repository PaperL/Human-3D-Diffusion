import numpy as np
import torch as th
from mmhuman3d.core.evaluation.eval_utils import keypoint_mpjpe


def calculate_3d_loss(body_model, model_output, gt_keypoints3d):
    batch_size = model_output.shape[0]
    pred_pose = th.FloatTensor(model_output[:, :216].detach().cpu().numpy()).reshape(batch_size, 24, 3, 3)
    pred_beta = th.FloatTensor(model_output[:, 216:].detach().cpu().numpy()).reshape(batch_size, 10)

    pred_output = body_model(
                betas=pred_beta,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False
            )
    pred_keypoints3d = pred_output['joints'].detach().cpu().numpy()

    H36M_TO_J17 = [
        6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
    ]
    H36M_TO_J14 = H36M_TO_J17[:14]
    joint_mapper = H36M_TO_J14

    pred_pelvis = pred_keypoints3d[:, 0]
    pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]
    pred_keypoints3d = (pred_keypoints3d - pred_pelvis[:, None, :]) * 1000
    gt_keypoints3d_mask = np.ones((batch_size, 17))
    gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0
    # print(pred_keypoints3d.shape, gt_keypoints3d.shape)
    alignment = 'procrustes'
    return keypoint_mpjpe(pred_keypoints3d, np.array(gt_keypoints3d), gt_keypoints3d_mask, alignment)
