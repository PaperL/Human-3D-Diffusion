from torch.utils.data import Dataset
from mmhuman3d.data.data_structures.human_data import HumanData
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mmhuman3d.data.datasets import build_dataset
from mmhuman3d.utils.geometry import rotation_matrix_to_angle_axis
import json
import torch

class PW3D(Dataset):
    """
        result: .json
        gt: .npz

        gt path: data/preprocessed_datasets/pw3d_test.npz
    """

    def __init__(self, result_path, data_size = -1):
        with open(result_path) as fp:
            self.res_file = json.load(fp)

        cfg = {'type': 'HumanImageDataset', 'body_model': {'type': 'GenderedSMPL', 'keypoint_src': 'h36m', 'keypoint_dst': 'h36m', 'model_path': 'data/body_models/smpl', 'joints_regressor': 'data/body_models/J_regressor_h36m.npy'}, 'dataset_name': 'pw3d', 'data_prefix': 'data', 'pipeline': [{'type': 'LoadImageFromFile'}, {'type': 'GetRandomScaleRotation', 'rot_factor': 0, 'scale_factor': 0}, {'type': 'MeshAffine', 'img_res': 224}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'ImageToTensor', 'keys': ['img']}, {'type': 'ToTensor', 'keys': ['has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx']}, {'type': 'Collect', 'keys': ['img', 'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'], 'meta_keys': ['image_path', 'center', 'scale', 'rotation']}], 'ann_file': 'pw3d_test.npz', 'test_mode': True}
        self.dataset = build_dataset(cfg)
        assert len(self.res_file["keypoints"]) == self.dataset.num_data

        if data_size > 0:
            self.dataset.num_data = data_size
            for key in self.res_file:
                self.res_file[key] = self.res_file[key][:data_size]

        print('Total data number:', self.dataset.num_data)

        _, self.gt_keypoints3d, _ = \
            self.dataset._parse_result(self.res_file)

        print('Parse finish.')
        self.body_model = self.dataset.body_model


    def __getitem__(self, index):
        # print(torch.tensor(self.res_file["poses"][index]).shape)
        # print(len(np.array(self.res_file["poses"][index]).flatten()), len(self.res_file["betas"][index]))
        # aa = rotation_matrix_to_angle_axis(torch.tensor(np.array(self.res_file["poses"][index])).reshape(24, 3, 3))
        return np.concatenate(
            # (aa.reshape(72,), np.array(self.res_file["betas"][index]))
            (np.array(self.res_file["poses"][index]).reshape(216,), np.array(self.res_file["betas"][index]))
            ,axis=0,
        ), np.array(self.gt_keypoints3d[index])


    def __len__(self):
        return self.dataset.num_data


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = PW3D("datasets/result_keypoints.json", 16)
    print(dataset[0][0].shape)
    # def forward(x):
    #     return

    # loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # for x in tqdm(loader):
    #     forward(x)

