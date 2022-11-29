from torch.utils.data import Dataset
from mmhuman3d.data.data_structures.human_data import HumanData
import numpy as np
from pathlib import Path
from tqdm import tqdm
import bisect
import _pickle as pkl
import torch

class HumanPoseVectorTrainSet(Dataset):
    def __init__(self, npz_folder_path_str, gt_folder_path_str):
        self.data_size = 0
        self.npz_sizes = []
        self.npz_data_id = []
        self.npz_file_paths = []

        npz_folder_path = Path(npz_folder_path_str)
        assert npz_folder_path.is_dir()
        assert npz_folder_path.exists()
        self.npz_file_paths = list(npz_folder_path.rglob('*.npz'))
        assert len(self.npz_file_paths) > 0
        print('Collect', len(self.npz_file_paths), 'npz file(s).')
        progress_bar = tqdm(self.npz_file_paths)

        gt_folder_path = Path(gt_folder_path_str)
        assert gt_folder_path.is_dir()
        assert gt_folder_path.exists()
        self.gt_file_paths = list(gt_folder_path.rglob('*.pkl'))

        self.file_map = {}
        for i in range(len(self.npz_file_paths)):
            npz_true_name = str(self.npz_file_paths[i])[len(npz_folder_path_str)+1:]
            npz_true_name = npz_true_name[:-21]
            #npz_true_name = npz_true_name[:npz_true_name.find("\\")]
            #print(self.gt_file_paths)
            #print(ss)
            for j in range(len(self.gt_file_paths)):
                if str.find(str(self.gt_file_paths[j]), npz_true_name) != -1:
                    #print(npz_true_name, "->", str(self.gt_file_paths[j]))
                    self.file_map[i] = j

        for i, npz_path in enumerate(progress_bar):
            progress_bar.set_description(npz_path.parent.stem)
            n = self.get_npz_size(i)
            self.npz_data_id.append(self.data_size)
            self.npz_sizes.append(n)
            self.data_size += n
        print('Total data number:', self.data_size)
        # print(self.npz_sizes, self.npz_data_id, self.npz_file_paths, sep='\n')

    def __getitem__(self, index):
        npz_id = bisect.bisect(self.npz_data_id, index) - 1
        id = self.npz_data_id[npz_id] - index
        npz_path = self.npz_file_paths[npz_id]
        dat = HumanData.fromfile(str(npz_path))
        ret = np.empty(0)
        gt_dat = pkl.load(open(str(self.gt_file_paths[self.file_map[npz_id]]), "rb"), encoding="bytes")
        gt_keypoints = torch.Tensor(np.array(gt_dat[b'jointPositions']))
        true_person_n = int(gt_keypoints.shape[0])
        true_frame_n = int(gt_keypoints.shape[1])
        gt_keypoints = gt_keypoints.reshape(true_person_n, true_frame_n, 24, 3)
        frame_id = dat['frame_id'][id]
        person_id = dat['person_id'][id]
        for smpl_type in dat['smpl'].keys():
            np_array = dat['smpl'][smpl_type][id].flatten()
            ret = np.concatenate((ret, np_array), axis=0)
        zero = torch.zeros(24, 3)
        return ret, gt_keypoints[person_id][frame_id] if person_id < true_person_n else zero

    def __len__(self):
        return self.data_size

    def get_npz_size(self, i):
        dat = HumanData.fromfile(str(self.npz_file_paths[i]))
        return dat.data_len

# dataset = HumanPoseVectorTrainSet("PW3D_NPZ_multi_person", "data/datasets/pw3d/sequenceFiles")

# for i in range(len(dataset)):
#     a, b = dataset[i]
#     print(a.shape, b.shape)
