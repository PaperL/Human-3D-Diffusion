from torch.utils.data import Dataset
from mmhuman3d.data.data_structures.human_data import HumanData
import numpy as np
from pathlib import Path
from tqdm import tqdm
import bisect


class HumanPoseVectorTrainSet(Dataset):
    def __init__(self, npz_folder_path_str):
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
        # print(index, npz_id)
        id = self.npz_data_id[npz_id] - index
        npz_path = self.npz_file_paths[npz_id]
        dat = HumanData.fromfile(str(npz_path))
        ret = np.empty(0)
        for smpl_type in dat['smpl'].keys():
            np_array = dat['smpl'][smpl_type][id].flatten()
            ret = np.concatenate((ret, np_array), axis=0)
        return ret

    def __len__(self):
        return self.data_size

    def get_npz_size(self, i):
        dat = HumanData.fromfile(str(self.npz_file_paths[i]))
        return dat.data_len
