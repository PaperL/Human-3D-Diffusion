import numpy as np
import torch

from pytorch3d.renderer.cameras import look_at_rotation

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.core.cameras.cameras import FoVPerspectiveCameras

device = 'cuda:0'
human = HumanData.fromfile("inference_result.npz")
cam = human["pred_cams"][0]
# kps = torch.Tensor(human["verts"][39]).to(device)

kps = torch.Tensor(np.load('joints.npy', allow_pickle=True))[:3, ...].to(device)

T = torch.zeros([1, 3]).to(device)
R = look_at_rotation(cam)

pers_cam = FoVPerspectiveCameras(
    R = R,
    T = T,
    device = device
)
print(pers_cam)

print(kps.shape)

resolution = (1024, 1024)

kps2d = pers_cam.transform_points_screen(kps, image_size=resolution)[..., :2].cpu()
print(kps2d.shape)
kps2d = np.array(kps2d)

from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

visualize_kp2d(
    kps2d,
    output_path='tmp1',
    resolution=resolution)