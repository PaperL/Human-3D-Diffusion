import pytorch3d
print(pytorch3d.__version__)
from pytorch3d.renderer import MeshRenderer
print(MeshRenderer)
from pytorch3d.structures import Meshes
print(Meshes)
from pytorch3d.renderer import cameras
print(cameras)
from pytorch3d.transforms import Transform3d
print(Transform3d)

import torch
device=torch.device('cuda')
from pytorch3d.utils import torus
Torus = torus(r=10, R=20, sides=100, rings=100, device=device)
print(Torus.verts_padded())