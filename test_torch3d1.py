import torch

from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

sphere_mesh = ico_sphere(level=3)
verts, faces, _ = load_obj("model.obj")
test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

sample_sphere = sample_points_from_meshes(sphere_mesh, 5000)
sample_test = sample_points_from_meshes(test_mesh, 5000)
loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)