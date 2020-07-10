
import torch
# there should be pipeline. pipeline is bigger that randlanet
from laplacian import load_obj
from network import LapCluster

from config import test_config as cfg

model   = LapCluster(cfg)

verts, faces = load_obj('4.obj')

verts = verts[None, :, :]
faces = faces[None, :, :]-1

verts = torch.from_numpy(verts)
faces = torch.from_numpy(faces)
inputs = dict()

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

inputs['verts'] = verts
inputs['faces'] = faces

result = model(inputs, device)
print(result.size())