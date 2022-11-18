"""
Converts legacy UDT model to modern PyTorch model format.
"""
import os

import torch

cp_files = [
    'model_best.pth.tar',
    'checkpoint.pth.tar'
]

for cp in cp_files:
    new_name = os.path.splitext(cp)[0]

    print(f'Saving to path {new_name}')
    x = torch.load(cp, encoding='latin1')
    torch.save(x, new_name)
