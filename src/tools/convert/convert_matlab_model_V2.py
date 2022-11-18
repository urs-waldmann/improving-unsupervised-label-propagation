"""
Second version of the UDT MATLAB conversion script. This script expects a different input format.
"""

import argparse
from collections import OrderedDict

import numpy as np
import torch
from scipy import io

from model.DcfNetFeature import DcfNetFeature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='net_param.mat', help='Input model from matlab.')
    parser.add_argument('--output', type=str, default='param.pth.tar', help='Pytorch output model.')
    args = parser.parse_args()

    state_dict = OrderedDict()

    p = io.loadmat(args.input, squeeze_me=True, simplify_cells=True)
    for layer in [0, 2]:
        weight_tensor = torch.Tensor(np.transpose(p['net']['layers'][layer]['weights'][0], (3, 2, 0, 1)))
        bias_tensor = torch.Tensor(np.squeeze(p['net']['layers'][layer]['weights'][1]))

        if layer == 0:
            # cv2 bgr input
            weight_tensor = torch.Tensor(weight_tensor.numpy()[:, ::-1, :, :].copy())

        state_dict[f'feature.{layer}.weight'] = weight_tensor
        state_dict[f'feature.{layer}.bias'] = bias_tensor

    torch.save(state_dict, args.output)

    # network test
    net = DcfNetFeature()
    net.eval()
    net.load_state_dict(torch.load(args.output))
