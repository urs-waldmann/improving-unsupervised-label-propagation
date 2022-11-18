"""
Script to convert UDT MATLAB models to PyTorch format.
"""
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.fft as fft
from scipy import io

from model.DcfNetFeature import DcfNetFeature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='net_param.mat', help='Input model from matlab.')
    parser.add_argument('--output', type=str, default='param.pth.tar', help='Pytorch output model.')
    args = parser.parse_args()

    # network test
    net = DcfNetFeature()
    net.eval()

    for idx, m in enumerate(net.modules()):
        print(idx, '->', m)

    for name, param in net.named_parameters():
        if 'bias' in name or 'weight' in name:
            print(param.size())

    p = io.loadmat(args.input)
    x = p['res'][0][0][:, :, ::-1].copy()
    x_out = p['res'][0][-1]

    pth_state_dict = OrderedDict()

    match_dict = dict()
    match_dict['feature.0.weight'] = 'conv1_w'
    match_dict['feature.0.bias'] = 'conv1_b'
    match_dict['feature.2.weight'] = 'conv2_w'
    match_dict['feature.2.bias'] = 'conv2_b'

    for var_name in net.state_dict().keys():
        print(var_name)
        key_in_model = match_dict[var_name]
        param_in_model = var_name.rsplit('.', 1)[1]

        if 'weight' in var_name:
            pth_state_dict[var_name] = torch.Tensor(np.transpose(p[key_in_model], (3, 2, 0, 1)))
        elif 'bias' in var_name:
            pth_state_dict[var_name] = torch.Tensor(np.squeeze(p[key_in_model]))

        if var_name == 'feature.0.weight':
            weight = pth_state_dict[var_name].data.numpy()
            weight = weight[:, ::-1, :, :].copy()  # cv2 bgr input
            pth_state_dict[var_name] = torch.Tensor(weight)

    torch.save(pth_state_dict, args.output)

    net.load_state_dict(torch.load(args.output))

    x_t = torch.Tensor(np.expand_dims(np.transpose(x, (2, 0, 1)), axis=0))
    x_pred = net(x_t).detach().cpu().numpy()
    pred_error = np.sum(np.abs(np.transpose(x_pred, (0, 2, 3, 1)).reshape(-1) - x_out.reshape(-1)))

    x_fft = fft.fftn(x_t, dim=[-2, -1])

    print('model_transfer_error:{:.5f}'.format(pred_error))
