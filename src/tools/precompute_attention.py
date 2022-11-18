import argparse
import os

import torch
from tqdm import tqdm

from dataset.davis import MultiObjectDavisDataset
from features import ViTFeatureExtractor


def main(args, configs):
    out_root = args.output

    for config in configs:
        name = f'vit_{config["arch"][4]}{config["patch_size"]}'
        out_path = os.path.join(out_root, name)

        os.makedirs(out_path, exist_ok=True)

        davis = MultiObjectDavisDataset(year='2017', split='trainval')
        feat_ext = ViTFeatureExtractor(**config, scale_size=480, device=torch.device('cuda'))

        for video in tqdm(davis, desc=name, total=len(davis)):
            frame = video.frame_at(0)

            attn = feat_ext.get_attention(frame).clone()

            out_file = os.path.join(out_path, f'{video.video_name}.attention.pth')
            torch.save(attn, out_file)

        del feat_ext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True,
                        help='Output path to save attention maps to.')

    vit_configs = [
        dict(arch='vit_base', patch_size=16),
        dict(arch='vit_small', patch_size=16),
        dict(arch='vit_small', patch_size=8),
        dict(arch='vit_base', patch_size=8),
    ]

    main(parser.parse_args(), configs=vit_configs)
