import argparse
import os
from os.path import join as pjoin

import torch.cuda
from tqdm import tqdm

from dataset.davis import MultiObjectDavisDataset
from features import ViTFeatureExtractor


def main(output_dir):
    dataset = MultiObjectDavisDataset(year='2017', split='val', resolution='480p')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fex = ViTFeatureExtractor(arch='vit_base', patch_size=8, scale_size=480, device=device)

    os.makedirs(output_dir, exist_ok=True)

    for video in dataset:
        out_file = pjoin(output_dir, f'{video.video_name}.pth')

        # Skip already computed results
        if os.path.exists(out_file):
            print(f'Skipping {video.video_name}.')
            continue

        features = []
        for i in tqdm(range(len(video))):
            preprocessed = fex.preprocess(video.frame_at(i))
            feats = fex.extract(preprocessed)
            features.append(feats.cpu())

        features = torch.stack(features, dim=0)
        torch.save(features, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Feature output directory.')
    args = parser.parse_args()

    main(output_dir=args.output_dir)
