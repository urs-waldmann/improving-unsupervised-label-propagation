"""
Script to convert multi-object DAVIS result to single-object results. The MO masks are read as indexed png files and
saved as colored png files, i.e. using black and white as pixel values. This is helpful to use the legacy evaluation
toolbox for DAVIS2016.
"""
import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory of multi-object davis results.")
    parser.add_argument("output_dir", type=str, help="Output directory for single-object davis results.")
    parser.add_argument("--allow-overwrite", action='store_true', default=False, help="Overwrite existing results.")

    args = parser.parse_args()

    if not osp.isdir(args.input_dir):
        raise ValueError("Input directory does not exist or is a file.")

    if osp.exists(args.output_dir):
        if osp.isdir(args.output_dir):
            if os.listdir(args.output_dir) and not args.allow_overwrite:
                raise ValueError("Output dir is not empty and allow_overwrite is not set.")
    else:
        os.makedirs(args.output_dir)

    video_dirs = [d for d in os.listdir(args.input_dir) if osp.isdir(os.path.join(args.input_dir, d))]

    for video_dir in tqdm(video_dirs):
        in_dir = osp.join(args.input_dir, video_dir)
        out_dir = osp.join(args.output_dir, video_dir)

        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        frame_names = [f for f in os.listdir(in_dir) if osp.isfile(osp.join(in_dir, f)) and f.endswith(".png")]
        for frame_name in frame_names:
            f_in = osp.join(in_dir, frame_name)
            f_out = osp.join(out_dir, frame_name)

            frame = np.asarray(Image.open(f_in)).copy()
            frame[frame > 0] = 1

            frame = Image.fromarray(frame)
            frame.putpalette([0, 0, 0, 255, 255, 255])
            frame.save(f_out)

    pass


if __name__ == '__main__':
    main()
