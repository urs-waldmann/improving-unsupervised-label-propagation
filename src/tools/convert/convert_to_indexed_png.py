"""
Helper script to convert single-object vos results to an indexed image format. This enables evaluation with tools like
the more modern DAVIS eval toolkit.
"""
import argparse
import glob
import os
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.vis_utils import davis_color_map


def imread_indexed(filename):
    im = Image.open(filename)

    annotation = np.atleast_3d(im)
    assert annotation.shape[2] == 1
    assert im.getpalette() is None

    annotation = annotation[..., 0]

    return annotation


def imwrite_indexed(filename, array, color_palette):
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def main(base_dir):
    palette = davis_color_map()

    dirs = [os.path.join(base_dir, n) for n in os.listdir(base_dir)]
    dirs = list(filter(os.path.isdir, dirs))

    for video_dir in tqdm(dirs):
        files = glob.glob(os.path.join(video_dir, '*.png'))

        for file in files:
            annot = imread_indexed(file)

            uniq = np.sort(np.unique(annot))
            if len(uniq) != 2:
                warnings.warn('More than two unique colors in non-indexed image.')
            else:
                annot[annot == uniq[0]] = 0
                annot[annot == uniq[1]] = 1

            imwrite_indexed(file, annot, palette)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Result converter',
                                     description='''
                                     This script converts mask images to indexed mask images. The conversion is
                                     performed in-place. Create a backup of your results beforehand. The script expects
                                     an input directory where each sub-directory contains a number of png mask files.
                                     The mask selection performs a simple globbing and converts every png!
                                     ''')
    parser.add_argument('--result-dir',
                        type=str,
                        required=True,
                        help='Directory with sub-folders for each video sequence.')
    args = parser.parse_args()
    main(args.result_dir)
