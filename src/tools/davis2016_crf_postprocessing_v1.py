"""
Helper script to perform CRF-based mask post-processing for DAVIS2016 results. The post-processing uses very bad
settings for the post-processing. Don't expect good results.
"""
import argparse
import os
from multiprocessing import Pool
from os.path import join, basename

import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from pydensecrf.utils import unary_from_labels

import config
from utils import path_utils


def refine_mask(mask: np.ndarray, image: np.ndarray, sxy=25.0, srgb=5., compat=5., num_iters=10, gt_prob=0.8,
                num_classes=None):
    if num_classes is None:
        num_classes = mask.max() + 1

    crf = dcrf.DenseCRF2D(mask.shape[1], mask.shape[0], num_classes)

    unary_energy = unary_from_labels(mask, num_classes, gt_prob=gt_prob, zero_unsure=False)
    crf.setUnaryEnergy(unary_energy)

    im = np.ascontiguousarray(image)
    crf.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=im, compat=compat)

    new_mask = np.argmax(crf.inference(num_iters), axis=0)
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1]))

    assert new_mask.max() < 256
    new_mask = new_mask.astype(np.uint8)

    return new_mask


def refine_video(data):
    frame_root = data['frame_root']
    out_dir = data['out_dir']
    video_dir = data['video_dir']

    print(f'refining: {frame_root}')

    frame_dir = join(frame_root, basename(video_dir))

    video_out_dir = join(out_dir, basename(video_dir))
    os.makedirs(video_out_dir, exist_ok=True)

    print(video_out_dir)

    mask_paths = path_utils.list_files(video_dir, filter_pattern=r'^.+\.png$')
    print(len(mask_paths))
    for mask_path in mask_paths:
        frame_path = join(frame_dir, os.path.splitext(basename(mask_path))[0] + '.jpg')
        out_mask_path = join(video_out_dir, basename(mask_path))

        mask_img = Image.open(mask_path)
        mask = np.asarray(mask_img)
        frame = np.asarray(Image.open(frame_path))

        refined = refine_mask(mask, frame)

        new_mask = Image.fromarray(refined)
        new_mask.putpalette(mask_img.getpalette())
        new_mask.save(out_mask_path, format='PNG')
    print(f'completed: {frame_root}')


def main(args):
    in_dir = args.input_dir
    frame_root = args.frame_dir
    out_dir = args.output_dir
    allow_overwrite = args.allow_overwrite

    if not os.path.isdir(in_dir):
        raise ValueError('Input path is not a directory.')
    if not os.path.isdir(frame_root):
        raise ValueError('Frame root is not a directory.')
    os.makedirs(out_dir, exist_ok=allow_overwrite)

    videos = [dict(video_dir=vp, frame_root=frame_root, out_dir=out_dir) for vp in path_utils.list_subdirs(in_dir)]

    with Pool(processes=8) as pool:

        pool.map(refine_video, videos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CRF-based refinement for DAVIS results.')
    parser.add_argument('-i', '--input-dir',
                        type=str,
                        required=True,
                        help='Directory where the unrefined results are stored.')
    parser.add_argument('-o', '--output-dir',
                        type=str,
                        required=True,
                        help='Directory where the refined output should be stored.')
    parser.add_argument('--frame-dir',
                        type=str,
                        default=join(config.dataset_davis2017_root, 'JPEGImages', '480p'),
                        help='Directory where the refined output should be stored.')
    parser.add_argument('--allow-overwrite',
                        action='store_true',
                        default=False,
                        help='Allows overwriting existing data in the output directory.')
    main(parser.parse_args())
