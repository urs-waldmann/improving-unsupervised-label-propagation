import argparse
import os

import cv2 as cv
import numpy as np
import torch

import utils.util
from dataset.JHMDB import Jhmdb
from keypoint import PropagationKeypointDcf
from features import get_feature_extractor
from utils.keypoint_visualization import draw_skeleton


def eval_video(out_dir, video, feat_extractor):
    filter = PropagationKeypointDcf(feat_extractor, interp_factor=0.025, lambda_=0.01)

    video_keypoints = []
    for i in range(len(video)):
        frame = video.frame_at(i)
        if i == 0:
            keypoints = video.keypoints_at(i)
            filter.init(frame, keypoints)
        else:
            keypoints = filter.update(frame)

        video_keypoints.append(keypoints)
        skeleton_canvas = frame.copy()
        draw_skeleton(skeleton_canvas, keypoints, line_thickness=2)
        cv.imshow('Skeleton', skeleton_canvas)
        cv.waitKey(1)
    video_keypoints = np.stack(video_keypoints, axis=2)

    # np.save(os.path.join(out_dir, 'keypoints.npy'), video_keypoints)


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')

    feat_extractor = get_feature_extractor(
        device,
        extractor_type=args.arch,
        arch_size=args.size,
        patch_size=args.patch_size,
        scale_size=args.scale_size,
    )

    jhmdb = Jhmdb(ds_root=args.dataset_path, split_type='test', split_num=1)

    for i, video in enumerate(jhmdb):
        print(f'[{i}/{len(jhmdb)}] Video: {video.video_name}.')

        vid_out_dir = os.path.join(args.output_dir, video.video_name)
        os.makedirs(vid_out_dir, exist_ok=True)

        eval_video(vid_out_dir, video, feat_extractor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Information propagation evaluator')

    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to the dataset root.")
    parser.add_argument("--disable-cuda", action='store_true', default=False,
                        help="Disables the use of GPU compute.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where outputs are saved.")

    parser.add_argument("--keypoint-topk", type=int, default=5,
                        help="Number of pixels used to compute keypoint coordinates.")
    parser.add_argument("--keypoint-sigma", type=float, default=0.5,
                        help="Spatial spread of the label functions.")

    arch_subparsers = parser.add_subparsers(metavar='ARCH', required=True, dest='arch',
                                            help='Feature extractor architecture.')

    vit_parser = arch_subparsers.add_parser("vit")
    vit_parser.add_argument("--size", choices=['tiny', 'small', 'base'], default='small')
    vit_parser.add_argument('--patch-size', default=16, type=int, choices={8, 16},
                            help='Model patch resolution')
    vit_parser.add_argument('--scale-size', default=480, type=int,
                            help='Scale shorter side of input to this size.')

    swin_parser = arch_subparsers.add_parser("swin")

    dummy_parser = arch_subparsers.add_parser("dummy")

    uvc_parser = arch_subparsers.add_parser("uvc")
    uvc_parser.add_argument('--scale-size', default=480, type=int,
                            help='Scale shorter side of input to this size.')

    arguments = parser.parse_args()

    os.makedirs(arguments.output_dir, exist_ok=True)
    arg_file = os.path.join(arguments.output_dir, 'arguments.txt')
    utils.util.save_args(arguments, arg_file)

    with torch.no_grad():
        evaluate(arguments)
