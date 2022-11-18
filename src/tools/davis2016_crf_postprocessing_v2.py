"""
Helper script to perform CRF-based mask post-processing for DAVIS2016 results. The post-processing uses very bad
settings for the post-processing. Don't expect good results.
"""
import argparse
import os

from tqdm import tqdm

from dataset.davis import Davis2016
from utils.mask_utils import read_mask, write_mask, refine_mask


def main(args):
    def safe_mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
        elif os.listdir(path) and not args.allow_overwrite:
            raise FileExistsError('The specified output directory is not empty.')

    out_dir = args.out_path
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir) and not args.allow_overwrite:
        raise FileExistsError('The specified output directory is not empty.')

    for tracker_path in args.in_paths:
        if len(args.in_paths) > 1:
            tracker_out_dir = os.path.join(out_dir, os.path.basename(tracker_path))
            safe_mkdir(tracker_out_dir)
        else:
            tracker_out_dir = out_dir

        dataset = Davis2016(resolution='480p', split='trainval')

        for video_idx, video in tqdm(enumerate(dataset), total=len(dataset), desc='Computing statistics'):
            video_out_dir = os.path.join(tracker_out_dir, video.video_name)
            safe_mkdir(video_out_dir)

            for fn in range(len(video)):
                mask_name = f'{fn:05d}.png'

                frame = video.frame_at(fn)
                in_mask = read_mask(os.path.join(tracker_path, video.video_name, mask_name))

                out_mask = refine_mask(frame, in_mask, num_steps=args.num_iterations, gt_prob=args.gt_prob)

                write_mask(os.path.join(video_out_dir, mask_name), out_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-overwrite',
                        action='store_true',
                        default=False,
                        help='Allows overwriting of existing results.')
    parser.add_argument('--gt-prob',
                        type=float,
                        default=0.99,
                        required=False,
                        help='Confidence in the initial mask prediction.')
    parser.add_argument('--num-iterations',
                        type=int,
                        default=5,
                        required=False,
                        help='Number of refinement iterations.')
    parser.add_argument('in_paths',
                        type=str,
                        nargs='+',
                        help='Input paths.')
    parser.add_argument('out_path',
                        type=str,
                        help='Output dir. Tracker subdirectories are created if more than one is given.')

    main(parser.parse_args())
