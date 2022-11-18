"""
Script to evaluate results on the DAVIS2016 benchmark dataset.
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.davis import MultiObjectDavisDataset
from dataset.evaluation.davis_utils import f_measure, db_statistics
from utils.mask_utils import read_mask, binmask_iou
from utils.path_utils import list_subdirs


def evaluate_video(mode, video, result_dir):
    metric_names = ['J_mean', 'J_recall', 'J_decay', 'F_mean', 'F_recall', 'F_decay']

    # For each frame, compute the iou and f measure between proposal and ground truth
    iou = np.zeros([len(video), ], dtype=np.float32)
    f_meas = np.zeros_like(iou)
    for fn in range(len(video)):
        gt_mask = video.mask_at(fn)

        # FIXME: distinguish between single-object and multi-object
        proposal_mask = read_mask(os.path.join(result_dir, video.video_name, f'{fn:05d}.png'))

        # FIXME: distinguish between supervised and unsupervised evaluation
        iou[fn] = binmask_iou(proposal_mask, gt_mask)
        f_meas[fn] = f_measure(proposal_mask, gt_mask)

    # Compute summarizing mean, recall and decay statistics.
    j_mean, j_recall, j_decay = db_statistics(iou)
    f_mean, f_recall, f_decay = db_statistics(f_meas)

    # Update the cache with the results of this video.
    return pd.Series(
        data=[j_mean, j_recall, j_decay, f_mean, f_recall, f_decay],
        index=metric_names,
        name=video.video_name
    )


def evaluate_run(mode, dataset, result_dir):
    result_file = os.path.join(
        result_dir, f'DAVIS{dataset.year}_{dataset.split}_{dataset.resolution}_{mode}_results.csv')

    if not os.path.isfile(result_file):

        rows = []
        for video_idx, video in tqdm(enumerate(dataset), total=len(dataset), desc='Computing statistics'):
            rows.append(evaluate_video(mode, video, result_dir))

        df = pd.DataFrame(data=rows)
        df.to_csv(result_file)

    return pd.read_csv(result_file, index_col=0)


def evaluate_davis(args):
    print(args)
    dataset = MultiObjectDavisDataset(dataset_root=args.dataset_root, year=args.dataset_year, split=args.dataset_split,
                                      resolution=args.dataset_resolution)

    if args.eval_all:
        eval_dirs = list_subdirs(args.input_dir, relative=False)
    else:
        eval_dirs = [args.input_dir]

    for result_dir in eval_dirs:
        evaluate_run(args.mode, dataset, result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-root', type=str, default=None, required=False,
                        help='Path where the dataset is located.')
    parser.add_argument('--dataset-year', choices=['2016', '2017'], default='2017',
                        help='Year of DAVIS release.')
    parser.add_argument('--dataset-resolution', choices=['480p'], default='480p',
                        help='Resolution of the dataset frames.')
    parser.add_argument('--dataset-split', choices=['trainval', 'train', 'val'],
                        default='trainval',
                        help='Dataset split to evaluate.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory')
    parser.add_argument('--eval-all', action='store_true', default=False,
                        help='Treats the input directory as a list of evaluation runs and evaluates them all.')
    parser.add_argument('--mode', choices=['single_object', 'multi_object'], required=True,
                        help="Chose between single and multi-object modes.")

    evaluate_davis(parser.parse_args())
