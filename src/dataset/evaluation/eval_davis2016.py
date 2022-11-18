"""
Script to evaluate results on the DAVIS2016 benchmark dataset.
"""
import argparse
import glob
import os
from io import StringIO

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from dataset.davis import Davis2016
from utils.mask_utils import read_mask, binmask_iou
from dataset.evaluation.davis_utils import f_measure, db_statistics


def eval_davis2016(tracker_reg, dataset_split):
    metric_names = ['J_mean', 'J_recall', 'J_decay', 'F_mean', 'F_recall', 'F_decay']

    dataset_resolution = '480p'
    dataset = Davis2016(resolution=dataset_resolution, split=dataset_split)

    # Get all matching results
    tracker_paths = glob.glob(os.path.join(config.results_root, 'DAVIS2016', dataset_resolution, tracker_reg))
    tracker_paths = list(filter(os.path.isdir, tracker_paths))
    print(f'Found {len(tracker_paths)} matching results.')

    results = {}
    for tracker_path in tracker_paths:
        cache_csv = tracker_path + ".csv"

        # Load the cached results if available
        if os.path.isfile(cache_csv):
            cache_df = pd.read_csv(cache_csv, index_col=0)
            cache = {n: cache_df[n] for n in cache_df.columns}
        else:
            cache = {}

        video_names = []
        for video_idx, video in tqdm(enumerate(dataset), total=len(dataset), desc='Computing statistics'):
            video_names.append(video.video_name)

            # If we computed the results already, there is no need to do the same work again.
            if video.video_name in cache:
                continue

            # For each frame, compute the iou and f measure between proposal and ground truth
            iou = np.zeros([len(video), ], dtype=np.float32)
            f_meas = np.zeros_like(iou)
            for fn in range(len(video)):
                gt_mask = video.mask_at(fn)
                proposal_mask = read_mask(os.path.join(tracker_path, video.video_name, f'{fn:05d}.png'))
                if args.bin_thresh is not None:
                    proposal_mask = proposal_mask > args.bin_thresh

                iou[fn] = binmask_iou(proposal_mask, gt_mask)
                f_meas[fn] = f_measure(proposal_mask, gt_mask)

            # Compute summarizing mean, recall and decay statistics.
            j_mean, j_recall, j_decay = db_statistics(iou)
            f_mean, f_recall, f_decay = db_statistics(f_meas)

            # Update the cache with the results of this video.
            cache[video.video_name] = pd.Series(
                data=[j_mean, j_recall, j_decay, f_mean, f_recall, f_decay],
                index=metric_names,
                name=video.video_name)

        # Update the cache csv.
        cache_df = pd.DataFrame(cache)
        cache_df.to_csv(cache_csv)

        # Select all videos from the cache that belong to the given dataset split.
        # They are all guaranteed to exist, because if they did not we computed them earlier.
        results[tracker_path] = cache_df.loc[:, video_names]

    keys, frames = map(list, zip(*results.items()))

    # Build DataFrame from results, columns are videos, index is two-level of 1. Method, 2. Metric
    df = pd.concat(frames, keys=keys)

    # Apply reasonable names to the index columns and transform them to normal columns for processing
    df = df.reset_index().rename(columns={'level_0': 'Method', 'level_1': 'Metric'})

    # Remove the leading path of the method to retain only the directory name
    df['Method'] = df['Method'].apply(os.path.basename)

    # Restore the index of the frame to the two-level index
    df = df.set_index(['Method', 'Metric'])

    # Convert all metrics to percentages
    df = df * 100

    # Mean over all videos and then pivoting to create a table with metrics as rows and methods as columns similar to
    # the results tables of DAVIS.
    mean_table = df.mean(axis=1).reset_index().pivot('Method', 'Metric', 0)
    mean_table = mean_table.loc[:, metric_names]

    return df.T, mean_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['DAVIS2016'], metavar='DATASET',
                        help='Dataset name used for the evaluation.')
    parser.add_argument('result_glob', type=str, metavar='RESULT_GLOB',
                        help='Globbing pattern to select the result directories.')
    parser.add_argument('--bin-thresh', type=float, default=None, required=False,
                        help='Binarization threshold for predicted masks. Can be None for binary masks.')
    parser.add_argument('--split', dest='dataset_split', choices=['trainval', 'train', 'val'], default='trainval',
                        help='Dataset split to evaluate.')

    parser.add_argument('--per-video', action='store_true', default=False, required=False,
                        help='Prints the per-video results.')
    parser.add_argument('--no-summary', action='store_true', default=False, required=False,
                        help='Suppresses the printing of summarized results.')
    parser.add_argument('--latex', action='store_true', default=False, required=False,
                        help='Prints the table in latex format.')

    args = parser.parse_args()

    if args.dataset == 'DAVIS2016':
        video_results, mean_results = eval_davis2016(args.result_glob, args.dataset_split)

        if args.per_video:
            print('Per-video results')
            print(video_results.to_string(float_format='{:.1f}'.format))

        if not args.no_summary:
            print('Mean results')
            print(mean_results.to_string(float_format='{:.1f}'.format))

        if args.latex:
            with StringIO() as f:
                mean_results.to_latex(f, float_format='{:.1f}'.format)  # , encoding='utf8')
                print(f.getvalue())

    else:
        raise ValueError(f'Invalid dataset variant. {args.dataset}')
