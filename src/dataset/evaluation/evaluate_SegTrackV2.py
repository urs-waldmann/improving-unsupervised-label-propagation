import argparse
import os
from os.path import isfile
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.SegTrackV2 import SegTrackV2
from dataset.evaluation.davis_utils import f_measure, db_statistics
from dataset.evaluation.evaluate_segmentation_dataset import prettyprint_results
from dataset.mask_dataset import SegmentationResultReader
from utils.mask_utils import binmask_iou
from utils.path_utils import list_subdirs


def evaluate_video(mode, video, vid_result):
    # FIXME: distinguish between single-object and multi-object

    metric_names = ['J_and_F', 'J_mean', 'J_recall', 'J_decay', 'F_mean', 'F_recall', 'F_decay']

    # For each frame, compute the iou and f measure between proposal and ground truth
    iou = np.zeros([len(video), ], dtype=np.float32)
    f_meas = np.zeros_like(iou)
    for fn in range(len(video)):
        gt_mask = video.mask_at(fn)
        proposal_mask = vid_result.mask_at(fn)

        if mode == 'single-object':
            proposal_mask = proposal_mask > 0

        # FIXME: distinguish between supervised and unsupervised evaluation
        iou[fn] = binmask_iou(proposal_mask, gt_mask)
        f_meas[fn] = f_measure(proposal_mask, gt_mask)

    # Compute summarizing mean, recall and decay statistics.
    j_mean, j_recall, j_decay = db_statistics(iou)
    f_mean, f_recall, f_decay = db_statistics(f_meas)

    # Update the cache with the results of this video.
    return pd.Series(
        data=[(j_mean + f_mean) / 2, j_mean, j_recall, j_decay, f_mean, f_recall, f_decay],
        index=metric_names,
        name=video.video_name
    )


def evaluate_run(mode, dataset, result_reader, result_file: Optional[str] = None):
    if result_file is None or not isfile(result_file):
        rows = []
        for video_idx, video in tqdm(enumerate(dataset), total=len(dataset), desc='Computing statistics'):
            rows.append(evaluate_video(mode, video, result_reader[video.video_name]))

        df = pd.DataFrame(data=rows)
        if result_file is not None:
            df.to_csv(result_file)

        return df
    else:
        return pd.read_csv(result_file, index_col=0)


def evaluate_davis(args):
    dataset = SegTrackV2(dataset_root=args.dataset_root)

    if args.eval_all:
        eval_dirs = list_subdirs(args.input_dir, relative=False)
    else:
        eval_dirs = [args.input_dir]

    result_data = {}
    for result_dir in eval_dirs:
        result_reader = SegmentationResultReader(result_dir)
        result_file = os.path.join(result_dir, f'{args.mode}_results.csv')
        df = evaluate_run(
            mode=args.mode,
            dataset=dataset,
            result_reader=result_reader,
            result_file=result_file
        )

        result_data[os.path.basename(result_dir)] = df.mean()

    df = pd.DataFrame(result_data).T

    prettyprint_results(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-root', type=str, default=None, required=False,
                        help='Path where the dataset is located.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory')
    parser.add_argument('--eval-all', action='store_true', default=False,
                        help='Treats the input directory as a list of evaluation runs and evaluates them all.')
    parser.add_argument('--mode', choices=['single-object', 'multi-object'], required=True,
                        help="Chose between single and multi-object modes.")

    evaluate_davis(parser.parse_args())
