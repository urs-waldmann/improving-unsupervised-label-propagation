import argparse
import os
from collections import OrderedDict
from typing import Tuple, Dict

import numpy as np

import config
from dataset.JHMDB import Jhmdb
from dataset.evaluation.keypoint_eval_utils import KeypointDatasetResultLoader, KeypointDatasetEvaluator


class JhmdbResultLoader(KeypointDatasetResultLoader):

    def __init__(self, dataset: Jhmdb):
        assert dataset is not None

        self.dataset = dataset

    def read_gt(self) -> Dict[str, np.ndarray]:
        ground_truth = OrderedDict()
        for vid_num, video in enumerate(self.dataset):
            ground_truth[video.video_name] = video.get_all_keypoints()

        return ground_truth

    def read_pred(self, path: str) -> Dict[str, np.ndarray]:
        predictions = OrderedDict()
        for video_name in self.dataset.video_names():
            pred_array = np.load(os.path.join(path, video_name, 'keypoints.npy'))

            if pred_array.shape[1] == 16:
                pred_array = pred_array[:, 1:, ...]

            predictions[video_name] = pred_array

        return predictions

    def get_frame_size(self) -> Tuple[int, int]:
        return 320, 240


def main(dataset_root, result_dir, pck_thresholds, eval_all,
         dataset_split_num, dataset_split_name, dataset_load_limit, output_path, compute_coverage):
    if eval_all:
        method_paths = [os.path.join(result_dir, d) for d in os.listdir(result_dir)]
        method_paths = [p for p in method_paths if os.path.isdir(p)]
    else:
        method_paths = [result_dir]

    dataset = Jhmdb(
        ds_root=dataset_root,
        split_type=dataset_split_name,
        split_num=dataset_split_num,
        per_class_limit=dataset_load_limit
    )
    result_loader = JhmdbResultLoader(dataset)
    evaluator = KeypointDatasetEvaluator(result_loader, method_paths, verbose=True)
    evaluator.print_table(pck_thresholds, compute_coverage=compute_coverage, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate the PCK metric on JHMDB keypoints")
    parser.add_argument('result_dir', type=str,
                        help='Directory where the per-video results are stored.')
    parser.add_argument('--output-path', type=str, required=False, default=None,
                        help="File path where the output should be written (format=csv)")
    parser.add_argument('--eval-all', action='store_true', default=False,
                        help='Treats the input path as a directory where different runs are stored.')
    parser.add_argument('--compute-coverage', action='store_true', default=False,
                        help='Computes the number of ground-truth points that are covered at all.')
    parser.add_argument('--dataset-dir', default=config.dataset_jhmdb_path,
                        help='Path to the JHMDB dataset root.')
    parser.add_argument('--dataset-split-num', type=int, default=1,
                        help='Dataset split number.')
    parser.add_argument('--dataset-split-name', type=str, default='test', choices=['test', 'train'],
                        help='Dataset split name.')
    parser.add_argument('--dataset-load-limit', type=int, default=-1,
                        help='Per class limit of sequences to process.')
    args = parser.parse_args()

    main(
        dataset_root=args.dataset_dir,
        result_dir=args.result_dir,
        pck_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        eval_all=args.eval_all,
        dataset_split_num=args.dataset_split_num,
        dataset_split_name=args.dataset_split_name,
        dataset_load_limit=args.dataset_load_limit,
        output_path=args.output_path,
        compute_coverage=args.compute_coverage
    )
