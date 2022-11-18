import argparse
import os
from collections import OrderedDict
from typing import Tuple, Dict

import numpy as np

from dataset.BADJA import BadjaDataset
from dataset.evaluation.keypoint_eval_utils import KeypointDatasetResultLoader, KeypointDatasetEvaluator


class BadjaResultLoader(KeypointDatasetResultLoader):

    def __init__(self, dataset: BadjaDataset) -> None:
        super().__init__()

        self.dataset = dataset

        self.ground_truth = OrderedDict()
        self.frame_masks = dict()
        for vn, video in enumerate(self.dataset):
            num_frames = len(video)
            has_gt = np.zeros((num_frames,), dtype=bool)
            gt = []
            for fn in range(num_frames):
                keypoints = video.keypoints_at(fn)
                has_gt[fn] = keypoints is not None
                if keypoints is not None:
                    gt.append(keypoints)

            gt = np.stack(gt, axis=2)

            self.ground_truth[video.video_name] = gt
            self.frame_masks[video.video_name] = has_gt

    def read_gt(self) -> Dict[str, np.ndarray]:
        return self.ground_truth

    def read_pred(self, path: str) -> Dict[str, np.ndarray]:
        predictions = OrderedDict()
        for video_name, has_gt in self.frame_masks.items():
            pred_array = np.load(os.path.join(path, video_name, 'keypoints.npy'))

            # because not every frame has gt annotations we have to filter out all frames that aren't annotated
            pred_array = pred_array[:, :, has_gt]

            predictions[video_name] = pred_array

        return predictions

    def get_frame_size(self) -> Tuple[int, int]:
        # TODO: Wrong frame size!
        return 1920, 1080


def main(dataset_root, result_dir, pck_thresholds, eval_all, output_path, compute_coverage):
    if eval_all:
        method_paths = [os.path.join(result_dir, d) for d in os.listdir(result_dir)]
        method_paths = [p for p in method_paths if os.path.isdir(p)]
    else:
        method_paths = [result_dir]

    dataset = BadjaDataset(dataset_root)
    result_loader = BadjaResultLoader(dataset)
    evaluator = KeypointDatasetEvaluator(result_loader, method_paths, verbose=True)
    evaluator.print_per_video_results(thresholds=pck_thresholds)
    # TODO: Does not work at the moment
    # evaluator.print_table(pck_thresholds, compute_coverage=compute_coverage, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate the PCK metric on keypoint datasets")
    parser.add_argument('result_dir', type=str,
                        help='Directory where the per-video results are stored.')
    parser.add_argument('--output-path', type=str, required=False, default=None,
                        help="File path where the output should be written (format=csv)")
    parser.add_argument('--eval-all', action='store_true', default=False,
                        help='Treats the input path as a directory where different runs are stored.')
    parser.add_argument('--compute-coverage', action='store_true', default=False,
                        help='Computes the number of ground-truth points that are covered at all.')

    parser.add_argument('--dataset-dir', default=None, required=False,
                        help='Path to the dataset root.')

    args = parser.parse_args()

    main(
        dataset_root=args.dataset_dir,
        result_dir=args.result_dir,
        pck_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        eval_all=args.eval_all,
        output_path=args.output_path,
        compute_coverage=args.compute_coverage
    )
