"""
Utility functions to compute statistics and evaluation metrics for the OTB benchmark datasets.
"""

import argparse
import glob
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.integrate import trapz
from tqdm import tqdm

import config
from dataset import OtbDataset
from utils import bbox
from utils.bbox_numpy import iou_xywh, box_xywh_center_distances


def compute_success_overlap(gt_boxes: np.ndarray, pred_boxes: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Computes the percentage of boxes with an overlap with the ground truth above the threshold.

    :param gt_boxes: [n, 4] ground truth bounding boxes
    :param pred_boxes: [n, 4] prediction bounding boxes
    :param thresholds: [t] overlap thresholds
    :return: Array [t] of success overlap per threshold
    """
    thresholds = np.atleast_1d(thresholds)
    assert thresholds.ndim == 1

    iou = iou_xywh(gt_boxes, pred_boxes)

    # same as np.count_nonzero(iou > thresholds, axis=1) / len(gt_bb)
    return np.mean(iou[np.newaxis, :] > thresholds[:, np.newaxis], axis=1)


def compute_success_error(gt_boxes: np.ndarray, pred_boxes: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Computes the percentage boxes with distance to the ground truth below threshold.

    :param gt_boxes: [n, 4] ground truth bounding boxes
    :param pred_boxes: [n, 4] prediction bounding boxes
    :param thresholds: [t] distance thresholds
    :return: [t] array of success error per threshold
    """
    thresholds = np.atleast_1d(thresholds)
    assert thresholds.ndim == 1

    dists = box_xywh_center_distances(gt_boxes, pred_boxes)
    return np.mean(dists[np.newaxis, :] <= thresholds[:, np.newaxis], axis=1)


def get_result_bboxes(tracker_path: str, video_name: str) -> np.ndarray:
    """
    Wrapper function for bbox result loading. Unfortunately, the ground truth and the results of Wang et al. have
    inconsistent formats, therefore this loading routine is necessary.

    :param tracker_path: path to the directory where the results are stored.
    :param video_name: canonical name of the video as used in the dataset definition.
    :return: np.ndarray with bboxes. Shape: [N,4] in xywh format.
    """

    # Mapping for inconsistencies in naming: canoncial name -> used name
    name_map = {
        'Jogging_1': 'Jogging',
        'Jogging_2': 'Jogging(2)',
        'Skating2_1': 'Skating2',
        'Skating2_2': 'Skating2(2)',
        'Human4': 'Human4_2',
    }

    result_path = os.path.join(tracker_path, video_name + '.txt')
    if not os.path.exists(result_path):
        # File with canonical name does not exist, retry with mapped name.
        vn = name_map.get(video_name, video_name)
        result_path = os.path.join(tracker_path, vn + '.txt')
        if not os.path.exists(result_path):
            # Mapped name also missing -> We don't know where to find the file.
            raise IOError(f'File {result_path} is missing.')

    # First, try to load with ',' delimiter, if this fails try with ' '. If this fails too, a ValueError is thrown.
    try:
        temp = np.loadtxt(result_path, delimiter=',').astype(np.float64)
    except ValueError:
        temp = np.loadtxt(result_path, delimiter=' ').astype(np.float64)

    return np.array(temp)


def compute_otb_metrics(dataset, thresholds_error, thresholds_overlap, tracker_paths):
    # Per video, per tracker, per threshold statistics over all frames of a video.
    success_overlap = np.zeros((len(dataset), len(tracker_paths), len(thresholds_overlap)))
    success_error = np.zeros((len(dataset), len(tracker_paths), len(thresholds_error)))
    # Compute the AUC metric for all tracker, dataset and threshold combinations
    for video_idx, video in tqdm(enumerate(dataset), total=len(dataset), desc='Computing statistics'):
        gt = bbox.boxes_as_xywh(video.gt_boxes())

        for tracker_idx, tracker_path in enumerate(tracker_paths):

            bb = get_result_bboxes(tracker_path, video.video_name)

            if bb.shape != gt.shape:
                # There are a few existing results where the size of the ground truth does not match the actual result
                # length because a few frames are missing in the results. This has only minimal effect on the
                # performance, thus we still want to compute the results.
                print(video.video_name)
                warnings.warn(f'Shapes of gt and bb differ: bb: {bb.shape}, gt: {gt.shape}.', RuntimeWarning)
                diff = gt.shape[0] - bb.shape[0]
                gt2 = gt[diff:, :]
                print(f'Truncating front. {gt2.shape}')
            else:
                gt2 = gt

            success_overlap[video_idx, tracker_idx] = compute_success_overlap(gt2, bb, thresholds_overlap)
            success_error[video_idx, tracker_idx] = compute_success_error(gt2, bb, thresholds_error)
    return success_error, success_overlap


def evaluate_otb(tracker_paths: List[str], *, show_plot: bool, show_table: bool, dataset_name: str):
    """
    Compute success error and success overlap statistics of the dataset.

    Success Overlap: AUC with IoU thresholds of 0 to 1 in 0.05 increments.
    Success Error: Distance Precision with thresholds between 1 and 50 pixels, 1px increments.

    AUC of success plot (#IoU>thresh vs thresh) == mean IoU for number of thresholds -> infinity

    :param tracker_paths: List of paths to the results to evaluate
    :param show_plot: True to show result plots of the result or False to work without GUI.
    :param show_table: True to show table of results.
    :param dataset_name: Name of the OTB variant to use for evaluation. Either OTB2015 or OTB2013.
    """
    tracker_names = [os.path.basename(p) for p in tracker_paths]

    thresholds_overlap = np.arange(0, 1.05, 0.05)
    thresholds_error = np.arange(0, 51, 1)

    dataset = OtbDataset(variant=dataset_name)
    success_error, success_overlap = compute_otb_metrics(dataset, thresholds_error, thresholds_overlap, tracker_paths)

    # if only_2013:
    #     video_indices = [i for i, video in enumerate(dataset) if video.contained_in('OTB2013')]
    #     success_error = success_error[video_indices, :, :]
    #     success_overlap = success_overlap[video_indices, :, :]

    se_per_tracker = success_error.mean(axis=0)
    so_per_tracker = success_overlap.mean(axis=0)

    # Distance precision with 20px threshold, times 100 to obtain percentages.
    dp20 = se_per_tracker[:, np.searchsorted(thresholds_error, 20)] * 100
    # The AUC is the area under the success plot, times 100 to obtain percentages.
    auc = trapz(so_per_tracker, thresholds_overlap, axis=1) * 100

    if show_table:
        df = pd.DataFrame({
            'Tracker': tracker_names,
            'AUC': auc,
            'DP': dp20,
        })
        df = df.set_index('Tracker')
        with pd.option_context('display.precision', 2):
            print(df.to_string())

        table = PrettyTable()
        table.field_names = ['Tracker', 'AUC', 'DP']
        table.align['Tracker'] = 'l'
        table.align['AUC'] = 'r'
        table.align['DP'] = 'r'
        table_auc = np.round(auc, 2)
        table_dp20 = np.round(dp20, 2)
        for i in range(len(tracker_names)):
            table.add_row([tracker_names[i], table_auc[i].item(), table_dp20[i].item()])
        auc_max = np.argmax(table_auc).item()
        dp_max = np.argmax(table_dp20).item()
        print(table.get_string())
        print(f'Best AUC: {table_auc[auc_max].item():.4f} ({tracker_names[auc_max]})')
        print(f'Best DP:  {table_dp20[dp_max].item():.4f} ({tracker_names[dp_max]})')

    if show_plot:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(dataset_name)
        se_legend = [f'{n:s} [{dp.item():.01f}]' for n, dp in zip(tracker_names, dp20)]
        so_legend = [f'{n:s} [{a.item():.01f}]' for n, a in zip(tracker_names, auc)]
        cc = cycler(linestyle=['-', '--', ':', '-.']) * cycler(color=plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10)))

        ax[0].set_prop_cycle(cc)
        ax[0].plot(se_per_tracker.T)
        ax[0].set_xlabel('Location error threshold (pixels)')
        ax[0].set_ylabel('Precision rate')
        ax[0].set_title('Precision plots of OPE')
        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(thresholds_error[0], thresholds_error[-1])
        ax[0].legend(se_legend)

        ax[1].set_prop_cycle(cc)
        ax[1].plot(thresholds_overlap, so_per_tracker.T)
        ax[1].set_xlabel('Overlap threshold')
        ax[1].set_ylabel('Success rate')
        ax[1].set_title('Success plots of OPE')
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(0, 1)
        ax[1].legend(so_legend)

        plt.show()


def eval_auc(tracker_glob: str = '*', show_plot: bool = True, dataset_name: str = 'OTB2015'):
    tracker_paths = glob.glob(os.path.join(config.results_root, 'OTB2015', tracker_glob))
    print(f'Found {len(tracker_paths)} matching trackers.')

    return evaluate_otb(tracker_paths, show_plot=show_plot, show_table=True, dataset_name=dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute Distance Precision (DP) and Area under Curve (AUC) for tracking results.")

    parser.add_argument('tracker', type=str, metavar='TRACKER_GLOB',
                        help='Globbing pattern to select trackers relative to the OTB2015 result root.')
    parser.add_argument('--dataset-name', type=str, choices={'OTB2015', 'OTB2013'}, default='OTB2015',
                        help='Dataset variant to use for the evaluation.')
    args = parser.parse_args()

    eval_auc(args.tracker, dataset_name=args.dataset_name)
