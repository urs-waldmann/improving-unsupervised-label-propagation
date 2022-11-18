import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Sequence, Iterable, Dict

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm


def compute_pcks(joint_distances, thresholds):
    """
    Computes the PCK value for the given distance array and each of the provided threshold values.

    :param joint_distances: Distances between prediction and ground truth joint location. Shape: [num_keypoints, num_frames].
    :param thresholds: int,list or array of threshold values.
    :return: Array of pck values corresponding to the threshold values.
    """
    thresholds = np.atleast_1d(thresholds)
    assert thresholds.ndim == 1 and joint_distances.ndim == 2

    num_close_joints = np.count_nonzero(joint_distances[..., np.newaxis] <= thresholds, axis=1)
    num_all_joints = np.count_nonzero(~np.isnan(joint_distances), axis=1)
    pck = 100.0 * num_close_joints / num_all_joints[:, np.newaxis]

    return np.mean(pck, axis=0)


def compute_joint_distances(gts, thresh_radii, jnt_visible_set, predictions):
    """
    Computes the distances between joints and ground truths relative to the corresponding thresh radius.

    :param gts:             List of ground truths per video. Arrays must have shape [2, num_kp, num_frames]
    :param thresh_radii:    List of ground truths per video. Arrays must have shape [num_frames]
    :param jnt_visible_set: List of ground truths per video. Arrays must have shape [num_kp, num_frames]
    :param predictions:     List of predictions per video. Arrays must have shape [2, num_kp, num_frames]
    :return:
    """
    all_distances = []
    for gt, radii, pred, jv in zip(gts, thresh_radii, predictions, jnt_visible_set):
        # gt:    [2, num_keypoints, num_frames]
        # pred:  [2, num_keypoints, num_frames]
        # radii: [num_frames]
        # jv:    [num_keypoints, num_frames]

        distances = np.linalg.norm(pred - gt, axis=0) / radii

        # Set distance to nan for all invisible joints
        distances[np.logical_not(jv)] = np.nan

        # drop the first frame where gt annotations are available to the method
        distances = distances[:, 1:]

        all_distances.append(distances)

    return np.concatenate(all_distances, axis=1)  # shape: [num_keypoints, num_all_frames]


def compute_bbox_scales(gts, joint_visible_set, *, method_name='keypoint_box'):
    from utils.keypoint_visualization import JHMDB_JOINTS

    def head_bone(ground_truth, joint_visible):
        head = ground_truth[:, JHMDB_JOINTS['Head'], :]
        upper_back = ground_truth[:, JHMDB_JOINTS['UpperBack'], :]

        return np.linalg.norm(upper_back - head, axis=0)

    def torso_diam(ground_truth, joint_visible):
        upper_back = ground_truth[:, JHMDB_JOINTS['UpperBack'], :]
        lower_back = ground_truth[:, JHMDB_JOINTS['LowerBack'], :]
        return np.linalg.norm(lower_back - upper_back, axis=0)

    def keypoint_box(ground_truth, joint_visible):
        gt2 = ground_truth.copy()
        gt2[:, np.logical_not(joint_visible)] = np.nan

        gt_max = np.nanmax(gt2, axis=1)  # Shape [2, num_frames]
        gt_min = np.nanmin(gt2, axis=1)  # Shape [2, num_frames]

        # TODO: What can we do if all joints are invisible?
        return 0.6 * np.linalg.norm(gt_max - gt_min, axis=0)  # Shape [num_frames]

    if method_name == 'keypoint_box':
        method = keypoint_box
    elif method_name == 'torso_diam':
        method = torso_diam
    elif method_name == 'head_bone':
        method = head_bone
    else:
        raise ValueError(f'Invalid method name {method_name}')

    return [method(gt, jv) for gt, jv in zip(gts, joint_visible_set)]


def compute_pck_for_dataset(gts, preds, pck_thresholds=None, radius_method='keypoint_box'):
    """
    Computes the PCK metric with the given thresholds for each of the provided videos.

    :param gts: List of ground truths per video. Arrays must have shape [2, num_kp, num_frames]
    :param preds: List of predictions per video. Arrays must have shape [2, num_kp, num_frames]
    :param pck_thresholds: List of threshold values for pck computation.
    :return: List of PCK results (mean over all videos) for each threshold
    """
    assert len(gts) == len(preds)

    is_joint_visible = [np.float32(points[0] >= 0) for points in preds]

    thresh_radii = compute_bbox_scales(gts, is_joint_visible, method_name=radius_method)
    joint_distances = compute_joint_distances(gts, thresh_radii, is_joint_visible, preds)

    values = list(compute_pcks(joint_distances, pck_thresholds))

    return values


def compute_overall_coverage(gts: List[np.ndarray], preds: List[np.ndarray], *, size=(320, 240),
                             use_weighted_mean=True):
    """
    Computes the keypoint coverage (Percentage of keypoints covered by predictions independent of prediction
    correctness).

    :param gts: List of ground truths per video. Arrays must have shape [2, num_kp, num_frames]
    :param preds: List of predictions per video. Arrays must have shape [2, num_kp, num_frames]
    :param size: Tuple of width and height or List with one tuple of width and height for each video.
    :param use_weighted_mean: True to let each video contribute proportional to the video length, otherwise all contribute equally
    :return:
    """

    def coverage(ground_truth, predictions):
        # shapes: [2, num_kp, num_frames]

        # Number of visible keypoints per frame. Shape: [num_frames, ]
        predictions = np.all(predictions > 0, 0).sum(0)
        ground_truth = np.all(ground_truth > 0, 0).sum(0)

        # Clamp predictions number to ground_truth, this ensures that the over-prediction in one frame does not
        # transfer to other frames, artificially improving the coverage.
        sel = predictions > ground_truth
        predictions[sel] = ground_truth[sel]

        # Fraction of keypoints visible per frame, averaged over the entire video
        return (predictions / ground_truth).mean() * 100.0

    num_videos = len(gts)
    assert len(preds) == num_videos

    if isinstance(size, Tuple):
        assert len(size) == 2

        def get_size(vid_num):
            return size
    elif isinstance(size, List):
        assert len(size) == num_videos

        def get_size(vid_num):
            return size[vid_num]
    else:
        raise ValueError(f'Invalid value of "size": {size}. Must be List or Tuple.')

    weights = np.zeros((len(gts), 1), dtype=np.float64)
    output = np.zeros((len(gts), 2), dtype=np.float64)
    for video_number, (gt, pred) in enumerate(zip(gts, preds)):
        w, h = get_size(video_number)
        xy, num_kp, num_frames = gt.shape
        assert xy == 2

        p0 = gt[:, :, 0]
        mask = (0 <= p0[0, :]) * (p0[0, :] < w) * (0 <= p0[1, :]) * (p0[1, :] < h)

        weights[video_number, 0] = num_frames
        output[video_number, 0] = coverage(gt, pred)
        output[video_number, 1] = coverage(gt[:, mask, :], pred[:, mask, :])

    if use_weighted_mean:
        return (output * weights).sum(axis=0) / weights.sum()
    else:
        return output.mean(axis=0)


class KeypointDatasetResultLoader(ABC):
    @abstractmethod
    def read_gt(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def read_pred(self, path: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def get_frame_size(self) -> Tuple[int, int]:
        raise NotImplementedError()


class KeypointDatasetEvaluator(ABC):

    def __init__(
            self,
            result_loader: KeypointDatasetResultLoader,
            method_paths: List[str],
            method_names: List[str] = None,
            *,
            verbose=True
    ):
        self.verbose = verbose

        self.method_paths = method_paths

        if method_names is not None:
            assert len(method_names) == len(self.method_paths)
            self.method_names = method_names
        else:
            self.method_names = [os.path.basename(p) for p in self.method_paths]

        self.frame_size = result_loader.get_frame_size()

        self._emit_msg('Loading Dataset...')
        self.ground_truth_map = result_loader.read_gt()

        self.video_names = sorted(self.ground_truth_map.keys())
        self.ground_truth = [self.ground_truth_map[k] for k in self.video_names]

        self._emit_msg('Loading predictions:')
        self.prediction_map = OrderedDict()
        self.predictions = OrderedDict()
        for p in self._with_progress(self.method_paths):
            preds = result_loader.read_pred(p)
            self.prediction_map[p] = preds
            self.predictions[p] = [preds[k] for k in self.video_names]
        self._emit_msg('Predictions loaded.')

    def _with_progress(self, seq: Sequence) -> Iterable:
        if self.verbose:
            return tqdm(seq, total=len(seq))
        else:
            return seq

    def _emit_msg(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def compute_statistics(self, compute_coverage,
                           pck_thresholds,
                           *,
                           radius_method='keypoint_box'):

        self._emit_msg('Evaluating results:')

        results = OrderedDict()
        for method_dir, preds in self._with_progress(self.predictions.items()):
            values = compute_pck_for_dataset(
                self.ground_truth, preds, pck_thresholds=pck_thresholds, radius_method=radius_method)

            if compute_coverage:
                c1, c2 = compute_overall_coverage(self.ground_truth, preds, size=self.frame_size)
                values += [c1, c2]

            results[method_dir] = values
        return results

    def print_per_video_results(self, thresholds: Union[float, List[float], np.ndarray]):
        for method_name, (method_dir, preds) in zip(self.method_names, self._with_progress(self.predictions.items())):

            print(method_name)
            vid_rows = []
            for vn, a, b in zip(self.video_names, self.ground_truth, preds):
                vid_rows.append([vn] + list(compute_pck_for_dataset([a], [b], pck_thresholds=thresholds)))
            print(pd.DataFrame(vid_rows).to_string())

    def print_table(self,
                    thresholds: Union[float, List[float], np.ndarray],
                    *,
                    compute_coverage: bool = True,
                    output_path: Optional[str] = None):

        results = self.compute_statistics(compute_coverage, pck_thresholds=thresholds)

        columns = ['Method'] + thresholds + (['c1', 'c2'] if compute_coverage else [])
        rows = []

        result_table = PrettyTable(columns)
        result_table.align['Method'] = 'l'

        for method_name, (method_dir, values) in zip(self.method_names, results.items()):
            rows.append([method_name] + values)
            result_table.add_row([method_name] + ['{:.02f}'.format(v) for v in values])

        print(result_table.get_string())

        if output_path is not None:
            import pandas as pd

            df = pd.DataFrame(data=rows, columns=columns)
            df.to_csv(output_path)
