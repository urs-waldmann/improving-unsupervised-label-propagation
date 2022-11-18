from typing import List, Tuple

import numpy as np


def get_pck(gts, preds, pck_thresholds, frame_h, frame_w):
    """
    Computes the PCK metric with the given thresholds for each of the provided videos.

    :param gts: List of ground truths per video. Arrays must have shape [2, num_kp, num_frames]
    :param preds: List of predictions per video. Arrays must have shape [2, num_kp, num_frames]
    :param pck_thresholds: List of threshold values for pck computation.
    :return: List of PCK results (mean over all videos) for each threshold
    """
    assert len(gts) == len(preds)

    # Filter out keypoints that aren't visible in the video.
    joint_visibility = [compute_visibility_mask(points, frame_h, frame_w) for points in gts]
    gts = [points[:, mask, :] for points, mask in zip(gts, joint_visibility)]
    preds = [points[:, mask, :] for points, mask in zip(preds, joint_visibility)]
    box_scales = [compute_box_scale(points) for points in gts]
    distances = [compute_joint_dists(gt, pred, scale) for gt, pred, scale in zip(gts, preds, box_scales)]

    values = [compute_pck(distances, thresh) for thresh in pck_thresholds]

    return values


def compute_pck(distances, thresh):
    count_close = 0
    count_all = 0
    for dist in distances:
        count_close += np.count_nonzero(dist <= thresh)
        count_all += dist.size

    return count_close / count_all * 100.0


def compute_joint_dists(gt, pred, scales):
    distances = np.linalg.norm(pred - gt, axis=0) / scales

    # drop the first frame where gt annotations are available to the method
    distances = distances[:, 1:]

    return distances  # Shape: [num_points, num_frames - 1, ]


def compute_box_scale(points, bbox_scale=0.6):
    xy, num_keypoints, num_frames = points.shape
    assert xy == 2 and num_keypoints > 0 and num_frames > 0

    gt_max = np.max(points, axis=1)  # Shape [2, num_frames]
    gt_min = np.min(points, axis=1)  # Shape [2, num_frames]

    diff = gt_max - gt_min  # Shape: [2, num_frames], [width and height, num_frames]

    # TODO: The PCK paper calls for max(h,w) not the diagonal length
    return bbox_scale * np.linalg.norm(diff, axis=0)  # Shape: [num_frames, ]


def compute_visibility_mask(points, frame_h, frame_w):
    """
    Computes a visibility mask based on the first frame. Points that aren't visible there, should not count.

    :param points: Keypoint array of shape [2, num_keypoints, num_frames]
    :param frame_h: Frame height.
    :param frame_w: Frame width.
    :return: Mask of shape [num_keypoints, ]
    """
    w = frame_w
    h = frame_h
    xy, num_kp, num_frames = points.shape
    assert xy == 2

    p0 = points[:, :, 0]
    mask = (0 <= p0[0, :]) * (p0[0, :] < w) * (0 <= p0[1, :]) * (p0[1, :] < h)
    return mask


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
