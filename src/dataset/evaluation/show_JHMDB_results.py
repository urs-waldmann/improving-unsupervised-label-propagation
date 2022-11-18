import argparse
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

import config
from dataset.JHMDB import Jhmdb
from utils.keypoint_visualization import draw_skeleton, JHMDB_SKELETON_BICOLOR, JHMDB_SKELETON


def main(dataset_root, result_dir, dataset_split_num, dataset_split_name, dataset_load_limit):
    dataset = Jhmdb(ds_root=dataset_root, split_type=dataset_split_name, split_num=dataset_split_num,
                    per_class_limit=dataset_load_limit)

    for video in tqdm(dataset):
        pred_path = os.path.join(result_dir, video.video_name, 'keypoints.npy')
        pred = np.load(pred_path)

        for i, frame_info in enumerate(video):
            frame = frame_info['frame']
            ground_truth = frame_info['keypoints']

            canvas = frame.copy()

            draw_skeleton(canvas, ground_truth, JHMDB_SKELETON, line_thickness=4)
            draw_skeleton(canvas, pred[..., i], JHMDB_SKELETON_BICOLOR, line_thickness=4)

            # for j in range(pred.shape[1]):
            #     point = None
            #     try:
            #         point = (int(pred[0, j, i]), int(pred[1, j, i]))
            #     except ValueError:
            #         # Ignore nan values caused by conversion
            #         pass
            #     if point is not None:
            #         cv.drawMarker(canvas, point, (0, 255, 0))
            #
            #     gt_point = (int(ground_truth[0, j]), int(ground_truth[1, j]))
            #     cv.drawMarker(canvas, gt_point, (0, 0, 255))

            cv.imshow('Frame', canvas)
            k = cv.waitKey()
            if k == ord('q'):
                return
            elif k == ord('n'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate the PCK metric on JHMDB keypoints")
    parser.add_argument('result_dir', type=str,
                        help='Directory where the per-video results are stored.')
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
        dataset_split_num=args.dataset_split_num,
        dataset_split_name=args.dataset_split_name,
        dataset_load_limit=args.dataset_load_limit
    )
