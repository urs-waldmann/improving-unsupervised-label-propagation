import argparse
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from dataset.BADJA import BadjaDataset
from dataset.PigeonDataset import get_pigeon_dataset
from utils.vis_utils import davis_color_map


def main(dataset_root, result_dir):
    dataset = BadjaDataset(dataset_root=dataset_root, clip_to_first_annotation=True)
    color_map = davis_color_map()

    pause = True
    for video in tqdm(dataset):
        pred_path = os.path.join(result_dir, video.video_name, 'keypoints.npy')
        pred = np.load(pred_path)

        label_path = os.path.join(result_dir, video.video_name, 'labelmap.npy')
        label = np.load(label_path) if os.path.isfile(label_path) else None

        if label is not None:
            import matplotlib.pyplot as plt
            a = label[:20, ...]
            num_frames, num_masks, h, w = a.shape

            a = np.transpose(a, (1, 2, 0, 3))
            a = a.reshape((num_masks * h, num_frames * w))

            plt.imshow(a)
            plt.show()

        i = 0
        while i < len(video):
            frame = video.frame_at(i)
            ground_truth = video.keypoints_at(i)

            canvas = frame.copy()
            for j in range(pred.shape[1]):
                try:
                    point = (int(pred[0, j, i]), int(pred[1, j, i]))
                except ValueError:
                    # Ignore nan values caused by conversion
                    point = None

                if point is not None:
                    cv.drawMarker(canvas, point, (0, 255, 0))

                if ground_truth is not None:
                    gt_point = (int(ground_truth[0, j]), int(ground_truth[1, j]))
                    cv.drawMarker(canvas, gt_point, (0, 0, 255))

            if label is not None:
                labels = label[i, ...]

                c, h, w = labels.shape
                for m in range(1, c):
                    vis_mask = cv.resize(labels[m, :, :], (1920, 1080))[..., np.newaxis]
                    canvas = (1 - vis_mask) * canvas + vis_mask * color_map[m].reshape(1, 1, 3)
                canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            cv.imshow('Frame', canvas)
            if pause:
                k = cv.waitKey()
            else:
                k = cv.waitKey(5)

            if k == ord('p'):
                pause = not pause
            elif k == ord('q'):
                return
            elif k == ord('n'):
                break
            elif k == ord('r'):
                i -= 1
            else:
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str,
                        help='Directory where the per-video results are stored.')
    parser.add_argument('--dataset-dir', default=None, required=False,
                        help='Path to the dataset root.')
    args = parser.parse_args()

    main(
        dataset_root=args.dataset_dir,
        result_dir=args.result_dir,
    )
