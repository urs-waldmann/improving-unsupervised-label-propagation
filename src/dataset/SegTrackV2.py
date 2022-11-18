import glob
import os
from os.path import join
from typing import Optional, List

import cv2 as cv
import numpy as np

import config
from dataset.mask_dataset import MaskDataset, MaskVideo, SimpleMaskVideo


class SegTrackV2Video(SimpleMaskVideo):
    num_objects: int
    _root: str
    _imageset_root: str
    _jpeg_root: str
    _frame_paths: List[str]
    _mask_paths: List[List[str]]

    def __init__(self, dataset_path: str, video_name: str):
        self.video_name = video_name
        self._root = dataset_path
        self._imageset_root = join(self._root, 'ImageSets')
        self._jpeg_root = join(self._root, 'JPEGImages')

        # Format: first line is the path prefix and following lines are the file names
        with open(join(self._imageset_root, video_name + '.txt'), 'r', encoding='utf8') as f:
            path_prefix = f.readline().rstrip('/\n')  # strip trailing slash and newline from path prefix
            images = list(map(lambda l: l.rstrip('\n'), f.readlines()))
        assert len(images) > 0

        # Build path prefixes for images and ground truth files
        image_prefix = join(self._jpeg_root, path_prefix)
        # Ground truth files are in numbered sub-directories if there is more than one object. Thus we need to handle
        # mutliple paths.
        gt_prefix = join(self._root, 'GroundTruth', path_prefix)
        gt_objects = sorted(filter(os.path.isdir, map(lambda x: join(gt_prefix, x), os.listdir(gt_prefix))))
        gt_obj_paths = [gt_prefix] if len(gt_objects) == 0 else gt_objects

        assert len(gt_obj_paths) > 0
        assert all(i + 1 == int(os.path.split(j)[-1]) for i, j in enumerate(gt_objects))
        self.num_objects = len(gt_obj_paths)

        def determine_extension(prefix):
            # Find the file extension of image files and raise error for ambiguities
            first_img_candidates = glob.glob(join(prefix, images[0] + '.*'))
            image_exts = list(map(lambda n: os.path.splitext(n)[-1], first_img_candidates))
            assert len(image_exts) == 1, 'There are multiple images with the same name but different file extension.'
            return image_exts[0]

        # Find out the file extension of the ground truth and image files. The extensions are png and bmp but they are
        # not used consistently.
        image_ext = determine_extension(image_prefix)
        gt_exts = [determine_extension(p) for p in gt_obj_paths]

        def build_img_list(item_producer):
            return list(map(item_producer, images))

        frame_paths = build_img_list(
            lambda n: join(image_prefix, n + image_ext))
        mask_paths = build_img_list(
            lambda n: [join(prefix, n + gt_ext) for prefix, gt_ext in zip(gt_obj_paths, gt_exts)])

        super().__init__(video_name, frame_paths, mask_paths)

    def __repr__(self) -> str:
        return f'SegTrackV2Video(name={self.video_name}, ' \
               f'len={len(self)}, ' \
               f'num_objects={self.num_objects}, ' \
               f'resolution={self.get_resolution()})'

    def mask_at(self, index: int) -> np.ndarray:
        mask_paths = self._mask_paths[index]

        masks = [cv.imread(mask_path, cv.IMREAD_GRAYSCALE) > 0 for mask_path in mask_paths]
        mask = np.zeros_like(masks[0], dtype=np.uint8)
        for i, m in enumerate(masks):
            mask[m] = i + 1

        return mask


class SegTrackV2(MaskDataset):
    """
    Dataset loader for the SegTrackV2 dataset.

    Source: https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip
    """

    def __init__(self, dataset_root: Optional[str] = None):
        self._root = dataset_root if dataset_root is not None else config.dataset_segtrackv2_path
        super().__init__()

    def _read_video_names(self) -> List[str]:
        with open(join(self._root, 'ImageSets', 'all.txt'), 'r', encoding='utf8') as f:
            video_names = list(map(lambda n: n.lstrip('*').rstrip('\n'), f.readlines()))
        return video_names

    def _create_video(self, video_name) -> MaskVideo:
        return SegTrackV2Video(self._root, video_name)


if __name__ == '__main__':
    from utils.vis_utils import draw_semantic_masks, davis_color_map
    import numpy as np

    dataset = SegTrackV2()
    single_object = False

    color_map = davis_color_map()
    for video in dataset:
        print(video)
        for frame_info in video:
            frame = frame_info['frame']
            mask = frame_info['mask']
            if single_object:
                mask = (mask > 0).astype(np.uint8)
            frame = draw_semantic_masks(frame, mask, color_map, draw_outline=True)
            cv.imshow('Frame', frame)
            cv.waitKey(1)
