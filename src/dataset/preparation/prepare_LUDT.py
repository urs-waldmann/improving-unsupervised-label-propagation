import argparse
import json
import os
from typing import List

import cv2 as cv
import numpy as np
import scipy.stats
from tqdm import tqdm

import config
from utils.bbox import BBox
from utils.image_utils import crop_sampling, cv_to_torch, torch_to_cv
from tracking.kcf.kcf import KCFTracker
from tracking.kcf.kcf_matlab import matlab_kcf_tracker
from utils.util import chunked


class LUDTExtractor:
    """
    This class implements the preprocessing of video data for training of according to LUDT (Wang. et al., 2020).

    For each input video a fixed number of consecutive frames is selected. In the first frame a grid of bounding boxes
    is used as potential starting points. The boxes some overlap and are padded to the image borders, such that objects
    don't leave the frame immediately. The entropy for each box is computed and the box with highest entropy initializes
    the track extraction. The boxes in following frames are selected with a KCF tracker which was initialized with the
    initial box. Each track box is padded relative to the output size. Then the contents are cropped from the frame and
    rescaled to the output size.
    """

    def __init__(self,
                 grid_cell_size=12,
                 grid_block_size=2,
                 grid_padding=3,
                 output_size=(125, 125),
                 output_padding_factor=2,
                 num_frames_per_video=100,
                 chunked_preprocessing=False,
                 grayscale_entropy=False,
                 crop_padding_mode='zeros',
                 ):
        """
        :param grid_cell_size: number of grid cells in the first frame entropy selection
        :param grid_block_size: number of grid cells occupied by each selectable block in the entropy selection
        :param grid_padding: number of grid cells padding the the blocks to the top and left in entropy selection
        :param output_size: output size of saved image patches
        :param num_frames_per_video: number of frames to preprocess per video
        :param chunked_preprocessing: True to enable chunking of videos instead of dropping frames exceeding
        num_frames_per_video.
        :param grayscale_entropy: True to compute the entropy of a grayscale image instead of a color image. This
        option corrects the flawed entropy computation of the original paper where the correlation of color channels is
        not considered.
        """

        self.grid_cell_size = grid_cell_size
        self.grid_block_size = grid_block_size
        self.grid_padding = grid_padding
        # Number of grid cells without the padding border and with space to fit the entire block without interfering
        # with the padding.
        self.num_blocks = self.grid_cell_size - 2 * self.grid_padding - (self.grid_block_size - 1)

        self.output_size = output_size
        self.output_padding_factor = output_padding_factor

        self.num_frames_per_video = num_frames_per_video
        self.chunked_preprocessing = chunked_preprocessing

        self.grayscale_entropy = grayscale_entropy

        self.crop_padding_mode = crop_padding_mode

    @staticmethod
    def _crop_box(image: np.ndarray, box: BBox) -> np.ndarray:
        """
        Crops an image to the size of the bounding box. The coordinate system is the usual image coordinate system.

        :param image: input image in cv format [h,w,c]
        :param box: bounding box to crop
        :return: cropped image patch in cv format
        """
        h, w = image.shape[:2]
        y_min = int(max(box.y1, 0))
        y_max = int(min(box.y2, h))
        x_min = int(max(box.x1, 0))
        x_max = int(min(box.x2, w))

        crop = image[y_min:y_max, x_min:x_max, ...]

        if np.size(crop) == 0:
            raise ValueError('Invalid crop size. The cropped area is empty.')

        return crop

    def preprocess(self, dataset_path, output_path):
        """
        Preprocesses the ILSVRC VID dataset for training LUDT.

        .
        +---Annotations
        +---ImageSets
        +---Data
            +---VID
                +---snippets
                +---test
                |   +---ILSVRC2015_test_00000000
                +---train
                |   +---ILSVRC2015_VID_train_0000
                |       +---ILSVRC2015_train_00000000
                +---val
                    +---ILSVRC2015_val_00000000

        :param dataset_path: path to the dataset root.
        :param output_path: output directory for the preprocessed data
        """
        from os.path import join

        assert os.path.isdir(dataset_path), 'Dataset path does not point to a valid directory.'

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f'Created output directory: {output_path}.')

        data_path = join(dataset_path, 'Data', 'VID')

        # 'train' needs special treatment because it is split into folders of 1k videos each
        split_dirs = ['test', 'val'] + [join('train', s) for s in os.listdir(join(data_path, 'train'))]

        for split_num, split_dir in enumerate(split_dirs):
            split_path = join(data_path, split_dir)

            for video_dir in tqdm(iterable=os.listdir(split_path),
                                  desc=f'[{split_num + 1}/{len(split_dirs)}] {split_dir} Video'):
                video_input_path = join(split_path, video_dir)

                variant_name = split_dir.replace('/', '_').replace('\\', '_')
                video_output_dir = join(output_path, variant_name, video_dir)
                os.makedirs(video_output_dir, exist_ok=True)

                if self.chunked_preprocessing:
                    self._preprocess_video_chunking(video_input_path, video_output_dir)
                else:
                    self._preprocess_video_dropping(video_input_path, video_output_dir)

        print('Completed data extraction.')

        print('Creating manifest file.')
        self.create_manifest(output_path)
        print('Finished manifest generation.')

    def create_manifest(self, extracted_path: str) -> None:
        manifest = []
        variants = os.listdir(extracted_path)
        for variant in variants:
            variant_path = os.path.join(extracted_path, variant)
            if not os.path.isdir(variant_path):
                continue

            videos = os.listdir(variant_path)

            if self.chunked_preprocessing:
                videos = [os.path.join(v, s) for v in videos for s in os.listdir(os.path.join(variant_path, v))]

            for video in videos:
                video_path = os.path.join(variant_path, video)
                if not os.path.isdir(video_path):
                    continue

                frames = [f for f in os.listdir(video_path) if f.lower().endswith('.jpeg')]

                manifest.append({
                    'variant': variant,
                    'video': video,
                    'frames': frames,
                })

        manifest_file = os.path.join(extracted_path, 'manifest.json')
        with open(manifest_file, 'w', encoding='utf8') as out_file:
            json.dump(manifest, out_file)

    def _preprocess_video_dropping(self, video_path, video_output_dir):
        """
        Preprocessing of a video according to the method specified in the paper, i.e. take a fixed number of frames and
        drop the remaining ones.

        :param video_path: input frame path
        :param video_output_dir: output frame path
        """
        frame_names = sorted(os.listdir(video_path))
        frame_names = frame_names[:self.num_frames_per_video]

        in_files = [os.path.join(video_path, fn) for fn in frame_names]
        out_files = [os.path.join(video_output_dir, f'{fn:06d}.JPEG') for fn in range(len(frame_names))]

        self._preprocess_frames(in_files, out_files)

    def _preprocess_video_chunking(self, video_path, video_output_dir):
        """
        Preprocessing of a video where the video is split into chunks of size num_frames_per_video. The output is
        guaranteed to have the given size. Shorter chunks are dropped.

        :param video_path: input frame path
        :param video_output_dir: output frame path
        """
        assert self.chunked_preprocessing

        frame_names = sorted(os.listdir(video_path))
        for i, chunk in enumerate(chunked(frame_names, self.num_frames_per_video)):
            if len(chunk) < self.num_frames_per_video:
                continue

            seq_output_dir = os.path.join(video_output_dir, f'sequence_{i:08d}')
            os.makedirs(seq_output_dir, exist_ok=True)

            in_files = [os.path.join(video_path, fn) for fn in chunk]
            out_files = [os.path.join(seq_output_dir, f'{fn:06d}.JPEG') for fn in range(len(chunk))]

            # self.preprocess_frames_matlab(in_files, out_files)
            self._preprocess_frames(in_files, out_files)

    def _preprocess_frames_matlab(self, input_files: List[str], output_files: List[str]) -> None:
        """
        Preprocesses a single video. The video is given in the form of paths to image files.

        :param input_files: list of paths to video frames
        :param output_files: list of output paths for each extracted patch (must be as long as input_files)
        """
        assert len(input_files) > 0
        assert len(input_files) == len(output_files)

        frame = cv.imread(input_files[0], cv.IMREAD_COLOR)
        box = self._select_highest_entropy_box(frame)
        target_sz = np.array(box.size())
        pos = np.array(box.p1())
        boxes = matlab_kcf_tracker(input_files, pos, target_sz)

        for i, (in_file, out_file) in enumerate(zip(input_files, output_files)):
            # read the frame as color image
            frame = cv.imread(in_file, cv.IMREAD_COLOR)

            resized_crop = self._resample_bbox(frame, BBox.from_xywh(boxes[i, :]))

            cv.imwrite(out_file, resized_crop)

    def _preprocess_frames(self, input_files: List[str], output_files: List[str]) -> None:
        """
        Preprocesses a single video. The video is given in the form of paths to image files.

        :param input_files: list of paths to video frames
        :param output_files: list of output paths for each extracted patch (must be as long as input_files)
        """
        assert len(input_files) > 0
        assert len(input_files) == len(output_files)

        tracker = KCFTracker(backend='torch', features='hog', padding=1.8)
        for i, (in_file, out_file) in enumerate(zip(input_files, output_files)):

            # read the frame as color image
            frame = cv.imread(in_file, cv.IMREAD_COLOR)

            next_box: BBox
            if i == 0:
                # compute the best bounding box to start the track
                box = self._select_highest_entropy_box(frame)

                tracker.begin_track(frame, box)
                next_box = box
            else:
                next_box = tracker.track(frame)

            # resized_crop = self.crop_tracked_box(frame, next_box)
            resized_crop = self._resample_bbox(frame, next_box)

            cv.imwrite(out_file, resized_crop)

    def _create_boxes(self, w: int, h: int) -> List[BBox]:
        """
        Creates the proposal bounding boxes for entropy evaluation.

        The box grid is created according to the constructor parameters. The grid_size specifies into how may parts
        cells the image should be split. Then the given number of padding cells are skipped in all directions. In the
        grid of remaining cells all consecutive blocks of block_size x block_size are returned.

        :param w: width of the image
        :param h: height of the image
        :return: bounding boxes
        """
        w_step = w / self.grid_cell_size
        h_step = h / self.grid_cell_size

        boxes = []
        for j in range(self.num_blocks):
            for i in range(self.num_blocks):
                x1 = self.grid_padding + i
                y1 = self.grid_padding + j
                boxes.append(BBox.from_xyxy([
                    x1 * w_step,
                    y1 * h_step,
                    (x1 + self.grid_block_size) * w_step,
                    (y1 + self.grid_block_size) * h_step
                ]))

        return boxes

    def _select_highest_entropy_box(self, image: np.ndarray) -> BBox:
        """
        Computes the bounding box with highest entropy in the selection grid.

        :param image: Input image in cv format [H,W,C]
        :return: bounding box
        """

        h, w, _ = image.shape
        boxes = self._create_boxes(w, h)

        # vis = ff.copy()
        # draw_boxes(vis, boxes)
        # cv.imshow('Boxes', vis)
        # cv.waitKey()

        # Usually the entropy is computed in a gray-scale image, but the implementation in the paper uses the linearized
        # color image patch. This seems wrong, because the color channels are correlated in some way, but that's the
        # way it is now... The entropy values computed in this way also agree with the ones shown in Fig. 7 of the
        # paper.
        if self.grayscale_entropy:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        entropy = np.zeros(len(boxes))
        for i, box in enumerate(boxes):
            crop = self._crop_box(image, box)

            # compute a histogram of pixel values.
            _, counts = np.unique(crop, return_counts=True)
            entropy[i] = scipy.stats.entropy(counts, base=2)

        best = np.argmax(entropy).item()

        return boxes[best]

    def _crop_tracked_box(self, frame: np.ndarray, bbox: BBox):
        """
        Pads and then crops and resizes the given bounding box from the frame. The padded box is clipped to the image
        size and then resized. This can introduce big distortions and changes in aspect ratio. Furthermore, when the box
        is outside of the image boundaries, this method cannot produce a result.

        :param frame: input frame in cv format [H,W,C]
        :param bbox:  bounding box to crop
        :return: cropped and resized bbox content.
        """

        cx, cy, w, h = bbox.as_cwh()

        # apply padding similar to the paper
        padded_w = (w * (1 + self.output_padding_factor))
        padded_h = (h * (1 + self.output_padding_factor))

        crop = self._crop_box(frame, BBox.from_cwh([cx, cy, padded_w, padded_h]))
        resized_crop = cv.resize(crop, self.output_size, interpolation=cv.INTER_AREA)
        return resized_crop

    def _resample_bbox(self, frame: np.ndarray, bbox: BBox):
        """
        Pads and then extracts and rescales the content of the given bounding box. The output box has the size specified
        in self.output_size. The padding size is controlled by self.output_padding_factor. This function works, even if
        the bbox is out of frame. The resampling uses bilinear interpolation. Values outside of the image are clipped to
        the boundary values.

        :param frame: input frame in cv format [H,W,C]
        :param bbox:  bounding box to sample
        :return: cropped and resized box content
        """

        image = cv_to_torch(frame.astype(np.float32), add_batch=True).cuda()

        cx, cy, w, h = bbox.as_cwh()

        # apply padding similar to the paper
        padded_w = w * (1 + self.output_padding_factor)
        padded_h = h * (1 + self.output_padding_factor)

        sampled_image = crop_sampling(image, self.output_size,
                                      box=BBox.from_cwh((cx, cy, padded_w, padded_h)),
                                      padding_mode=self.crop_padding_mode)

        return torch_to_cv(sampled_image).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LUDT Dataset Preprocessor and Extractor',
        description='Preprocessing of video data for LUDT (Wang. et al., 2020). The preprocessor consumes the '
                    'ILSVRC2015 VID dataset and extracts samples for unsupervised training of LUDT. Each video '
                    'is split into chunks, then the first chunk is selected. Based on the image entropy a patch '
                    'is selected in the first frame and then tracked with a KCF tracker through the remaining frames. '
                    'Finally, the the track boxes are used to extract and resize image patches for training.')

    parser.add_argument('--input', required=False, type=str, default=config.dataset_imagenet_path,
                        help='Input directory where ILSVRC2015 is stored.')
    parser.add_argument('--output', required=False, type=str, default=config.dataset_ludt_imagenet_path,
                        help='Output directory where the preprocessed dataset should be stored.')
    parser.add_argument('--grayscale-entropy', required=False, default=False, action='store_true',
                        help='Deviate from the paper and compute the entropy of the grayscale image to avoid color '
                             'channel correlation influence.')
    parser.add_argument('--chunked-processing', required=False, default=False, action='store_true',
                        help='Deviate from the paper and process all video chunks, not only the first one per video.')
    parser.add_argument('--chunk-size', required=False, default=100, type=int,
                        help='Number of frames in each video chunk.')
    parser.add_argument('--manifest-only', action='store_true', default=False,
                        help='Only compute the manifest for already completed data, no actual preprocessing.')
    parser.add_argument('--crop-padding', choices=['zeros', 'border', 'reflection'], default='zeros',
                        help='Padding mode for bounding boxes that aren\'t within the boundaries of the image.')

    args = parser.parse_args()

    extractor = LUDTExtractor(
        num_frames_per_video=args.chunk_size,
        chunked_preprocessing=args.chunked_processing,
        grayscale_entropy=args.grayscale_entropy,
    )
    if args.manifest_only:
        extractor.create_manifest(args.output)
    else:
        extractor.preprocess(args.input, args.output)
