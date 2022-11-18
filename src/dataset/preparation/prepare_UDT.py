import argparse
import glob
import json
import os
from os.path import join

import cv2 as cv
import numpy as np

import config


class UDTExtractor:
    def __init__(self, output_size: int, padding: float, num_val):
        self.output_size = output_size
        self.output_padding = padding
        self.num_val = num_val

    def preprocess(self, dataset_path, output_path):
        assert os.path.isdir(dataset_path), 'Dataset path does not point to a valid directory.'

        if os.path.exists(output_path):
            if os.listdir(output_path):
                raise FileExistsError('Output path exists and is not empty.')

        patch_output_path = join(output_path, f'crop_{self.output_size :d}_{self.output_padding :1.1f}')
        os.makedirs(patch_output_path)

        img_base_path = join(dataset_path, 'Data', 'VID', 'train')
        sub_sets = sorted({'ILSVRC2015_VID_train_0000',
                           'ILSVRC2015_VID_train_0001',
                           'ILSVRC2015_VID_train_0002',
                           'ILSVRC2015_VID_train_0003'})

        num_all_frame = 1298523
        manifest = {
            'down_index': np.zeros(num_all_frame, np.int),
            'up_index': np.zeros(num_all_frame, np.int),
        }

        out_idx = 0
        for subset_name in sub_sets:
            subset_dir = join(img_base_path, subset_name)
            video_names = sorted(os.listdir(subset_dir))

            for vn, video_name in enumerate(video_names):
                video_dir = join(subset_dir, video_name)
                frame_names = sorted(glob.glob(join(video_dir, '*.JPEG')))
                vid_len = len(frame_names)

                for fn, frame_name in enumerate(frame_names):
                    frame_path = join(video_dir, frame_name)
                    original_image = cv.imread(frame_path)

                    patch = self._crop_patch(original_image)

                    frame_out_path = join(patch_output_path, f'{out_idx:08d}.jpg')
                    cv.imwrite(frame_out_path, patch)

                    # how many frames to the first frame
                    manifest['down_index'][out_idx] = fn
                    # how many frames to the last frame
                    manifest['up_index'][out_idx] = vid_len - fn
                    out_idx += 1

                    print(f'[{out_idx}/{num_all_frame}]')

        # NEVER use the last frame as template! I do not like bidirectional
        template_id = np.where(manifest['up_index'] > 1)[0]
        rand_split = np.random.choice(len(template_id), len(template_id))

        split_idx = len(template_id) - self.num_val
        manifest['train_set'] = template_id[rand_split[:split_idx]]
        manifest['val_set'] = template_id[rand_split[split_idx:]]
        # to list for json
        manifest['train_set'] = manifest['train_set'].tolist()
        manifest['val_set'] = manifest['val_set'].tolist()
        manifest['down_index'] = manifest['down_index'].tolist()
        manifest['up_index'] = manifest['up_index'].tolist()

        manifest_file = join(output_path, 'dataset.json')
        with open(manifest_file, 'w') as out_file:
            json.dump(manifest, out_file, indent=2)

    def _crop_patch(self, image: np.ndarray):
        w = float((image.shape[1] / 6) * (1 + self.output_padding))
        h = float((image.shape[0] / 6) * (1 + self.output_padding))
        x = float((image.shape[1] / 2) - w / 2)
        y = float((image.shape[0] / 2) - h / 2)

        a = (self.output_size - 1) / w
        b = (self.output_size - 1) / h
        c = -a * x
        d = -b * y

        # The power of affine warping is not necessary, because only scaling is performed. Therefore, this implementation
        # is probably incredibly slow compared to a simple cv::resize.
        # FIXME: implement using resize
        mapping = np.array([[a, 0, c],
                            [0, b, d]], dtype=np.float)

        return cv.warpAffine(image, mapping, (self.output_size, self.output_size),
                             borderMode=cv.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UDT Dataset Preprocessor and Extractor')

    parser.add_argument('--input', required=False, type=str, default=config.dataset_imagenet_path,
                        help='Input directory where ILSVRC2015 is stored.')
    parser.add_argument('--output', required=False, type=str, default=config.dataset_udt_imagenet_path,
                        help='Output directory where the preprocessed dataset should be stored.')
    parser.add_argument('--output-size', default=125, type=int, required=False,
                        help='Crop output size.')
    parser.add_argument('--padding', default=2, type=float, required=False,
                        help='Crop padding size.')
    parser.add_argument('--num-val', default=3000, type=int, required=False,
                        help='Size of the validation dataset split.')

    args = parser.parse_args()

    extractor = UDTExtractor(args.output_size, args.padding, args.num_val)
    extractor.preprocess(args.input, args.output)
