import argparse
import hashlib
import re
import shutil
from multiprocessing.pool import ThreadPool

import numpy as np
import os

import config
import json

from dataset.preparation.preparation_utils import download_url, extract_zip

META_DIR = os.path.join(os.path.dirname(__file__), 'otb')
ANNOTATIONS_FILE = os.path.join(META_DIR, 'OTB_annotations.json')
URLS_FILE = os.path.join(META_DIR, 'OTB2015_urls.txt')

otb2013_videos = ["Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace", "BlurOwl", "Bolt", "Box", "Car1",
                  "Car4", "CarDark", "CarScale", "ClifBar", "Couple", "Crowds", "David", "Deer", "Diving", "DragonBaby",
                  "Dudek", "Football", "Freeman4", "Girl", "Human3", "Human4", "Human6", "Human9", "Ironman", "Jump",
                  "Jumping", "Liquor", "Matrix", "MotorRolling", "Panda", "RedTeam", "Shaking", "Singer2", "Skating1",
                  "Skating2_1", "Skating2_2", "Skiing", "Soccer", "Surfer", "Sylvester", "Tiger2", "Trellis", "Walking",
                  "Walking2", "Woman"]


def download_and_expand_dataset(out_dir, num_workers, keep_zips):
    with open(URLS_FILE, 'r', encoding='utf8') as f:
        urls = f.read().splitlines()

    def process_url(pair):
        expected_md5, url = pair.split(' ')

        zip_file_name = os.path.basename(url)
        zip_file_path = os.path.join(out_dir, zip_file_name)

        video_name = os.path.splitext(zip_file_name)[0]
        if os.path.exists(os.path.join(out_dir, video_name)):
            print(f'Skipping {video_name}')
            return

        description = f'{zip_file_name:30s}'
        download_url(url, zip_file_path, description)

        with open(zip_file_path, 'rb') as f:
            actual_md5 = hashlib.md5(f.read()).hexdigest()

            if actual_md5.lower() != expected_md5.lower():
                err_msg = f'Hash mismatch for video {zip_file_name}'
                raise RuntimeError(err_msg)

        extract_zip(zip_file_path, out_dir)

        if keep_zips:
            zip_dir = os.path.join(out_dir, '__zips')
            if not os.path.isdir(zip_dir):
                os.mkdir(zip_dir)

            shutil.move(zip_file_path, zip_dir)
        else:
            os.remove(zip_file_path)

    with ThreadPool(processes=num_workers) as pool:
        print(pool.map(process_url, urls))


def create_ground_truth(out_dir):
    with open(ANNOTATIONS_FILE, 'r', encoding='utf8') as f:
        otb2015 = json.load(f)

        for video_key, video_manifest in otb2015.items():
            old_bbox_path: str = video_manifest['gt']

            with open(os.path.join(out_dir, old_bbox_path), 'r', encoding='utf8') as f:
                gt = np.array([list(map(float, re.split('[, \t]', line))) for line in f], dtype=np.float32)

            new_bbox_path = old_bbox_path[:-4] + '.npy'
            out_file = os.path.join(out_dir, new_bbox_path)
            np.save(out_file, gt)

            video_manifest['gt'] = new_bbox_path

    otb2013_path = os.path.join(out_dir, 'OTB2013_manifest.json')
    otb2015_path = os.path.join(out_dir, 'OTB2015_manifest.json')

    otb2013 = {k: otb2015[k] for k in otb2013_videos}

    with open(otb2015_path, 'w', encoding='utf8') as f:
        json.dump(otb2015, f)
    with open(otb2013_path, 'w', encoding='utf8') as f:
        json.dump(otb2013, f)


def prepare_otb(out_dir, incremental, num_workers, keep_zips, no_download):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        if not os.path.isdir(out_dir):
            raise ValueError('Specified out_dir is not a directory.')

        if not incremental and not no_download and os.listdir(out_dir):
            raise ValueError('Specified out_dir is not empty.')

    if not no_download:
        download_and_expand_dataset(out_dir, num_workers, keep_zips)

    create_ground_truth(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OTB(2013,2015) Download and Preprocessing')
    parser.add_argument('--output', '-o', type=str, default=config.dataset_otb_path, metavar='OUTPUT_DIR',
                        help='Output directory for the dataset. Must be empty if --incremental is not set.')
    parser.add_argument('--incremental', action='store_true', default=False,
                        help='Allows to resume a partial download of the dataset. Assumes that existing data is valid.')
    parser.add_argument('--download-workers', '-j', type=int, default=8, metavar='NUM_PARALLEL_DOWNLOADS',
                        help='Number of simultaneous video downloads.')
    parser.add_argument('--keep-zips', action='store_false', default=True,
                        help='Move the downloaded zip files to <out>/__zips instead of deleting them after extraction.')
    parser.add_argument('--no-download', action='store_true', default=False,
                        help='Only create the new manifest and ground truth annotations for downloaded dataset.')

    args = parser.parse_args()

    prepare_otb(
        out_dir=args.output,
        incremental=args.incremental,
        num_workers=args.download_workers,
        keep_zips=args.keep_zips,
        no_download=args.no_download,
    )
