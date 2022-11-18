import argparse
import os.path

from torchvision.datasets.utils import download_url, check_integrity, extract_archive

import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Downloads the ILSVRC2015_VID dataset.')
    parser.add_argument('--dataset-root', required=False, default=config.dataset_root,
                        help='Dataset root directory.')
    parser.add_argument('--delete-archive', action='store_true', default=False,
                        help='Delete the downloaded archive after extraction.')
    args = parser.parse_args()

    dataset_url = 'http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz'
    dataset_md5 = '5feb88ac3345e5b5eae71f6ec8a91325'
    archive_name = 'ILSVRC2015_VID.tar.gz'

    archive_path = os.path.join(args.dataset_root, archive_name)
    if not os.path.isfile(archive_path):
        download_url(dataset_url, args.dataset_root, archive_name, dataset_md5)
    elif not check_integrity(archive_path, dataset_md5):
        raise RuntimeError('Dataset archive already exists but is corrupted.')
    else:
        print('Using downloaded dataset archive.')

    print(f"Extracting {archive_path} to {args.dataset_root}")
    extract_archive(archive_path, args.dataset_root, args.delete_archive)
