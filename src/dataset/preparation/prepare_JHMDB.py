import argparse
import os

import config
from dataset.preparation.preparation_utils import download_url, check_md5, extract_archive

urls = [
    ('0ACA5AF988892C8C2AFE652A30544B8C', 'https://files.is.tue.mpg.de/jhmdb/JHMDB_video.zip'),
    ('23726FBAC4BC69A726DC3DAF2E7AB4C1', 'https://files.is.tue.mpg.de/jhmdb/Rename_Images.tar.gz'),
    ('3D1AC91BCB8544DDFDD9547997DB239F', 'https://files.is.tue.mpg.de/jhmdb/joint_positions.zip'),
    ('7719F617CB0AC06F54A5CECF7EE1D435', 'https://files.is.tue.mpg.de/jhmdb/README_joint_positions.txt'),
    ('9CF4B7FCB2AB8518B00CF42A7014E0F3', 'https://files.is.tue.mpg.de/jhmdb/estimated_joint_positions.zip'),
    ('EEBF6C6C71FAA16709B709B758710D17', 'https://files.is.tue.mpg.de/jhmdb/puppet_mask.zip'),
    ('E79D1A979156B59DE30912D3AB13F712', 'https://files.is.tue.mpg.de/jhmdb/README_puppet_mask.txt'),
    ('727DF5CE6AE0FAAE1859628C3703F6D1', 'https://files.is.tue.mpg.de/jhmdb/puppet_flow_ann.zip'),
    ('0EED5CA95D9376B5215BFEC1686024F5', 'https://files.is.tue.mpg.de/jhmdb/puppet_flow_com.zip'),
    ('9B0E1DF51A0EC13AE61155166A011FF0', 'https://files.is.tue.mpg.de/jhmdb/README_puppet_flow.txt'),
    ('15B9F2F7A99D63C07F3BE0C9ECDE43B5', 'https://files.is.tue.mpg.de/jhmdb/splits.zip'),
    ('359C924C762D4457239CCFD74E3F5E62', 'https://files.is.tue.mpg.de/jhmdb/README_splits.txt'),
    ('E459DF140CDE2125C61970151A0EA8CB', 'https://files.is.tue.mpg.de/jhmdb/sub_splits.zip'),
    ('E81F6BF07D6B6850865AD1D0B32C4944', 'https://files.is.tue.mpg.de/jhmdb/demo_compute_pose.zip'),
    ('1776792C9C9D2DB31C494C277BA08650', 'https://files.is.tue.mpg.de/hjhuang/demo_run_pose_estimation.zip'),
]


def download_files(jhmdb_root, allowed_files=None):
    files = []
    for md5, url in urls:
        if (allowed_files is not None) and os.path.basename(url) not in allowed_files:
            continue
        filename = os.path.split(url)[-1]
        out_file = os.path.join(jhmdb_root, filename)
        if os.path.exists(out_file):
            print(f'Skipping {out_file} ({url}).')
        else:
            download_url(url, out_file, f"Downloading {filename}.", chunk_size=4096)

        check_md5(out_file, md5)

        files.append(out_file)

    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=False, default=config.dataset_jhmdb_path,
                        help="Output root where the dataset will be created.")
    parser.add_argument('--allow-overwrite', action='store_true', default=False,
                        help="Allows overwriting existing data.")
    parser.add_argument('--download-all', action='store_true', default=False,
                        help="Downloads not only the necessary files, but all files that belong to the dataset.")
    parser.add_argument('--extract', action='store_true', default=False,
                        help="Extracts the archives.")

    args = parser.parse_args()

    ds_root = args.output
    try:
        os.makedirs(ds_root)
    except OSError:
        if not args.allow_overwrite:
            print(f'Directory already exists: {ds_root}')
            raise

    allowed_files = None if args.download_all else {
        'joint_positions.zip', 'puppet_mask.zip', 'Rename_Images.tar.gz', 'splits.zip', 'sub_splits.zip'}
    files = download_files(ds_root, allowed_files)

    if args.extract:
        for file in files:
            extract_archive(file, ds_root)
