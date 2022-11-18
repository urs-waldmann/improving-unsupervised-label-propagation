"""
Incomplete implementation! Go to the download url manually and extract the downloaded zip file.
"""

import argparse
import os
from pathlib import Path

import config
from dataset.preparation.preparation_utils import download_url, extract_zip

DAVIS_URLS = {
    'DAVIS2016': {
        'TrainVal': 'https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip'
    },
    'DAVIS2017': {
        'TrainVal': {
            "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
            "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip",
        },
        'Test-Dev': {
            # Test-Dev 2017 Images and First-Frame Annotations
            "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip",
            "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-Full-Resolution.zip",
        },
        'Test-Challenge': {
            # Test-Challenge 2017 Images and First-Frame Annotations
            "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip",
            "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-Full-Resolution.zip",
        }
    },
    'DAVIS2019': {
        'TrainVal': {
            # TrainVal Images and Annotations
            "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip",
            "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip",
        },
        'Test-Dev': {
            # Test-Dev 2019 Images
            "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-dev-480p.zip",
            "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-dev-Full-Resolution.zip",
        },
        'Test-Challenge': {
            # Test-Challenge 2019 Images
            "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-challenge-480p.zip",
            "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2019-Unsupervised-test-challenge-Full-Resolution.zip",
        }
    },
    'Object Categories': {
        # TrainVal, Test-Dev, Test-Challenge
        "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017_semantics-480p.zip",
        "Full": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017_semantics-Full-resolution.zip",
    },
    'Scribbles': {
        # TrainVal Annotated scribbles
        "All resolutions": "https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip",
    },
}


class DavisPreprocessor:
    def __init__(self, args):
        self.args = args

        self.out_dir = args.output
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        if not os.path.isdir(self.out_dir):
            raise ValueError('Provided output location is not a directory.')

    def prepare_dataset(self):

        url = DAVIS_URLS['DAVIS2016']['TrainVal']
        url_path = Path(url)

        zip_dir = os.path.join(self.out_dir, '__zip')
        if not os.path.exists(zip_dir):
            os.mkdir(zip_dir)

        zip_file = os.path.join(zip_dir, url_path.name)
        out_path = os.path.join(self.out_dir, url_path.stem)

        download_url(url, zip_file, "DAVIS2016 TrainVal")
        extract_zip(zip_file, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVIS dataset download and preparation')
    parser.add_argument('--variant', type=str, choices=['DAVIS2016'], default='DAVIS2016',
                        help='Dataset variant')
    parser.add_argument('--output', type=str, default=config.dataset_davis2016_root)

    args = parser.parse_args()

    preprocessor = DavisPreprocessor(args)
    preprocessor.prepare_dataset()

    pass
