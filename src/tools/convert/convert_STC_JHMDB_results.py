import argparse
import os
from os.path import join
from pathlib import Path

import numpy as np


def main(args):
    in_dir = args.input_dir
    out_dir = args.output_dir
    list_path = join(in_dir, 'testlist_split1.txt') if args.list_file is None else args.list_file

    # Find out the video processing ordering to associate numbers with video names.
    with open(list_path, 'r', encoding='utf8') as f:
        names = []
        for line in f:
            a, b = map(str.strip, line.split())
            names.append(Path(a if b.endswith('.mat') else b).name)

    for i, name in enumerate(names):
        kpts = np.load(join(in_dir, f'{i}.dat'), allow_pickle=True)

        vid_dir = join(out_dir, name)
        os.makedirs(vid_dir, exist_ok=True)

        # These sequences are broken and contain invalid images at the end
        if name == 'The_Slow_Clap_clap_u_cm_np1_fr_med_4':
            kpts = kpts[:, :, :27]
        elif name == 'The_Slow_Clap_clap_u_nm_np1_fr_bad_20':
            kpts = kpts[:, :, :18]

        kpts[0, :, :] = kpts[0, :, :] / 40.0 * 320
        # kpts[1, :, :] = kpts[1, :, :] / 30.0 * 240
        kpts[1, :, :] = kpts[1, :, :] / 40.0 * 240

        np.save(join(vid_dir, 'keypoints.npy'), kpts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Helper to convert STC results on JHMDB to a usable format',
        description='This script converts the outputs of the STC reference implementation to a usable format that can '
                    'be evaluated with out evaluation toolkit. This involves changing the on-disk structure of '
                    'results, using directories for each video instead of integer numbers as names. Further, STC saves '
                    'its keypoints in feature coordinates, thus it is necessary to transform them to image coordinates.'
    )
    parser.add_argument('--input-dir',
                        type=str,
                        required=True,
                        help='Input directory (STC results on JHMDB).')
    parser.add_argument('--ouput-dir',
                        type=str,
                        required=True,
                        help='Output directory for the converted results.')
    parser.add_argument('--list-file',
                        type=str,
                        required=False,
                        default=None,
                        help='List of input images and ground truth. Default is a file called testlist_split1.txt'
                             'located in the input directory. Format: '
                             '<prefix>/joint_positions/<action>/<video>/joint_positions.mat '
                             '<prefix>/Rename_Images/<action>/<video>')
    main(parser.parse_args())
