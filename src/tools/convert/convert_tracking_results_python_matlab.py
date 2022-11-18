"""
Helper script to convert tracking results from matlab to python format.
"""
import argparse
import os

import numpy as np
from tqdm import tqdm


def convert_sequence(out_file, seq_file):
    try:
        temp = np.loadtxt(seq_file, delimiter=',').astype(np.float)
    except ValueError:
        temp = np.loadtxt(seq_file, delimiter=' ').astype(np.float)
    boxes = np.array(temp)
    if to_matlab:
        boxes[:, 0:2] += 1
    else:
        boxes[:, 0:2] -= 1

    np.savetxt(out_file, boxes, fmt='%.02f', delimiter=',')


def convert_all(root_dir):
    trackers = os.listdir(root_dir)

    for tracker in trackers:
        tracker_dir = os.path.join(root_dir, tracker)
        sequences = os.listdir(tracker_dir)

        out_dir = os.path.join(root_dir, tracker + ('_mat' if to_matlab else '_py'))
        os.mkdir(out_dir)

        for sequence in tqdm(sequences, desc=tracker, total=len(sequences)):
            seq_dir = os.path.join(tracker_dir, sequence)

            out_file = os.path.join(out_dir, sequence)

            convert_sequence(out_file, seq_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        type=str,
                        required=True,
                        help='Input directory where the result sequences are located.')
    parser.add_argument('--from-format',
                        type=str,
                        choices={'matlab', 'python'},
                        required=True,
                        help='Input format of the data.')

    args = parser.parse_args()

    to_matlab = args.from_format == 'python'
    convert_all(args.input_dir)
