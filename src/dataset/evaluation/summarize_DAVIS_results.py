import argparse
import os
import warnings

import pandas as pd

from dataset.evaluation.evaluate_segmentation_dataset import prettyprint_results, read_yml_cache
from utils.path_utils import list_subdirs


def evaluate_davis(args):
    if args.eval_all:
        eval_dirs = list_subdirs(args.input_dir, relative=False)
    else:
        eval_dirs = [args.input_dir]

    data = []
    for result_dir in eval_dirs:
        found = False
        for spec in {'so', 'mo'}:
            method_name = spec + '__' + os.path.basename(result_dir)
            result_yml = os.path.join(result_dir, f'results-{spec}.yml')
            if os.path.isfile(result_yml):
                data.append(read_yml_cache(method_name, result_yml))
                found = True

        if not found:
            print(f'Could not find results file for {result_dir}')

    df = pd.DataFrame(data).sort_index()

    if args.sort_key is not None:
        if args.sort_key not in df.columns:
            warnings.warn(f'Could not locate sort key in columns. '
                          f'Key={args.sort_key}, Columns={set(df.columns)}. Skipping sort.')
        else:
            df = df.sort_values(args.sort_key)

    prettyprint_results(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory')
    parser.add_argument('--eval-all', action='store_true', default=False,
                        help='Treats the input directory as a list of evaluation runs and evaluates them all.')
    parser.add_argument('--sort-key', type=str, default=None,
                        help='Key to sort the values by.')
    evaluate_davis(parser.parse_args())
