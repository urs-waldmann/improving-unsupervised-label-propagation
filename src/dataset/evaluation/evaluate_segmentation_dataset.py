import argparse
import os
import warnings
from os.path import isfile

import pandas as pd
import yaml
from prettytable import PrettyTable

from dataset.SegTrackV2 import SegTrackV2
from dataset.davis import MultiObjectDavisDataset
from dataset.evaluation.segmentation_eval_utils import VideoSegmentationEvaluator
from dataset.mask_dataset import SegmentationResultReader, MaskDataset
from utils.path_utils import list_subdirs


def prettyprint_results(df: pd.DataFrame) -> None:
    """
    Prints a result dataframe as pretty table. The index is shown as the 'Method'. The values are converted to
    percentages (i.e. multiplied by 100) and formatted to two digits precision.
    :param df: input data frame
    :return: None
    """
    result_table = PrettyTable(['Method'] + list(df.columns))
    result_table.align['Method'] = 'l'

    for method_name in df.index:
        result_table.add_row([method_name] + ['{:.02f}'.format(v * 100) for v in df.loc[method_name]])

    print(result_table.get_string())


def read_yml_cache(config_name: str, result_yml: str) -> pd.Series:
    """
    Loads the dataset result summary from a yml file created with the DAVIS evaluation toolbox. Additionally, J&F is
    computed.
    :param config_name: Name of the configuration the results belong to (name of the results series object).
    :param result_yml: Path to the result.yml file.
    :return: Series with the result statistics for the dataset.
    """
    with open(result_yml, 'r', encoding='utf8') as f:
        results = yaml.safe_load(f)
        j_stats = results['dataset']['J']
        f_stats = results['dataset']['F']
        scores = {
            'J_and_F': (j_stats['mean'] + f_stats['mean']) / 2,
            'J_mean': j_stats['mean'],
            'J_recall': j_stats['recall'],
            'J_decay': j_stats['decay'],
            'F_mean': f_stats['mean'],
            'F_recall': f_stats['recall'],
            'F_decay': f_stats['decay'],
        }
    return pd.Series(data=scores, name=config_name)


def read_or_compute(dataset: MaskDataset, result_dir: str, evaluator: VideoSegmentationEvaluator) -> pd.Series:
    """
    Computes the scores for the given result_dir and dataset. If the results are cached already, the cached value is
    loaded instead of recomputing the entire score.

    :param dataset: Dataset object belonging to the results in result_dir
    :param result_dir: directory with the sequence results for the given dataset
    :param evaluator: Evaluator to use for result computation
    :return: Series with result scores for the given result dir
    """
    config_name = os.path.basename(result_dir)

    spec = 'mo' if evaluator.multi_object else 'so'
    long_spec = 'multi-object' if evaluator.multi_object else 'single-object'

    # Current format: Raw scores for each frame, object and video listed in table
    cache_csv_new = os.path.join(result_dir, f'{spec}_results.csv')
    # Old csv format: Results for each video
    cache_csv = os.path.join(result_dir, f'{long_spec}_results.csv')
    # DAVIS tool format: yml with dataset and sequence results
    cache_yml = os.path.join(result_dir, f'results-{spec}.yml')

    if isfile(cache_csv_new):
        seq = evaluator.compute_dataset_mean(evaluator.compute_statistics(pd.read_csv(cache_csv_new, index_col=0)))
    elif isfile(cache_csv):
        seq = pd.read_csv(cache_csv, index_col=0).mean()
    elif isfile(cache_yml):
        seq = read_yml_cache(config_name, cache_yml)
    else:
        result_reader = SegmentationResultReader(result_dir)
        result = evaluator.evaluate(dataset, result_reader)
        seq = result.mean

        # Save in cache for later
        result.raw_scores.to_csv(cache_csv_new)

    seq.name = config_name
    return seq


def evaluate_dataset(args):
    multi_object = args.mode == 'multi-object'

    if args.eval_all:
        eval_dirs = list_subdirs(args.input_dir, relative=False)
        if not args.include_hidden:
            eval_dirs = list(filter(lambda x: not os.path.basename(x).startswith('.'), eval_dirs))
    else:
        if args.include_hidden:
            warnings.warn(f'--include-hidden has no effect when --eval-all is not used.')
        eval_dirs = [args.input_dir]

    # TODO: Load dataset from dataset config json
    if args.dataset == 'SegTrackV2':
        dataset = SegTrackV2(dataset_root=args.dataset_root)
    elif args.dataset == 'DAVIS2017val':
        dataset = MultiObjectDavisDataset(dataset_root=args.dataset_root, year='2017', split='val')
    elif args.dataset == 'DAVIS2016val':
        dataset = MultiObjectDavisDataset(dataset_root=args.dataset_root, year='2016', split='val')
    else:
        raise ValueError('Invalid dataset name.')

    # Compute statistics or read cached data to build result dataframe
    evaluator = VideoSegmentationEvaluator(multi_object=multi_object, verbose=True)
    result_data = [read_or_compute(dataset, result_dir, evaluator) for result_dir in eval_dirs]
    df = pd.DataFrame(result_data)

    # Sort by sort-key or index if no key is given
    df = df.sort_index()
    if args.sort_key is not None:
        if args.sort_key in df.columns:
            df = df.sort_values(args.sort_key)
        else:
            warnings.warn(f'Could not locate sort key in columns. '
                          f'Key={args.sort_key}, Columns={set(df.columns)}. Skipping sort.')

    prettyprint_results(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices={'DAVIS2016val', 'DAVIS2017val', 'SegTrackV2'},
                        help='Dataset to evaluate')
    parser.add_argument('--dataset-root',
                        type=str,
                        default=None,
                        required=False,
                        help='Path where the dataset is located.')
    parser.add_argument('--input-dir',
                        type=str,
                        required=True,
                        help='Result directory or directory of configuration results (--eval-all).')
    parser.add_argument('--eval-all',
                        action='store_true',
                        default=False,
                        help='Treats the input directory as a list of evaluation runs and evaluates them all.')
    parser.add_argument('--include-hidden',
                        action='store_true',
                        default=False,
                        help='Include hidden directories (starting with ".") in the list of configuration directories.')
    parser.add_argument('--mode',
                        choices={'single-object', 'multi-object'},
                        required=True,
                        help="Chose between single and multi-object modes.")
    parser.add_argument('--sort-key',
                        type=str,
                        default=None,
                        help='Key to sort the values by.')

    evaluate_dataset(parser.parse_args())
