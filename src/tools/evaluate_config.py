"""
This script is used to evaluate one or multiple json configurations. It is the primary way to interact with the code
in this repository.
"""

import argparse
import json
import os
import time

import pandas as pd
import torch

import config as global_config
import utils.util
from tools.evaluation_config import instantiate_config, read_config, validate_config
from label_prop import CachedLabelPropagationEvaluator


@torch.no_grad()
def evaluate_configuration(config, device, dataset_root, output_dir, *, is_remote=True, cache_dir=None):
    evaluator, dataset = instantiate_config(config, device, dataset_root)
    evaluator.verbose = not is_remote

    if cache_dir is not None:
        print('Using feature cache. Make sure, that the feature extractor and cache dir match and that the '
              'feature cache is up to date. No automatic validation is performed. It is your sole responsibility '
              'to ensure the correctness of the combination.')
        evaluator = CachedLabelPropagationEvaluator(cache_dir, evaluator)

    times = []
    for vid_num, (video_name, video_io_factory) in enumerate(dataset.iter_videos()):
        video_name = video_name.strip()
        print(f"    [{vid_num}/{len(dataset)}] Video: {video_name}.", flush=True)

        vid_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(vid_out_dir, exist_ok=True)

        with video_io_factory(vid_out_dir) as video_io:
            start = time.time()
            evaluator.eval_video(video_io)
            duration = time.time() - start

            num_frames = len(video_io)
            times.append((video_name, num_frames / duration, duration / num_frames))

    # Save times per video and average to file
    df = pd.DataFrame(times, columns=['video', 'fps', 'time_per_frame']).set_index('video')
    print(f'Average fps: {df["fps"].mean():.02f}', flush=True)
    df.to_csv(os.path.join(output_dir, 'times.csv'))


def evaluate_single_config(args, output_dir_for_conf, config, device):
    os.makedirs(output_dir_for_conf, exist_ok=True)

    # Save arguments, such that it is easy to identify where the results came from
    arg_file = os.path.join(output_dir_for_conf, 'arguments.txt')
    utils.util.save_args(args, arg_file, config=config)

    evaluate_configuration(
        config,
        device,
        args.dataset_path,
        output_dir_for_conf,
        is_remote=args.use_remote,
        cache_dir=args.feature_cache
    )


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
    args.device = device

    if args.config.startswith('{'):
        # Interpret config as config string.
        config = json.loads(args.config)
        validate_config(config)

        evaluate_single_config(args, args.output_dir, config, device)
    else:
        # Config string is a path
        args.config = global_config.resolve_args_path(args.config)

        if os.path.isfile(args.config):
            # There is only a single config, create single-item list with paths
            paths = [(args.config, args.output_dir)]
        else:
            # Path is a directory, read all configs and create mapping to output directories
            paths = sorted([
                (os.path.join(args.config, fn), os.path.join(args.output_dir, os.path.splitext(fn)[0]))
                for fn in os.listdir(args.config) if fn.endswith('.json')])

            print(f'Found {len(paths)} configurations.')

            # Filter output directories if desired
            if args.only_new_configs:
                paths = [t for t in paths if not os.path.isdir(t[1])]

            print(f'Evaluating {len(paths)} configurations.')

        # Perform the actual evaluation
        for config_path, output_dir in paths:
            config = read_config(config_path, validate_schema=True)
            evaluate_single_config(args, output_dir, config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Label propagation evaluation.')
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        required=False,
                        help="Path to the dataset root. If this path is not given, the default path is derived from "
                             "config.py")
    parser.add_argument("--disable-cuda", action='store_true',
                        default=False,
                        help="Disables the use of GPU for inference.")
    parser.add_argument("--output-dir",
                        type=str,
                        required=True,
                        help="Directory where outputs are saved.")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to the config file or config string. Paths cannot start with '{'. The script "
                             "understands config path mappings supported by config.resolve_args_path. Thus, a paths "
                             "'$config/<remainder>' is resolved to '<repo-location>/share/config/<remainder>'.")
    parser.add_argument("--only-new-configs",
                        action="store_true",
                        default=False,
                        help="Evaluates only configurations that haven't got a corresponding output directory. Only "
                             "applicable if the configurations are passed as a config directory. No validation of the "
                             "result directories is performed, only the presence of the corresponding result directory "
                             "is checked. Therefore, it is necessary to delete directories created by failed runs "
                             "beforehand.")
    parser.add_argument("--use-remote",
                        action="store_true",
                        default=False,
                        help="Set to true to indicate that the remote execution is desired. This makes the printed "
                             "output more friendly to remote logging tools and also measures the execution times.")
    parser.add_argument("--feature-cache",
                        default=None,
                        required=False,
                        help="Directory where feature cache files for each video sequence are stored. Use this option "
                             "only, if the configurations all use the same feature extractor. Otherwise the results "
                             "will be wrong! This is not validated automatically!")
    evaluate(parser.parse_args())
