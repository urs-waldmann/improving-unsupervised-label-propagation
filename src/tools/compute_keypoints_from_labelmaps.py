import argparse
import json
import os.path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

import config
import utils.path_utils
from keypoint import KeypointCodec


def evaluate_run(run_path: str, out_dir: str, codec_config: Dict, device, chunk_size, frame_size: Tuple[int, int]):
    video_names = utils.path_utils.list_subdirs(run_path, relative=True)

    codec = None

    for video_name in tqdm(video_names, desc=os.path.basename(out_dir)):
        label_file = os.path.join(run_path, video_name, 'labelmap.npy')

        labels = torch.from_numpy(np.load(label_file))

        if codec is None:
            b, c, h, w = labels.shape
            codec = KeypointCodec(
                keypoint_size=frame_size, label_size=(w, h), with_background=True,
                label_distribution_type='Gaussian', label_spread=1.0, label_subpix_accurate=True,
                **codec_config
            )

        if chunk_size > 0:
            chunks = [labels[s:s + chunk_size, ...] for s in range(0, labels.shape[0], chunk_size)]
            kp_chunks = [codec.decode(chunk.to(device=device)) for chunk in chunks]
            keypoints = torch.cat(kp_chunks, dim=0)
        else:
            keypoints = codec.decode(labels.to(device=device))  # shape: [num_frames, xy, num_keypoints]

        # The usual evaluation code expects shape: [xy, num_keypoints, num_frames]
        keypoints = np.transpose(keypoints, (1, 2, 0))

        video_out_dir = os.path.join(out_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        keypoint_file = os.path.join(video_out_dir, 'keypoints.npy')
        np.save(keypoint_file, keypoints)


def main(args):
    in_dir = args.input_dir

    if not os.path.isdir(in_dir):
        raise ValueError(f'Input path is not a directory: "{in_dir}"')

    try:
        if args.label_codecs.startswith('[') or args.label_codecs.startswith('{'):
            codec_configs = json.loads(args.label_codecs)
        else:
            label_codecs_path = config.resolve_args_path(args.label_codecs)
            with open(label_codecs_path, 'r', encoding='utf8') as f:
                codec_configs = json.load(f)
    except json.JSONDecodeError as e:
        print('Failed to parse codec configuration.')
        print(e)
        exit(1)
        return

    device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')

    if isinstance(codec_configs, dict):
        codec_configs = [codec_configs]

    codec_keys = {'decode_method', 'decode_top_k', 'decode_upscale_factor', 'decode_moment_threshold'}
    for cfg in codec_configs:
        ks = set(cfg.keys())
        if not ks.issubset(codec_keys):
            invalid = ks - codec_keys
            raise ValueError(f'Codec configuration contains invalid keys {invalid}. Allowed keys are {codec_keys}.')
        if 'decode_method' not in ks:
            raise ValueError('Codec configuration must contain key "decode_method".')

    def conf_to_string(conf):
        out = conf['decode_method']
        if 'decode_top_k' in conf:
            out += f'_top{conf["decode_top_k"]}'
        if 'decode_upscale_factor' in conf:
            out += f'_upscale{conf["decode_upscale_factor"]}'
        if 'decode_moment_threshold' in conf:
            out += f'_mthresh{conf["decode_moment_threshold"]}'
        return out

    dataset = args.dataset
    if dataset == 'jhmdb':
        frame_size = (320, 240)
    elif dataset == 'pigeonv1' or dataset == 'pigeonv2':
        frame_size = (1920, 1080)
    else:
        raise ValueError(f'Invalid dataset name {dataset}')

    config_paths = utils.path_utils.list_subdirs(in_dir) if args.eval_all else [in_dir]

    for config_path in config_paths:
        config_name = os.path.basename(config_path)

        for codec_config in codec_configs:
            output_path = os.path.join(args.output_dir, f'{config_name}_{conf_to_string(codec_config)}')
            os.makedirs(output_path, exist_ok=True)

            evaluate_run(config_path, output_path, codec_config, device, args.chunk_size, frame_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='Input directory to a single configuration or multiple configs. The interpretation of the'
                             'input directory depends on the usage of the "--eval-all" flag')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output root where the directories for all configurations should be created.')
    parser.add_argument('--label-codecs', type=str, required=True,
                        help='Json object with config params or array of config params.')
    parser.add_argument('--eval-all', action='store_true', default=False,
                        help='Interprets the input directory as the root of multiple input configs instead of a single'
                             'configuration.')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disables the use of cuda.')
    parser.add_argument('--chunk-size', type=int, default=-1,
                        help='Chunk size to use for processing of the videos. This helps alleviate the memory pressure.')
    parser.add_argument('--dataset', type=str, choices={'jhmdb', 'pigeonv1', 'pigeonv2'}, default='jhmdb',
                        help='Dataset that is evaluated. This influences the assumed frame size.')
    main(parser.parse_args())
