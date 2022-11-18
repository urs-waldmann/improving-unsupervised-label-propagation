import argparse
import os

import torch

import utils.util
from tools.evaluate_config import evaluate_configuration


def config_from_args(args):
    if args.arch == 'vit':
        feat_ext_config = dict(
            name=args.arch, scale_size=args.scale_size, patch_size=args.patch_size, variant=args.size)
    elif args.arch == 'uvc':
        feat_ext_config = dict(name=args.arch, scale_size=args.scale_size)
    elif args.arch == 'dummy' or args.arch == 'swin':
        feat_ext_config = dict(name=args.arch)
    else:
        raise ValueError(f'Unsupported feature extractor: {args.arch}')

    if args.dataset == 'davis2016':
        dataset_config = dict(name='davis', year='2016', variant='val')
    elif args.dataset == 'davis2017':
        dataset_config = dict(name='davis', year='2017', variant='val')
    elif args.dataset == 'jhmdb_keypoint':
        dataset_config = dict(
            name='keypoint_dataset', keypoint_sigma=args.keypoint_sigma, keypoint_topk=args.keypoint_topk,
            data=dict(nam='jhmdb', split_num=1, split='test'))
    else:
        raise ValueError(f'Unsupported dataset name: {args.dataset}')

    config = {
        'feat_ext': feat_ext_config,
        'dataset': dataset_config,
        'label_propagation': {
            'name': 'affinity',
            'implementation': 'full',
            'affinity_norm': 'dino',
            'topk_implementation': 'full',
            'label_normalization': 'none',
            'feature_normalization': True,
            'affinity_topk': args.affinity_topk,
            'neighborhood_size': args.size_neighborhood,
        },
        'evaluator': {
            'num_context': args.num_context,
            'recreate_labels': False
        },
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Information propagation evaluator')

    parser.add_argument("--dataset", choices=["davis2016", "davis2017", "jhmdb_keypoint"],
                        default="davis2017", required=True, help="Dataset to use for evaluation.")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to the dataset root.")
    parser.add_argument("--disable-cuda", action='store_true', default=False,
                        help="Disables the use of GPU compute.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where outputs are saved.")
    parser.add_argument("--num-context", type=int, default=7,
                        help="Number of preceding frames used as context.")
    parser.add_argument("--size-neighborhood", default=12, type=int,
                        help="Size of the maximal allowed displacement in the reconstruction.")
    parser.add_argument("--affinity-topk", type=int, default=5,
                        help="Number of pixels used for the restoration of the propagated label.")
    parser.add_argument("--keypoint-topk", type=int, default=5,
                        help="Number of pixels used to compute keypoint coordinates.")
    parser.add_argument("--keypoint-sigma", type=float, default=0.5,
                        help="Spatial spread of the label functions.")
    parser.add_argument('--evaluator', default='affinity', choices=['affinity', 'dcf', 'uvc_original'],
                        help='Implementation of the evaluator to use.')

    arch_subparsers = parser.add_subparsers(metavar='ARCH', required=True, dest='arch',
                                            help='Feature extractor architecture.')

    vit_parser = arch_subparsers.add_parser("vit")
    vit_parser.add_argument("--size", choices=['tiny', 'small', 'base'], default='small')
    vit_parser.add_argument('--patch-size', default=16, type=int, choices={8, 16},
                            help='Model patch resolution')
    vit_parser.add_argument('--scale-size', default=480, type=int,
                            help='Scale shorter side of input to this size.')

    swin_parser = arch_subparsers.add_parser("swin")

    dummy_parser = arch_subparsers.add_parser("dummy")

    uvc_parser = arch_subparsers.add_parser("uvc")
    uvc_parser.add_argument('--scale-size', default=480, type=int,
                            help='Scale shorter side of input to this size.')

    arguments = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not arguments.disable_cuda else 'cpu')
    arguments.device = device
    config = config_from_args(arguments)

    os.makedirs(arguments.output_dir, exist_ok=True)
    arg_file = os.path.join(arguments.output_dir, 'arguments.txt')
    utils.util.save_args(arguments, arg_file, config=config)

    evaluate_configuration(config, device, arguments.dataset_path, arguments.output_dir, is_remote=False)
