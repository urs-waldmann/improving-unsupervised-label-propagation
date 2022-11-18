import argparse
import itertools
import json
import os
from typing import Iterator, Dict

from jsonschema import ValidationError

from config import resolve_args_path


def build_configs() -> Iterator[Dict]:
    feat_extractors = [
        dict(name='vit', scale_size=480, patch_size=8, variant='base'),
        dict(name='uvc', scale_size=480)
    ]

    datasets = [
        dict(
            name='keypoint_dataset',
            codec=dict(
                with_background=True,
                label_distribution_type='Gaussian',
                label_subpix_accurate=True,
                label_spread=sigma,
                decode_method='topk',
                decode_top_k=11
            ),
            save_config=dict(
                heatmap_name_pattern=None,
                marker_name_pattern=None,
                skeleton_name_pattern=None,
                labelmap_file_name=None,
            ),
            data=dict(
                name='jhmdb',
                split='test',
                split_num=1,
                per_class_limit=2
            )
        ) for sigma in [0.5, 1.0]
    ]

    evaluator = dict(num_context=7, recreate_labels=False)

    nh_sizes = [5, 12]
    topk_ks = [5, 10, 20]

    label_props = [
        dict(
            name='affinity',
            implementation='full',
            affinity_norm='uvc+softmax',
            topk_implementation='full',
            label_normalization='none',
            feature_normalization=True,
            affinity_topk=topk_k,
            neighborhood_size=nh_size,
        )
        for nh_size, topk_k in itertools.product(nh_sizes, topk_ks)
    ]

    for dataset, feat_ext, label_prop in itertools.product(datasets, feat_extractors, label_props):
        yield dict(feat_ext=feat_ext, dataset=dataset, evaluator=evaluator, label_propagation=label_prop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', metavar='CONFIG_DIR', type=str, required=False, default=None,
                        help='Output directory where the configs are generated.')
    parser.add_argument('--no-validation', action='store_true', default=False,
                        help='Bypass the config validation step.')
    args = parser.parse_args()

    base_dir = resolve_args_path(args.config_dir)

    configs = list(build_configs())
    print('Found ', len(configs), ' configs.')

    if not args.no_validation:
        from tools.evaluation_config import validate_config

        for config in configs:
            try:
                validate_config(config)
            except ValidationError as e:
                print('Configuration validation failed:')
                print(e)
                break

        print('Validated configurations')

    if base_dir is None:
        print('No output dir specified. No files are generated.')
        exit(1)

    print(f'Generating config files in "{base_dir}".')
    for i, config in enumerate(configs):
        out_file = os.path.join(base_dir, f'keypoint_config_{i:05d}.json')
        with open(out_file, 'w', encoding='utf8') as f:
            json.dump(config, f)
