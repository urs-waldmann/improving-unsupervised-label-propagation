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
        dict(name='keypoint_dataset', keypoint_topk=5, keypoint_sigma=0.5,
             data=dict(name='jhmdb', split='test', split_num=1, per_class_limit=2)
             )
    ]

    ctx_size = [0, 3, 7]
    recreate_labels = [True, False]

    evaluators = [dict(num_context=nc, recreate_labels=rec) for nc, rec in itertools.product(ctx_size, recreate_labels)]

    use_global_aff = True
    feat_norms = [True]
    aff_norms = ['dino', 'uvc', 'uvc+softmax', 'softmax', 'none']
    nh_sizes = [-1, 5, 12]
    batched_topk = [False]
    topk_ks = [3, 5]
    label_norm = [False, True]

    label_props = [
        dict(
            name='affinity',
            implementation='full' if use_global_aff else 'local',
            affinity_norm=aff_norm,
            topk_implementation='batched' if use_batched_topk else 'full',
            label_normalization='minmax' if use_label_norm else 'none',
            feature_normalization=feat_norm,
            affinity_topk=topk_k,
            neighborhood_size=nh_size,
        )
        for feat_norm, aff_norm, nh_size, use_batched_topk, topk_k, use_label_norm in itertools.product(
            feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm
        )
    ]

    for dataset, feat_ext, evaluator, label_prop in itertools.product(
            datasets, feat_extractors, evaluators, label_props):
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

    if not args.no_validate:
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
