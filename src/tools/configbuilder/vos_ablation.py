import argparse
import os
from copy import deepcopy

from config import resolve_args_path
from tools.evaluation_config import read_config, write_config


def build_configs(config_dir):
    variations = [
        ('dataset.codec.channel_normalization', 'none'),
        ('dataset.codec.interpolation_mode', 'nearest'),
        ('dataset.codec.interpolation_mode', 'bicubic'),
        ('evaluator.num_context', 1),
        ('evaluator.num_context', 3),
        ('evaluator.num_context', 5),
        ('evaluator.num_context', 7),
        ('evaluator.num_context', 10),
        ('evaluator.num_context', 15),
        ('evaluator.num_context', 20),
        ('evaluator.recreate_labels', True),
        ('label_propagation.topk_implementation', 'batched'),
        ('label_propagation.affinity_norm', 'dino+softmax'),
        ('label_propagation.affinity_norm', 'softmax'),
        ('label_propagation.affinity_norm', 'none'),
        ('label_propagation.label_normalization', 'minmax'),
        ('label_propagation.feature_normalization', False),
        ('label_propagation.affinity_topk', 3),
        ('label_propagation.affinity_topk', 5),
        ('label_propagation.affinity_topk', 10),
        ('label_propagation.affinity_topk', 15),
        ('label_propagation.affinity_topk', 20),
        ('label_propagation.affinity_topk', 25),
        ('label_propagation.neighborhood_size', -1),
        ('label_propagation.neighborhood_size', 3),
        ('label_propagation.neighborhood_size', 5),
        ('label_propagation.neighborhood_size', 7),
    ]

    baseline_path = os.path.join(config_dir, 'baseline.json')
    baseline = read_config(baseline_path, validate_schema=True)

    for key, value in variations:
        config = deepcopy(baseline)
        config_path = os.path.join(config_dir, f'{key}.{value}.json')

        key_parts = key.split('.')
        sub = config
        for part in key_parts[:-1]:
            sub = sub[part]
        sub[key_parts[-1]] = value

        write_config(config_path, config, validate_schema=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', metavar='CONFIG_DIR', type=str, required=True,
                        help='Output directory where the configs are generated.')
    args = parser.parse_args()

    build_configs(resolve_args_path(args.config_dir))
