import argparse
import os
from copy import deepcopy

from config import resolve_args_path
from tools.evaluation_config import read_config, write_config


def build_configs(config_dir):
    variations = [
        ('evaluator.num_context', 1),
        ('evaluator.num_context', 3),
        ('evaluator.num_context', 5),
        ('evaluator.num_context', 10),
        ('evaluator.num_context', 15),
        ('evaluator.num_context', 20),
        ('evaluator.use_first_frame_annotations', False),
        ('label_propagation.neighborhood_size', 3),
        ('label_propagation.neighborhood_size', 7),
        ('label_propagation.neighborhood_size', 9),
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
