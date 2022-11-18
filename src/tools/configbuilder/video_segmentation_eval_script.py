import argparse
import os

from tools.evaluation_config import write_config
from tools.configbuilder.video_segmentation_eval_configs import label_prop_configs


def evaluate(args):
    config_root = args.output_dir
    os.makedirs(config_root, exist_ok=True)

    prop_configs = list(label_prop_configs())
    for i, (label_prop_name, label_prop) in enumerate(prop_configs):
        config = dict(
            feat_ext=dict(name='vit', scale_size=args.scale_size, patch_size=args.patch_size, variant=args.size),
            dataset=dict(name='davis', split=args.dataset_variant, year=args.dataset_year),
            evaluator=dict(num_context=args.num_context, recreate_labels=False),
            label_propagation=label_prop
        )

        config_path = os.path.join(config_root, label_prop_name + '.json')
        write_config(config_path, config, validate_schema=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Information propagation evaluator')

    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where outputs are saved.")

    parser.add_argument("--dataset-year", type=str, default='2017', choices={'2017', '2016'},
                        help="Davis dataset year.")
    parser.add_argument("--dataset-variant", type=str, default='train', choices={'train', 'val'},
                        help="Davis dataset variant.")
    parser.add_argument("--num-context", type=int, default=7,
                        help="Number of preceding frames used as context.")
    parser.add_argument("--size", choices=['tiny', 'small', 'base'], default='small')
    parser.add_argument('--patch-size', default=16, type=int, choices={8, 16},
                        help='Model patch resolution')
    parser.add_argument('--scale-size', default=480, type=int,
                        help='Scale shorter side of input to this size.')

    evaluate(parser.parse_args())
