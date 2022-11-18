import itertools
from typing import Iterator, Tuple, Callable, Dict


def build_config(feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm, *, use_global_aff=True) -> \
        Iterator[Tuple[str, Dict]]:
    for feat_norm, aff_norm, nh_size, use_batched_topk, topk_k, use_label_norm in itertools.product(
            feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm
    ):
        name = f'FeatNorm{feat_norm}' \
               f'_{aff_norm}' \
               f'_nhsize{nh_size}' \
               f'_BatchedTopK{use_batched_topk}' \
               f'_K{topk_k}' \
               f'_LabelNorm{use_label_norm}'

        labelprop_config = dict(
            name='affinity',
            implementation='full' if use_global_aff else 'local',
            affinity_norm=aff_norm,
            topk_implementation='batched' if use_batched_topk else 'full',
            label_normalization='minmax' if use_label_norm else 'none',
            feature_normalization=feat_norm,
            affinity_topk=topk_k,
            neighborhood_size=nh_size,
        )

        yield name, labelprop_config


def label_prop_base_configs() -> Iterator[Tuple[str, Dict]]:
    feat_norms = [True, False]
    aff_norms = ['dino', 'uvc', 'dino+softmax', 'uvc+softmax', 'softmax', 'none']
    nh_sizes = [-1, 5, 12]
    batched_topk = [True, False]
    topk_ks = [5, 10, 20]
    label_norm = [False]

    return build_config(feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm)


def label_prop_nh_size_configs() -> Iterator[Tuple[str, Dict]]:
    feat_norms = [True]
    aff_norms = ['dino']
    nh_sizes = [1, 2, 3, 4, 7, 9, 11, 13, 14, 15, 16, 17, 18, 20, 24, 30]
    batched_topk = [False]
    topk_ks = [5]
    label_norm = [False]

    return build_config(feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm)


def label_prop_nh_size_configs2() -> Iterator[Tuple[str, Dict]]:
    feat_norms = [True]
    aff_norms = ['dino']
    nh_sizes = [19, 21, 22]
    batched_topk = [False]
    topk_ks = [5]
    label_norm = [False]

    return build_config(feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm)


def label_prop_nh_size_configs3() -> Iterator[Tuple[str, Dict]]:
    feat_norms = [True]
    aff_norms = ['dino']
    nh_sizes = [19]
    batched_topk = [False]
    topk_ks = [5]
    label_norm = [False]

    return build_config(feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm)


def label_prop_nh_size_configs4() -> Iterator[Tuple[str, Dict]]:
    feat_norms = [True]
    aff_norms = ['dino']
    nh_sizes = [19]
    batched_topk = [False]
    topk_ks = [5]
    label_norm = [False]

    return build_config(feat_norms, aff_norms, nh_sizes, batched_topk, topk_ks, label_norm, use_global_aff=False)


label_prop_configs: Callable[[], Iterator[Tuple[str, Dict]]] = label_prop_base_configs

if __name__ == '__main__':
    configs = list(label_prop_configs())
    num_configs = len(configs)
    print(f'Number of configurations: {num_configs}')

    for name, _ in configs:
        print(name)
