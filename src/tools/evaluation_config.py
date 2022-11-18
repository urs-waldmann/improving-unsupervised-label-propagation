import json
import os
from typing import Dict, Optional, Union, Tuple

import jsonschema as jss
import torch
from torch import nn

import config as global_config
from dataset.BADJA import BadjaDataset
from dataset.JHMDB import Jhmdb
from dataset.PigeonDataset import get_pigeon_dataset
from dataset.SegTrackV2 import SegTrackV2
from dataset.davis import MultiObjectDavisDataset
from dataset.mask_dataset import MaskDataset
from keypoint import DcfKeypointLabelPropagation
from features import SwinFeatureExtractor, DummyFeatureExtractor, UvcFeatureExtractor, ViTFeatureExtractor, \
    StcFeatureExtractor
from keypoint import KeypointVideoIO, TrackingKeypointVideoIO, KeypointLabelCodec, KeypointCodec, \
    KeypointDatasetIterator
from label_prop import LabelPropagationEvaluator, OriginalUvcLabelPropagation, GroundTruthLabelInitializer, \
    BasicAffinityNorm, SoftmaxAffinityNorm
from segmentation import IndexedMaskLabelCodec, BinaryMaskLabelCodec, SingleObjectAttentionLabelInitializer, \
    SingleObjectAttentionLabelInitializerV2, SingleObjectAttentionLabelInitializerV3, MaskDatasetIterator
from utils.vis_utils import davis_color_map


def read_config(config_file_path: str, validate_schema=True) -> Dict:
    with open(config_file_path, 'r', encoding='utf8') as f:
        config_dict = json.load(f)

    if validate_schema:
        validate_config(config_dict)

    return config_dict


def write_config(config_file_path: str, config_dict: Dict, validate_schema=True) -> None:
    if validate_schema:
        validate_config(config_dict)

    with open(config_file_path, 'w', encoding='utf8') as f:
        json.dump(config_dict, f, indent='  ')


def validate_config(config_dict):
    with open(global_config.labelprop_schema_file, 'r', encoding='utf8') as f:
        schema = json.load(f)
    jss.validate(config_dict, schema=schema)


def instantiate_config(
        config: Union[Dict, str],
        device: torch.device,
        dataset_root: Optional[str],
        *,
        use_legacy_labelprop=False):
    if isinstance(config, str):
        config = read_config(config)

    feat_extractor = instantiate_feat_ext_config(config['feat_ext'], device)

    dataset = instantiate_dataset_config(config['dataset'], dataset_root, feat_extractor)

    if use_legacy_labelprop:
        label_prop = instantiate_legacy_labelprop_config(config['label_propagation'])
    else:
        label_prop = instantiate_labelprop_config(config['label_propagation'])

    evaluator = instantiate_evaluator_config(config['evaluator'], feat_extractor, label_prop)

    return evaluator, dataset


def instantiate_feat_ext_config(feature_config, device):
    if feature_config['name'] == 'swin':
        # TODO: Implement the additional options in the config and here
        feat_extractor = SwinFeatureExtractor(device=device)
    elif feature_config['name'] == 'dummy':
        feat_extractor = DummyFeatureExtractor(device=device)
    elif feature_config['name'] == 'uvc':
        feat_extractor = UvcFeatureExtractor(
            device=device,
            scale_size=feature_config["scale_size"],
            use_equal_sidelen=feature_config.get('use_equal_sidelen', True),
            model_key=feature_config.get('model_key', 'jhmdb')
        )
    elif feature_config['name'] == 'stc':
        feat_extractor = StcFeatureExtractor(
            device=device,
            scale_size=feature_config["scale_size"],
            use_equal_sidelen=feature_config.get('use_equal_sidelen', False)
        )
    elif feature_config['name'] == 'vit':
        feat_extractor = ViTFeatureExtractor(
            arch=f'vit_{feature_config["variant"]}',
            patch_size=feature_config["patch_size"],
            scale_size=feature_config["scale_size"],
            device=device,
            weight_source=feature_config.get('weight_source')
        )
    else:
        raise ValueError(f'Invalid architecture option: {feature_config["name"]}')
    return feat_extractor


def instantiate_dataset_config(dataset_config, dataset_root, feat_extractor):
    color_palette = davis_color_map()

    # Wrap legacy davis configuration into segmentation_dataset shim to allow simple instantiation.
    if dataset_config['name'] == 'davis':
        dataset_config = {
            'name': 'segmentation_dataset',
            'data': dataset_config
        }

    if dataset_config['name'] == 'segmentation_dataset':
        dataset = instantiate_segmentation_dataset(color_palette, dataset_config, dataset_root, feat_extractor)
    elif dataset_config['name'] == 'keypoint_dataset':
        dataset = instantiate_keypoint_dataset(color_palette, dataset_config, dataset_root, feat_extractor)
    else:
        raise ValueError(f'Invalid dataset name: {dataset_config["name"]}')
    return dataset


def instantiate_segmentation_dataset(color_palette, dataset_config, dataset_root, feat_extractor):
    if 'codec' in dataset_config:
        codec_config = dataset_config['codec']

        if codec_config['channel_normalization'] == 'minmax':
            norm_minmax = True
        elif codec_config['channel_normalization'] == 'none':
            norm_minmax = False
        else:
            raise ValueError(f'Invalid value for channel_normalization: {codec_config["channel_normalization"]}')

        codec_params = {
            "interpolation_mode": codec_config['interpolation_mode'],
            "normalize_minmax": norm_minmax
        }
    else:
        codec_params = {}

    data_config = dataset_config['data']

    mode = data_config.get('mode', 'multi-object')
    if mode == 'single-object':
        codec_class = BinaryMaskLabelCodec
    elif mode == 'multi-object':
        codec_class = IndexedMaskLabelCodec
    else:
        raise ValueError(f'Invalid inference mode: {mode}.')

    def codec_factory(input_size: Tuple[int, int], label_size: Tuple[int, int]):
        return codec_class(input_size, label_size, **codec_params)

    dataset_iter = MaskDatasetIterator(
        feat_extractor=feat_extractor,
        color_map=color_palette,
        label_codec_factory=codec_factory,
        dataset=(instantiate_segmentation_dataset_data(data_config, dataset_root))
    )
    return dataset_iter


def instantiate_segmentation_dataset_data(dataset_config, dataset_root) -> MaskDataset:
    name = dataset_config['name']

    if name == 'davis':
        return MultiObjectDavisDataset(
            dataset_root=dataset_root,
            year=dataset_config['year'],
            split=dataset_config['split']
        )
    elif name == 'segtrackv2':
        return SegTrackV2(dataset_root=dataset_root)
    else:
        raise ValueError(f'Invalid segmentation dataset name: {name}')


def instantiate_keypoint_dataset(color_palette, dataset_config, dataset_root, feat_extractor):
    if 'codec' in dataset_config:
        if 'keypoint_sigma' in dataset_config or 'keypoint_topk' in dataset_config:
            raise ValueError('Use of legacy codec parameters together with "codec" object.')

        codec_config = dataset_config['codec']

        def codec_factory(frame_size, label_size):
            return KeypointLabelCodec(
                KeypointCodec(
                    keypoint_size=frame_size,
                    label_size=label_size,
                    **codec_config
                ))
    else:
        def codec_factory(frame_size, label_size):
            return KeypointLabelCodec(
                KeypointCodec(
                    keypoint_size=frame_size,
                    label_size=label_size,
                    label_spread=dataset_config['keypoint_sigma'],
                    with_background=True,
                    decode_top_k=dataset_config['keypoint_topk']))
    save_config = dataset_config.get('save_config', {})
    data_config = dataset_config['data']
    if data_config['name'] == 'jhmdb':
        data_source = Jhmdb(
            ds_root=dataset_root,
            split_num=data_config['split_num'],
            split_type=data_config['split'],
            per_class_limit=data_config.get('per_class_limit', -1),
        )
    elif data_config['name'] == 'pigeon':
        data_source = get_pigeon_dataset(
            dataset_root=dataset_root,
            dataset_version=data_config.get('version', 'v1')
        )
    elif data_config['name'] == 'badja':
        data_source = BadjaDataset(clip_to_first_annotation=True)
    else:
        raise ValueError(f'Invalid keypoint dataset name: {data_config["name"]}')
    data_iter_type = dataset_config.get("dataset_iterator_type", "default")
    if data_iter_type == "default":
        base_class = KeypointVideoIO
    elif data_iter_type == "tracking":
        base_class = TrackingKeypointVideoIO
    else:
        raise ValueError(f'Invalid dataset_iterator_type: {data_iter_type}')
    dataset = KeypointDatasetIterator(
        feat_extractor=feat_extractor,
        color_map=color_palette,
        label_codec_factory=codec_factory,
        dataset=data_source,
        base_class=base_class,
        **save_config
    )
    return dataset


def instantiate_labelprop_config(labelprop_config):
    from label_prop.legacy import DinoAffinityNorm, UvcResnetAffinityNorm

    from label_prop import AffinityLabelPropagation, FullPropagator, UniversalPropagator, LocalPropagator

    if labelprop_config['name'] == 'affinity':

        aff_norm_name = labelprop_config['affinity_norm']

        if aff_norm_name == "none":
            aff_norm, aff_post_sel_norm = None, BasicAffinityNorm()
        elif aff_norm_name == "softmax":
            aff_norm, aff_post_sel_norm = None, SoftmaxAffinityNorm()
        elif aff_norm_name == "dino":
            aff_norm, aff_post_sel_norm = DinoAffinityNorm(), BasicAffinityNorm()
        elif aff_norm_name == "dino+softmax":
            aff_norm, aff_post_sel_norm = DinoAffinityNorm(), SoftmaxAffinityNorm()
        elif aff_norm_name == "uvc":
            aff_norm, aff_post_sel_norm = UvcResnetAffinityNorm(), BasicAffinityNorm()
        elif aff_norm_name == "uvc+softmax":
            aff_norm, aff_post_sel_norm = UvcResnetAffinityNorm(), SoftmaxAffinityNorm()
        else:
            raise ValueError(f'Invalid value of affinity_norm: {aff_norm_name}')

        if labelprop_config['topk_implementation'] == 'batched':
            use_batched_topk = True
        elif labelprop_config['topk_implementation'] == 'full':
            use_batched_topk = False
        else:
            raise ValueError(f'Invalid value of topk_implementation: {labelprop_config["topk_implementation"]}')

        if labelprop_config['label_normalization'] == 'minmax':
            label_norm = True
        elif labelprop_config['label_normalization'] == 'none':
            label_norm = False
        else:
            raise ValueError(f'Invalid value of label_normalization: {labelprop_config["label_normalization"]}')

        universal_propagator = UniversalPropagator(
            affinity_topk=labelprop_config['affinity_topk'],
            affinity_norm=aff_post_sel_norm,
        )
        if labelprop_config['implementation'] == 'full':
            propagator = FullPropagator(
                propagator=universal_propagator,
                neighborhood_size=labelprop_config['neighborhood_size'],
                use_batched_topk=use_batched_topk,
                apply_nh_to_reference=labelprop_config.get('apply_nh_to_reference', True),
            )
        elif labelprop_config['implementation'] == 'local':
            if 'apply_nh_to_reference' in labelprop_config:
                raise ValueError('Cannot combine "apply_nh_to_reference" with "implementation"="local".')

            propagator = LocalPropagator(
                propagator=universal_propagator,
                neighborhood_size=labelprop_config['neighborhood_size'],
                use_batched_topk=use_batched_topk
            )
        else:
            raise ValueError(f'Invalid implementation type for labelprop: {labelprop_config["implementation"]}')

        label_prop = AffinityLabelPropagation(
            propagator=propagator,
            feature_normalization=labelprop_config['feature_normalization'],
            affinity_normalization=aff_norm,
            label_normalization=label_norm,
        )
    elif labelprop_config['name'] == 'dcf':
        label_prop = DcfKeypointLabelPropagation(
            use_cos_window=labelprop_config['use_cos_window']
        )
    elif labelprop_config['name'] == 'uvc_original':
        label_prop = OriginalUvcLabelPropagation(affinity_top_k=labelprop_config['affinity_topk'])
    else:
        raise ValueError(f'Invalid value for labelpropagation: {labelprop_config["name"]}')
    return label_prop


def instantiate_legacy_labelprop_config(labelprop_config):
    from label_prop.legacy import DinoAffinityNorm, UvcResnetAffinityNorm, SoftmaxAffinityNorm, FullAffinity, \
        FullAffinityLabelPropagation, LocalAffinityLabelPropagation
    if labelprop_config['name'] == 'affinity':

        if labelprop_config['affinity_norm'] == "none":
            aff_norm = None
        elif labelprop_config['affinity_norm'] == "softmax":
            aff_norm = SoftmaxAffinityNorm()
        elif labelprop_config['affinity_norm'] == "dino":
            aff_norm = DinoAffinityNorm()
        elif labelprop_config['affinity_norm'] == "dino+softmax":
            aff_norm = nn.Sequential(DinoAffinityNorm(), SoftmaxAffinityNorm())
        elif labelprop_config['affinity_norm'] == "uvc":
            aff_norm = UvcResnetAffinityNorm()
        elif labelprop_config['affinity_norm'] == "uvc+softmax":
            aff_norm = nn.Sequential(UvcResnetAffinityNorm(), SoftmaxAffinityNorm())
        else:
            raise ValueError(f'Invalid value of affinity_norm: {labelprop_config["affinity_norm"]}')

        if labelprop_config['topk_implementation'] == 'batched':
            use_batched_topk = True
        elif labelprop_config['topk_implementation'] == 'full':
            use_batched_topk = False
        else:
            raise ValueError(f'Invalid value of topk_implementation: {labelprop_config["topk_implementation"]}')

        if labelprop_config['label_normalization'] == 'minmax':
            label_norm = True
        elif labelprop_config['label_normalization'] == 'none':
            label_norm = False
        else:
            raise ValueError(f'Invalid value of label_normalization: {labelprop_config["label_normalization"]}')

        if labelprop_config['implementation'] == 'full':
            label_prop = FullAffinityLabelPropagation(
                affinity=FullAffinity(
                    feature_normalization=labelprop_config['feature_normalization'],
                    affinity_normalization=aff_norm
                ),
                neighborhood_size=labelprop_config['neighborhood_size'],
                use_batched_topk=use_batched_topk,
                topk_k=labelprop_config['affinity_topk'],
                label_normalization=label_norm,
                apply_nh_to_reference=labelprop_config.get('apply_nh_to_reference', True)
            )
        elif labelprop_config['implementation'] == 'local':
            if 'apply_nh_to_reference' in labelprop_config:
                raise ValueError('Cannot combine "apply_nh_to_reference" with "implementation"="local".')

            label_prop = LocalAffinityLabelPropagation(
                feature_normalization=labelprop_config['feature_normalization'],
                affinity_normalization=aff_norm,
                neighborhood_size=labelprop_config['neighborhood_size'],
                use_batched_topk=use_batched_topk,
                topk_k=labelprop_config['affinity_topk'],
                label_normalization=label_norm
            )
        else:
            raise ValueError(f'Invalid implementation type for labelprop: {labelprop_config["implementation"]}')
    elif labelprop_config['name'] == 'dcf':
        label_prop = DcfKeypointLabelPropagation(
            use_cos_window=labelprop_config['use_cos_window']
        )
    elif labelprop_config['name'] == 'uvc_original':
        label_prop = OriginalUvcLabelPropagation(affinity_top_k=labelprop_config['affinity_topk'])
    else:
        raise ValueError(f'Invalid value for labelpropagation: {labelprop_config["name"]}')
    return label_prop


def instantiate_evaluator_config(evaluator_config, feat_extractor, label_prop):
    label_initializer_name = evaluator_config.get('label_initializer', 'ground_truth')
    if label_initializer_name == 'ground_truth':
        label_initializer = GroundTruthLabelInitializer(feat_extractor.device)
    elif label_initializer_name == 'so_attention':
        label_initializer = SingleObjectAttentionLabelInitializer(feat_extractor)
    elif label_initializer_name == 'so_attention_v2':
        label_initializer = SingleObjectAttentionLabelInitializerV2(
            feat_extractor, mass_percentage=0.6, sigma=1.0, num_iters=10, prb_limit=0.2)
    elif label_initializer_name == 'so_attention_v3':
        label_initializer = SingleObjectAttentionLabelInitializerV2(
            feat_extractor, mass_percentage=0.5, sigma=1.0, num_iters=10, prb_limit=0.2)
    elif label_initializer_name == 'so_attention_v4':
        label_initializer = SingleObjectAttentionLabelInitializerV3(
            feat_extractor, percentage=0.5, quantile=0.5, sigma=1.0, num_iters=10, prb_limit=0.2)
    elif label_initializer_name == 'so_attention_v5':
        label_initializer = SingleObjectAttentionLabelInitializerV3(
            feat_extractor, percentage=0.5, quantile=0.5, denoise_median_before_refine=3, interpolation_spline_deg=3,
            refine_mask=True, sxy=25, srgb=5, compat=5, sigma=5.0, num_iters=10, prb_limit=0.45,
            denoise_morph_before_fill=-1, denoise_median_before_fill=5, fill_holes=False)
    elif label_initializer_name == 'so_attention_v5_minprob':
        label_initializer = SingleObjectAttentionLabelInitializerV3(
            feat_extractor, percentage=0.5, quantile=0.5, denoise_median_before_refine=3, interpolation_spline_deg=3,
            refine_mask=True, sxy=25, srgb=5, compat=5, sigma=5.0, num_iters=10, prb_limit=0.2,
            denoise_morph_before_fill=-1, denoise_median_before_fill=5, fill_holes=False)
    else:
        raise ValueError(f'Invalid value for label_initializer: {evaluator_config["label_initializer"]}')

    evaluator = LabelPropagationEvaluator(
        feat_extractor=feat_extractor,
        label_prop=label_prop,
        num_context=evaluator_config['num_context'],
        recreate_labels=evaluator_config['recreate_labels'],
        label_initializer=label_initializer,
        use_first_frame_annotations=evaluator_config.get('use_first_frame_annotations', True)
    )
    return evaluator


if __name__ == '__main__':
    config_file = os.path.join(global_config.labelprop_config_root, 'config-test.json')
    print(read_config(config_file))
