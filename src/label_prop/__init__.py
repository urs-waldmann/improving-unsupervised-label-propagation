"""
Label propagation implementation.
"""
from label_prop.affinity import *
from label_prop.affinity_label_propagation import *
from label_prop.affinity_norm import *
from label_prop.functional import *
from label_prop.label_codec import *
from label_prop.label_initializer import *
from label_prop.label_propagation_base import *
from label_prop.label_propagation_evaluator import *
from label_prop.original_uvc_label_propagation import *
from label_prop.video_io import *

__all__ = [
    # affinity_norm.py
    'AffinityNorm',
    'NoOpAffinityNorm',
    'BasicAffinityNorm',
    'SoftmaxAffinityNorm',

    # affinity_label_propagation.py
    'UniversalPropagator',
    'BasePropagator',
    'LocalPropagator',
    'FullPropagator',
    'AffinityLabelPropagation',

    # original_uvc_label_propagation.py
    'OriginalUvcLabelPropagation',

    # label_initializer.py
    'LabelInitializer',
    'GroundTruthLabelInitializer',
    # label_codec.py
    'AbstractLabelCodec',

    # label_propagation_base.py
    'AbstractLabelPropagation',

    # label_propagation_evaluator.py
    'LabelPropagationEvaluator',
    'CachedLabelPropagationEvaluator',

    # affinity.py
    'compute_raw_affinity',
    'local_affinity_mask',

    # video_io.py
    'LabelPropVideoIO',
    'LabelCodecFactory',
    'DatasetIterator',
]
