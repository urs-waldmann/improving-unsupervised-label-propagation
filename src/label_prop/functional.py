from label_prop import UniversalPropagator
from label_prop.affinity_norm import NoOpAffinityNorm, BasicAffinityNorm, SoftmaxAffinityNorm


def propagate_full(affinity, ref_labels, k, *, normalize_softmax=False, softmax_temperature=20):
    """
    Propagates the reference label information to the target frame. The pixels of all reference frames compete at once.
    The reconstruction is formed directly from all winning pixels of all reference frames. k is shared across all
    reference frames.

    :param affinity: Affinity matrix of shape [num_ctx, h*w, h*w], semantics: [tar, ref]
    :param ref_labels: [num_ctx, num_labels, h, w]
    :param k: Number of positions to consider in the reconstruction of a single pixel
    :return: tar_label of shape [1, num_labels, h, w]
    """

    aff_norm = SoftmaxAffinityNorm(temperature=softmax_temperature) if normalize_softmax else BasicAffinityNorm()
    prop = UniversalPropagator(affinity_topk=k, affinity_norm=aff_norm)

    num_ctx, num_labels, h, w = ref_labels.shape
    affinity = affinity.transpose(2, 1).reshape(1, -1, h * w)  # [1, num_ctx*h*w, h*w], semantics: [ref, tar]
    ref_labels = ref_labels.transpose(0, 1).reshape(1, num_labels, -1)  # [1, num_labels, num_ctx*h*w]

    tar_label = prop.propagate(affinity, ref_labels)  # [1, num_labels, h*w]

    tar_label = tar_label.view(1, num_labels, h, w)
    return tar_label


def propagate_batched(affinity, ref_labels, k):
    """
    Propagates the reference label information to the target frame. Each of the reference frames in propagated
    individually and the final result is formed by averaging the reference frame propagations. k is applies to each
    reference frame individually.

    :param affinity: Affinity matrix of shape [num_ctx, h*w, h*w], semantics: [tar, ref]
    :param ref_labels: [num_ctx, num_labels, h, w]
    :param k: Number of positions to consider in the reconstruction of a single pixel
    :return: tar_label of shape [1, num_labels, h, w]
    """

    prop = UniversalPropagator(affinity_topk=k, affinity_norm=NoOpAffinityNorm())

    num_ctx, num_labels, h, w = ref_labels.shape

    affinity = affinity.transpose(2, 1)  # [num_ctx, h*w, h*w], semantics: [ref, tar]
    ref_labels = ref_labels.reshape(num_ctx, num_labels, -1)  # [num_ctx, num_labels, h*w]

    tar_label = prop.propagate(affinity, ref_labels)  # [num_ctx, num_labels, h*w]

    tar_label = tar_label.mean(dim=0)  # [num_labels, h*w]
    tar_label = tar_label.view(1, num_labels, h, w)
    return tar_label
