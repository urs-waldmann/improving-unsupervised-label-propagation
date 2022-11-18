import torch
import torch.nn.functional as F
from torch import nn

from label_prop.affinity import compute_raw_affinity, local_affinity_mask
from label_prop.affinity_norm import AffinityNorm
from label_prop.label_propagation_base import AbstractLabelPropagation
from utils.mask_utils import normalize_labels


class UniversalPropagator:
    """
    The universal propagator performs the actual label propagation. It is the responsibility of the caller to ensure,
    that the input data is encoded properly to achieve the desired effect. It can be configured to work with local and
    global affinity as well as batched and full propagation. Both of these are achieved by creating the correct input
    shapes.
    """

    def __init__(self, affinity_topk: int, affinity_norm: AffinityNorm):
        """
        :param affinity_topk: Number of reference locations to use for the label propagation. Must be >= 0 where 0
                              indicates that all locations are used.
        :param affinity_norm: Normalization function to apply to the input affinities.
        """
        assert affinity_topk is not None and affinity_topk >= 0
        self.affinity_topk = affinity_topk

        assert affinity_norm is not None
        self.affinity_norm = affinity_norm

    def retain_topk(self, affinities, dim):
        """
        Creates a sparse affinity matrix where only the top k values are retained.

        :param affinities: Input affinity matrix.
        :param dim: Dimensions along which the values are selected.
        :return: Affinity matrix of the same shape as the input affinities.
        """
        if self.affinity_topk < 1:
            return affinities

        top_values, top_indices = torch.topk(affinities, self.affinity_topk, dim=dim, largest=True)
        output = torch.zeros_like(affinities)
        output.scatter_(dim=dim, index=top_indices, src=top_values)
        return output

    def propagate(self, affinities, reference_values):
        """
        Computes bmm( norm(topk(affinities, k)), reference_values)

        :param affinities: Batch of affinities of shape [batch, ref, tar]
        :param reference_values: Batch of reference values [batch, channels, ref]
        :return: Propagated values of shape [batch, channels, tar]
        """

        assert affinities.ndim == 3 and reference_values.ndim == 3

        batch, ref_size, tar_size = affinities.shape
        batch2, channels, ref_size2 = reference_values.shape

        assert batch == batch2 and ref_size == ref_size2

        # Mask out all affinity entries below the topk number. -> Consider only the topk labels in the propagation.
        affinities = self.retain_topk(affinities, dim=1)
        affinities = self.affinity_norm(affinities, dim=1)

        # Reconstruct the mask for the target frame from the previous mask results
        # [batch, channels, ref] * [batch, ref, tar] -> [batch, channels, tar]
        target_values = torch.bmm(reference_values, affinities)

        return target_values


class BasePropagator:
    """
    Label propagator base class. This class represents an end-to-end label propagator that consumes features and label
    functions but performs the entire propagation, including affinity computation, internally.
    """

    def compute_affinity(self, feat_refs, feat_tar, label_refs):
        """
        Computes an affinity matrix from target and reference features.

        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: Affinity matrix of shape [num_ctx, tar, ref]
        """

        raise NotImplementedError()

    def propagate_labels(self, label_refs, affinity):
        """
        :param label_refs: [num_ctx, num_labels, h, w]
        :param affinity: Affinity matrix of shape [num_ctx, tar, ref]
        :return: Target labels of shape [1, num_labels, h, w]
        """
        raise NotImplementedError()


class LocalPropagator(BasePropagator):
    """
    Label propagation implementation that uses local affinity. The implementation of the affinity computation uses the
    SpatialCorrelationSampler module if it is installed. Otherwise, if falls back to a slower nn.Unfold-based
    implementation.
    """

    def __init__(self, propagator: UniversalPropagator, neighborhood_size: int, use_batched_topk: bool):
        """
        :param propagator: The actual propagation is delegated to this UniversalPropagator instance.
        :param neighborhood_size: Radius of the neighborhood size, i.e. actual size is 2 * nh_size + 1. Must be > 0.
        :param use_batched_topk: True to use a batched propagation implementation, otherwise full propagation is used.
        """
        super().__init__()
        assert neighborhood_size is not None and neighborhood_size > 0

        self.propagator = propagator
        self.neighborhood_size = neighborhood_size
        self.patch_size = neighborhood_size * 2 + 1
        self.use_batched_topk = use_batched_topk

        try:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.sampler = SpatialCorrelationSampler(
                kernel_size=1, patch_size=self.patch_size, stride=1, padding=0, dilation=1)
        except ImportError:
            print('Could not import SpatialCorrelationSampler. Falling back to unfold-based implementation.')
            self.sampler = None

    def _compute_local_correlation(self, feat_refs, feat_tar):
        """
        Computes the local affinities (aka correlation) between reference and target features.

        :param feat_refs: Shape: [num_context, num_feats, h, w]
        :param feat_tar: Shape: [1, num_feats, h, w]
        :return: Shape: [num_context, patch_size, patch_size, h, w]
        """

        # The SpatialCorrelationSampler is slower on the cpu but much faster on the gpu compared to unfold.
        if self.sampler is not None:
            feat_tar = feat_tar.contiguous()
            feat_refs = feat_refs.contiguous()
            if feat_refs.size(0) > 1:
                return torch.cat([self.sampler(feat_tar, feat_refs[i:i + 1]) for i in range(feat_refs.size(0))], dim=0)
            else:
                return self.sampler(feat_tar, feat_refs)
        else:
            b, c, h, w = feat_refs.shape
            cols = F.unfold(feat_refs, self.patch_size, padding=self.patch_size // 2)
            cols = cols.view(b, c, -1, h, w)

            prod = cols * feat_tar.view(1, c, 1, h, w)
            return prod.sum(dim=1).view(b, self.patch_size, self.patch_size, h, w)  # shape [b, patch_size**2, h, w]

    def compute_affinity(self, feat_refs, feat_tar, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [num_ctx, h*w, patch_size*patch_size], semantics: [tar, ref]
        """

        num_context, num_labels, h, w = label_refs.shape
        num_feats = feat_tar.shape[1]

        feat_tar = feat_tar.T.view(1, num_feats, h, w)
        feat_refs = feat_refs.view(num_context, num_feats, h, w)

        local_affinity = self._compute_local_correlation(feat_refs, feat_tar)
        local_affinity = local_affinity.view(num_context, self.patch_size * self.patch_size, h * w)

        local_affinity = local_affinity.transpose(1, 2)

        return local_affinity

    def propagate_labels(self, label_refs, affinity):
        num_ctx, num_lbl, h, w = label_refs.shape
        p2 = self.patch_size * self.patch_size

        # [num_ctx, num_lbl*patch_size*patch_size, h*w]
        label_cols = F.unfold(label_refs, kernel_size=self.patch_size, padding=self.neighborhood_size)
        # [num_ctx, num_lbl, patch_size*patch_size, h*w]
        label_cols = label_cols.view((num_ctx, num_lbl, p2, h * w))

        if self.use_batched_topk:
            batch = num_ctx * h * w
            ref = p2

            affinity = affinity.reshape(batch, ref, 1)
            label_cols = label_cols.permute(0, 3, 1, 2).reshape(batch, num_lbl, ref)

            # label_tar: [batch, channels, ref] [num_ctx*h*w, num_lbl, 1]
            label_tar = self.propagator.propagate(affinity, label_cols)
            label_tar = label_tar.view(num_ctx, h, w, num_lbl).permute(0, 3, 1, 2)  # [num_ctx, num_lbl, h, w]

            label_tar = label_tar.mean(0, keepdim=True)
        else:
            batch = 1 * h * w
            ref = num_ctx * p2

            affinity = affinity.permute(1, 0, 2).reshape(batch, ref, 1)
            label_cols = label_cols.permute(3, 1, 0, 2).reshape(batch, num_lbl, ref)

            # label_tar: [batch, channels, ref] [1*h*w, num_lbl, 1]
            label_tar = self.propagator.propagate(affinity, label_cols)
            label_tar = label_tar.view(1, h, w, num_lbl).permute(0, 3, 1, 2)  # [1, num_lbl, h, w]
        return label_tar


class FullPropagator(BasePropagator):
    """
    Label propagation implementation that uses full affinity.
    """

    def __init__(self, propagator: UniversalPropagator, neighborhood_size: int, use_batched_topk: bool,
                 *, apply_nh_to_reference: bool = True):
        """
        :param propagator: The actual propagation is delegated to this UniversalPropagator instance.
        :param neighborhood_size: Radius of the neighborhood size, i.e. actual size is 2 * nh_size + 1. Negative values
                                  indicate that no neighborhood should be used.
        :param use_batched_topk: True to use a batched propagation implementation, otherwise full propagation is used.
        :param apply_nh_to_reference: True to apply the neighborhood to the initial frame, too, not only the context
                                      frames. The reasoning to set this to false is, that the initial frame could be
                                      very old, compared to the context frames, thus the neighborhood would make it
                                      useless.
        """
        super().__init__()

        self.propagator = propagator
        self.neighborhood_size = neighborhood_size if neighborhood_size is not None else -1
        self.use_batched_topk = use_batched_topk
        self.apply_nh_to_reference = apply_nh_to_reference

        # Neighborhood mask cache. This variable is set to the most recently used neighborhood mask to avoid performing
        # the expensive recreation of this mask for each frame. It is updated every time the frame resolution changes,
        # which should not happen during the video.
        self._feat_h = None
        self._feat_w = None
        self._mask_neighborhood = None

    def compute_affinity(self, feat_refs, feat_tar, label_refs):
        affinity = compute_raw_affinity(feat_refs, feat_tar)  # [num_ctx, h*w, h*w], semantics: [tar, ref]

        if self.neighborhood_size >= 0:  # Mask enabled
            _, _, h, w = label_refs.shape
            if self._mask_neighborhood is None or self._feat_h != h or self._feat_w != w:
                # Create/recreate mask if not present or wrongly sized
                self._feat_h = h
                self._feat_w = w

                mask = local_affinity_mask(self._feat_h, self._feat_w, self.neighborhood_size)
                self._mask_neighborhood = torch.from_numpy(mask).to(device=label_refs.device, dtype=torch.float32)

            if self.apply_nh_to_reference:
                affinity *= self._mask_neighborhood
            else:
                affinity[1:, ...] *= self._mask_neighborhood

        return affinity

    def propagate_labels(self, label_refs, affinity):
        num_ctx, num_labels, h, w = label_refs.shape

        if self.use_batched_topk:
            affinity = affinity.transpose(2, 1)  # [num_ctx, h*w, h*w], semantics: [ref, tar]
            label_refs = label_refs.reshape(num_ctx, num_labels, -1)  # [num_ctx, num_labels, h*w]
            label_tar = self.propagator.propagate(affinity, label_refs)  # [num_ctx, num_labels, h*w]
            label_tar = label_tar.mean(dim=0)  # [num_labels, h*w]
        else:
            affinity = affinity.transpose(2, 1).reshape(1, -1, h * w)  # [1, num_ctx*h*w, h*w], semantics: [ref, tar]
            label_refs = label_refs.transpose(0, 1).reshape(1, num_labels, -1)  # [1, num_labels, num_ctx*h*w]
            label_tar = self.propagator.propagate(affinity, label_refs)  # [1, num_labels, h*w]

        label_tar = label_tar.view(1, num_labels, h, w)

        return label_tar


class AffinityLabelPropagation(AbstractLabelPropagation):
    """
    Wrapper class for the single-method AbstractLabelPropagation interface used in other parts of the code. This class
    wraps the propagation and normalization steps into a single method. It performs the tasks common to local and full
    propagation, i.e. feature, affinity and label normalization.
    """
    propagator: BasePropagator
    feature_normalization: bool
    affinity_normalization: nn.Module
    label_normalization: bool

    def __init__(self,
                 propagator: BasePropagator,
                 feature_normalization: bool,
                 affinity_normalization: nn.Module,
                 label_normalization: bool):
        super().__init__()

        self.propagator = propagator
        self.feature_normalization = feature_normalization
        self.affinity_normalization = affinity_normalization
        self.label_normalization = label_normalization

    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [1, num_labels, h, w]
        """

        if self.feature_normalization:
            feat_tar = F.normalize(feat_tar, p=2, dim=1)
            feat_refs = F.normalize(feat_refs, p=2, dim=1)

        affinity = self.propagator.compute_affinity(feat_refs, feat_tar, label_refs)

        if self.affinity_normalization is not None:
            affinity = self.affinity_normalization(affinity)

        label_tar = self.propagator.propagate_labels(label_refs, affinity)

        if self.label_normalization:
            label_tar = normalize_labels(label_tar, inplace=True)

        return label_tar
