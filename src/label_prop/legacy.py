import torch
from torch import nn as nn
from torch.nn import functional as F

from label_prop.label_propagation_base import AbstractLabelPropagation
from label_prop.affinity import local_affinity_mask, compute_raw_affinity
from utils.mask_utils import normalize_labels


def retain_topk_v1(data, k, dim):
    assert k >= 0
    # My version
    top_values, top_indices = torch.topk(data, k, dim=dim, largest=True)
    output = torch.zeros_like(data)
    output.scatter_(dim=dim, index=top_indices, src=top_values)
    return output


def retain_topk_v3(data, k, dim):
    assert k >= 0
    # From dino but modified for efficient value setting
    top_values = torch.topk(data, dim=dim, k=k).values
    top_values_min = torch.min(top_values, dim=dim).values
    mask = data < top_values_min.unsqueeze(dim)
    data.masked_fill_(mask=mask, value=0)
    return data


# There are slight semantic differences between the different implementations.
retain_topk = retain_topk_v1


def topk_bmm(m1, m2, k):
    """
    Top-k is performed along the Y axis of m1.

    The implementation is equivalent to the following code but avoids explicitly filtering m1:
    filtered_m1 = retain_topk(m1, k=k, dim=2)
    out = torch.bmm(filtered_m1, m2)

    :param m1: Shape: [B, X, Y]
    :param m2: Shape: [B, Y, Z]
    :param k:
    :return: [B, X, Z]
    """
    B = m1.size(0)
    X = m1.size(1)
    Y = m1.size(2)
    Z = m2.size(2)

    top_values, top_indices = torch.topk(m1, k, dim=2, largest=True)  # [B, X, k]

    # Find the label coefficients for the top k values
    top_indices = top_indices.view(B, X, k, 1, 1).expand(-1, -1, -1, -1, Z)  # [B, X, k, 1, Z]
    m2 = m2.view(B, 1, 1, Y, Z).expand(-1, X, k, -1, -1)  # [B, X, k, Y, Z]
    top_labels = torch.gather(m2, dim=3, index=top_indices)  # [B, X, k, 1, Z]
    top_labels = top_labels.squeeze(3)  # [B, X, k, Z]

    # Actual multiplication and summation of the label and affinity values
    label_predictions = top_labels * top_values.unsqueeze(-1)  # [B, X, k, Z]
    label_predictions = label_predictions.sum(dim=2)  # [B, X, Z]

    return label_predictions


def propagate_labels_mm(affinity, ref_labels, k, *, normalize_softmax=False, softmax_temperature=20):
    """
    Propagates the reference label information to the target frame. The pixels of all reference frames compete at once.
    The reconstruction is formed directly from all winning pixels of all reference frames. k is shared across all
    reference frames.

    :param affinity: Affinity matrix of shape [num_ctx, h*w, h*w], semantics: [tar, ref]
    :param ref_labels: [num_ctx, num_labels, h, w]
    :param k: Number of positions to consider in the reconstruction of a single pixel
    :return: tar_label of shape [1, num_labels, h, w]
    """

    num_ctx, num_labels, h, w = ref_labels.shape

    ref_labels = ref_labels.transpose(0, 1).reshape(num_labels, -1)  # [num_labels, num_ctx*h*w]
    affinity = affinity.transpose(2, 1).reshape(-1, h * w)  # [num_ctx*h*w, h*w], semantics: [ref, tar]

    # Mask out all affinity entries below the topk number. -> Consider only the topk labels in the propagation.
    if k > 0:
        affinity = retain_topk(affinity, k=k, dim=0)

    # Re-normalize affinity to sum up to 1 for each pixel
    if normalize_softmax:
        affinity = F.softmax(affinity * softmax_temperature, dim=0)
    else:
        affinity = affinity / torch.sum(affinity, keepdim=True, dim=0)

    # Reconstruct the mask for the target frame from the previous mask results
    # [num_labels, num_ctx*h*w] * [num_ctx*h*w, h*w] -> [num_labels, h*w]
    tar_label = torch.mm(ref_labels, affinity)

    tar_label = tar_label.view(1, num_labels, h, w)
    return tar_label


def propagate_labels_bmm(affinity, ref_labels, k):
    """
    Propagates the reference label information to the target frame. Each of the reference frames in propagated
    individually and the final result is formed by averaging the reference frame propagations. k is applies to each
    reference frame individually.

    :param affinity: Affinity matrix of shape [num_ctx, h*w, h*w], semantics: [tar, ref]
    :param ref_labels: [num_ctx, num_labels, h, w]
    :param k: Number of positions to consider in the reconstruction of a single pixel
    :return: tar_label of shape [1, num_labels, h, w]
    """

    num_ctx, num_labels, h, w = ref_labels.shape

    ref_labels = ref_labels.view(num_ctx, num_labels, -1)  # [num_ctx, num_labels, h*w]
    affinity = affinity.transpose(2, 1)  # [num_ctx, h*w, h*w], semantics: [ref, tar]

    # Mask out all affinity entries below the topk number. -> Consider only the topk labels in the propagation.
    if k > 0:
        affinity = retain_topk(affinity, k=k, dim=1)

    # TODO: THis is new!!!
    # affinity = affinity / torch.sum(affinity, keepdim=True, dim=1)

    # Reconstruct the mask for the target frame from the previous mask results
    # [num_ctx, num_labels, h*w] * [num_ctx, h*w, h*w] -> [num_ctx, num_labels, h*w]
    tar_label = torch.bmm(ref_labels, affinity)
    tar_label = tar_label.mean(dim=0)  # [num_labels, h*w]

    tar_label = tar_label.view(1, num_labels, h, w)
    return tar_label


class DinoAffinityNorm(nn.Module):
    def __init__(self, factor=0.1):
        super().__init__()
        self.factor = factor

    def forward(self, affinity):
        return torch.exp_(affinity / self.factor)


class UvcResnetAffinityNorm(nn.Module):
    def __init__(self, factor=512 ** -0.5):
        super().__init__()
        self.factor = factor

    def forward(self, affinity):
        return affinity / self.factor


class SoftmaxAffinityNorm(nn.Module):
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = temperature

    def forward(self, affinity):
        if self.temperature is not None:
            affinity = affinity * self.temperature

        return F.softmax(affinity, dim=2)


class FullAffinity(nn.Module):
    feature_normalization: bool
    affinity_normalization: nn.Module

    def __init__(self, feature_normalization: bool, affinity_normalization: nn.Module):
        super().__init__()
        self.feature_normalization = feature_normalization
        self.affinity_normalization = affinity_normalization

    def forward(self, feat_tar, feat_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :return: [num_ctx, h*w, h*w], semantics: [tar, ref]
        """

        if self.feature_normalization:
            feat_tar = F.normalize(feat_tar, p=2, dim=1)
            feat_refs = F.normalize(feat_refs, p=2, dim=1)

        affinity = compute_raw_affinity(feat_refs, feat_tar)

        if self.affinity_normalization is not None:
            affinity = self.affinity_normalization(affinity)

        return affinity


class FullAffinityLabelPropagation(AbstractLabelPropagation):
    affinity: FullAffinity
    neighborhood_size: int
    use_batched_topk: bool
    topk_k: int
    label_normalization: bool

    def __init__(self, affinity: FullAffinity, neighborhood_size: int, use_batched_topk: bool, topk_k: int,
                 label_normalization: bool, *, apply_nh_to_reference: bool = True):
        super().__init__()

        self.affinity = affinity
        self.neighborhood_size = neighborhood_size
        self.use_batched_topk = use_batched_topk
        self.topk_k = topk_k
        self.label_normalization = label_normalization
        self.apply_nh_to_reference = apply_nh_to_reference

        self._feat_h = None
        self._feat_w = None
        self._mask_neighborhood = None

    def get_mask(self, label_refs):
        if self.neighborhood_size is not None and self.neighborhood_size >= 0:
            # Mask enabled
            _, _, h, w = label_refs.shape
            if self._mask_neighborhood is None or self._feat_h != h or self._feat_w != w:
                # Create/recreate mask if not present or wrongly sized
                self._feat_h = h
                self._feat_w = w

                mask = local_affinity_mask(self._feat_h, self._feat_w, self.neighborhood_size)
                self._mask_neighborhood = torch.from_numpy(mask).to(device=label_refs.device, dtype=torch.float32)
        else:
            # Mask disabled
            self._mask_neighborhood = None

        return self._mask_neighborhood

    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [1, num_labels, h, w]
        """

        affinity = self.affinity(feat_tar, feat_refs)  # [num_ctx, h*w, h*w], semantics: [tar, ref]

        affinity_mask = self.get_mask(label_refs)
        if affinity_mask is not None:
            if self.apply_nh_to_reference:
                affinity *= affinity_mask
            else:
                affinity[1:, ...] *= affinity_mask

        if self.use_batched_topk:
            label_tar = propagate_labels_bmm(affinity, label_refs, self.topk_k)
        else:
            label_tar = propagate_labels_mm(affinity, label_refs, self.topk_k)

        if self.label_normalization:
            label_tar = normalize_labels(label_tar, inplace=True)

        return label_tar


class LocalAffinity(nn.Module):
    feature_normalization: bool
    affinity_normalization: nn.Module

    def __init__(self,
                 feature_normalization: bool,
                 affinity_normalization: nn.Module,
                 patch_size: int,
                 use_sampler: bool = True):
        super().__init__()
        self.feature_normalization = feature_normalization
        self.affinity_normalization = affinity_normalization
        self.patch_size = patch_size

        self.use_sampler = use_sampler
        if use_sampler:
            from spatial_correlation_sampler import SpatialCorrelationSampler

            self.sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.patch_size,
                stride=1,
                padding=0,
                dilation=1)

    def _compute_local_correlation(self, feat_refs, feat_tar):
        # The SpatialCorrelationSampler is slower on the cpu but much faster on the gpu compared to unfold.
        if self.use_sampler:
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

    def _compute_local_aff(self, feat_tar, feat_refs, label_refs):
        num_context, num_labels, h, w = label_refs.shape
        num_feats = feat_tar.shape[1]

        feat_tar = feat_tar.T.view(1, num_feats, h, w)
        feat_refs = feat_refs.view(num_context, num_feats, h, w)

        local_affinity = self._compute_local_correlation(feat_refs, feat_tar)
        local_affinity = local_affinity.view(num_context, self.patch_size * self.patch_size, h * w)

        local_affinity = local_affinity.transpose(1, 2)
        return local_affinity

    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [num_ctx, h*w, patch_size*patch_size], semantics: [tar, ref]
        """

        if self.feature_normalization:
            feat_tar = F.normalize(feat_tar, p=2, dim=1)
            feat_refs = F.normalize(feat_refs, p=2, dim=1)

        affinity = self._compute_local_aff(feat_tar, feat_refs, label_refs)

        if self.affinity_normalization is not None:
            affinity = self.affinity_normalization(affinity)

        return affinity


class LocalAffinityLabelPropagation(AbstractLabelPropagation):
    feature_normalization: bool
    affinity_normalization: nn.Module
    neighborhood_size: int
    use_batched_topk: bool
    topk_k: int
    label_normalization: bool

    def __init__(self,
                 feature_normalization: bool,
                 affinity_normalization: nn.Module,
                 neighborhood_size: int,
                 use_batched_topk: bool,
                 topk_k: int,
                 label_normalization: bool):
        super().__init__()

        # TODO: enforce positive patch size
        self.patch_size = neighborhood_size * 2 + 1
        self.affinity = LocalAffinity(feature_normalization, affinity_normalization, self.patch_size)

        self.neighborhood_size = neighborhood_size
        self.use_batched_topk = use_batched_topk
        self.topk_k = topk_k
        self.label_normalization = label_normalization

    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [1, num_labels, h, w]
        """

        num_ctx, num_lbl, h, w = label_refs.shape

        # [num_ctx, h*w, patch_size*patch_size], semantics: [tar, ref]
        local_affinity = self.affinity(feat_tar, feat_refs, label_refs)
        # [num_ctx, patch_size*patch_size, h*w], semantics: [ref, tar]
        local_affinity = local_affinity.transpose(1, 2)

        # [num_ctx, num_lbl*patch_size*patch_size, h*w]
        label_cols = F.unfold(label_refs, kernel_size=self.patch_size, padding=self.neighborhood_size)
        # [num_ctx, num_lbl, patch_size*patch_size, h*w]
        label_cols = label_cols.view((num_ctx, num_lbl, self.patch_size * self.patch_size, h * w))

        if self.topk_k > 0 and not self.use_batched_topk:
            # [num_ctx*patch_size*patch_size, h*w]
            local_affinity = local_affinity.view(num_ctx * self.patch_size * self.patch_size, h * w)
            local_affinity = retain_topk(local_affinity, self.topk_k, dim=0)
            local_affinity /= torch.sum(local_affinity, dim=0)

            # [num_ctx, 1, patch_size*patch_size, h*w]
            local_affinity = local_affinity.view(num_ctx, 1, self.patch_size * self.patch_size, h * w)

            # [num_ctx, num_lbl, h*w]
            label_tar = torch.sum(local_affinity * label_cols, dim=[0, 2]).unsqueeze(0)

            # # [num_ctx, patch_size*patch_size, num_lbl, h*w]
            # label_cols = label_cols.permute(0, 2, 1, 3)
            # # [num_ctx*patch_size*patch_size, num_lbl, h*w]
            # label_cols = label_cols.reshape((num_ctx * self.patch_size * self.patch_size, num_lbl, h * w))
            #
            # # [num_ctx*patch_size*patch_size, 1, h*w]
            # local_affinity = local_affinity.view(num_ctx * self.patch_size * self.patch_size, 1, h * w)
            # # [k, 1, h*w], [k, 1, h*w]
            # values, indices = local_affinity.topk(dim=0, k=self.topk_k)
            #
            # # Make topk values sum to 1
            # values = values / torch.sum(values, dim=0)
            #
            # # [k, num_lbl, h*w]
            # label_cols = torch.gather(label_cols, dim=0, index=indices.expand(self.topk_k, num_lbl, h * w))
            # # [num_lbl, h*w]
            # label_tar = torch.sum(values * label_cols, dim=0)
        else:
            # [num_ctx, 1, patch_size*patch_size, h*w], semantics: [ref, tar]
            local_affinity = local_affinity.unsqueeze(1)

            if self.topk_k > 0:
                # Topk over patch dimension
                local_affinity = retain_topk(local_affinity, k=self.topk_k, dim=2)

            # Inner product over patch values
            # [num_ctx, num_lbl, h*w]
            label_tar = torch.sum(local_affinity * label_cols, dim=2)
            # label_tar = torch.einsum('clks,clks->cls', local_affinity, label_cols)

            # Average over all label predictions
            # [num_lbl, h*w]
            label_tar = label_tar.mean(dim=0)

        label_tar = label_tar.view((1, num_lbl, h, w))

        if self.label_normalization:
            label_tar = normalize_labels(label_tar, inplace=True)

        return label_tar


class AffUvcEvalLabelPropagation(AbstractLabelPropagation):

    def __init__(self, size_mask_neighborhood: int, affinity_topk: int):
        super().__init__()
        assert size_mask_neighborhood is not None
        assert affinity_topk is not None and affinity_topk > 0

        self.size_mask_neighborhood = size_mask_neighborhood
        self.affinity_topk = affinity_topk

        self._feat_h = None
        self._feat_w = None
        self._mask_neighborhood = None

    def get_mask(self, label_refs):
        if self.size_mask_neighborhood is not None and self.size_mask_neighborhood >= 0:
            # Mask enabled
            _, _, h, w = label_refs.shape
            if self._mask_neighborhood is None or self._feat_h != h or self._feat_w != w:
                # Create/recreate mask if not present or wrongly sized
                self._feat_h = h
                self._feat_w = w

                mask = local_affinity_mask(self._feat_h, self._feat_w, self.size_mask_neighborhood)
                self._mask_neighborhood = torch.from_numpy(mask).to(device=label_refs.device, dtype=torch.float32)
        else:
            # Mask disabled
            self._mask_neighborhood = None

        return self._mask_neighborhood

    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [1, num_labels, h, w]
        """

        num_ctx, num_labels, h, w = label_refs.shape

        # Compute the affinity and apply softmax normalization
        affinity = compute_raw_affinity(feat_refs, feat_tar)  # [num_ctx, h*w, h*w] semantics: (tar, ref)
        affinity = F.softmax(affinity, dim=2)

        # Mask affinity to weight closer pixels more than ones farther away
        affinity_mask = self.get_mask(label_refs)
        if affinity_mask is not None:
            affinity = affinity * affinity_mask

        label_refs = label_refs.view(num_ctx, num_labels, -1)  # [num_ctx, num_labels, h*w]
        affinity = affinity.transpose(2, 1)  # [num_ctx, h*w, h*w], semantics: [ref, tar]

        # Mask out all affinity entries below the topk number. -> Consider only the topk labels in the propagation.
        affinity = retain_topk(affinity, k=self.affinity_topk, dim=1)

        # Reconstruct the mask for the target frame from the previous mask results
        # [num_ctx, num_labels, h*w] * [num_ctx, h*w, h*w] -> [num_ctx, num_labels, h*w]
        label_tar = torch.bmm(label_refs, affinity)
        label_tar = label_tar.mean(dim=0)  # [num_labels, h*w]

        label_tar = label_tar.view(1, num_labels, h, w)

        return label_tar
