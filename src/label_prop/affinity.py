import cv2
import numpy as np
import torch


def local_affinity_mask(h, w, nh_size):
    """
    Returns a neighborhood mask of shape h*w, h*w. Multiplying a flat grayscale image with this matrix is similar to
    convolution with a box filter of the given size.

    We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. local attention)

    :param h: height of the image
    :param w: width of the image
    :param nh_size: half size of the neighborhood. The full side length will be 2 * nh_size + 1
    :return: mask of shape [h*w, h*w]
    """
    assert nh_size >= 0

    # Dilation structuring element
    structure = np.ones(2 * nh_size + 1, dtype=np.uint8)

    # Dilation of the identity matrix creates the necessary diagonal lines in the output array.
    a = cv2.dilate(np.eye(w), structure)
    b = cv2.dilate(np.eye(h), structure)

    # Creates a copy of a for each element of b and multiplies the copy with the value of b.
    c = np.kron(b, a)
    return c


def compute_raw_affinity(feat_refs, feat_tar):
    """
    Computes the affinity between the target frame features and a set of reference frame features.

    :param feat_refs: Tensor of reference frame features: [num_ctx, num_feats, h*w]
    :param feat_tar: Target frame features. Tensor of shape [h*w, num_feats]
    :return: affinity matrix of shape [num_ctx, h*w, h*w], semantics: [tar, ref]
    """

    num_ctx, num_feats, hw = feat_refs.shape

    # Equivalent but slightly slower implementation:
    # feat_tar = feat_tar.unsqueeze(0).expand(num_ctx, -1, -1)  # [num_ctx, h*w, num_feats]
    # affinity = torch.bmm(feat_tar, feat_refs)  # [num_ctx, h*w, h*w], semantics: [tar, ref]

    feat_refs = feat_refs.transpose(0, 1)  # [num_feats, num_ctx, h*w]
    feat_refs = feat_refs.reshape(num_feats, num_ctx * hw)  # [num_feats, num_ctx*h*w]

    # [h*w, num_feats] * [num_feats, num_ctx*h*w] -> [num_ctx*h*w, h*w]
    affinity = torch.mm(feat_tar, feat_refs)  # [h*w, num_ctx*h*w], Semantics: [tar, ref]

    affinity = affinity.view(hw, num_ctx, hw)  # [h*w, num_ctx, h*w], Semantics: [tar, ref]
    affinity = affinity.transpose(0, 1)  # [num_ctx, h*w, h*w], Semantics: [tar, ref]

    return affinity
