import torch
import torch.nn.functional as F

from label_prop.label_propagation_base import AbstractLabelPropagation


class OriginalUvcLabelPropagation(AbstractLabelPropagation):
    def __init__(self, affinity_top_k):
        super().__init__()
        assert affinity_top_k is not None and affinity_top_k > 0

        self.topk_vis = affinity_top_k

    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [1, num_labels, h, w]
        """
        num_ctx, num_labels, h, w = label_refs.shape
        device = label_refs.device

        # [h*w, num_feats] -> [1, num_feats, 1, h, w]
        feat_tar = feat_tar.reshape(h, w, -1).permute(2, 0, 1).reshape(1, -1, 1, h, w).contiguous()

        # [num_ctx, num_feats, h*w] -> [B, num_feats, num_ctx, h, w]
        feat_refs = feat_refs.reshape(num_ctx, -1, h, w).permute(1, 0, 2, 3).reshape(1, -1, num_ctx, h, w).contiguous()

        # [num_ctx, num_labels, h, w] -> [num_ctx, h, w, num_labels]
        label_refs = label_refs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        new_labels = self._propagate_labels(feat_tar, feat_refs, label_refs)

        # [h, w, num_labels] -> [1, num_labels, h, w]
        new_labels = torch.from_numpy(new_labels).permute(2, 0, 1).unsqueeze(0).to(device=device)

        return new_labels

    def _propagate_labels(self, feat_tar, feat_ref, labels_ref):
        """

        :param feat_tar: [B, num_feats, 1, h, w]
        :param feat_ref: [B, num_feats, num_ctx, h, w]
        :param labels_ref: [num_ctx, h, w, num_labels]
        :return: propagated label array. Shape: [h, w, num_labels]
        """

        B, num_feats, num_ctx, h, w = feat_ref.shape

        affinity = self._compute_affinity(feat_tar, feat_ref)
        affinity = affinity.contiguous().view(B * num_ctx, h * w, h, w)

        label_predictions = OriginalUvcLabelPropagation._propagate_labels_impl(
            affinity, labels_ref, self.topk_vis)

        return label_predictions

    @staticmethod
    def _compute_affinity(tar_feats, ref_feats):
        """
        :param tar_feats: [B, num_feats, 1, h, w]
        :param ref_feats: [B, num_feats, video_len, h, w]
        :return: [B, video_len, h * w, h, w] Semantics: [B, vl, h*w ref, h tar, w tar]
        """

        B, num_feats, video_len, h, w = ref_feats.shape

        ref_feats_vec = ref_feats.view(B, num_feats, -1)  # [B, num_feats, video_len*h*w]
        ref_feats_vec = ref_feats_vec.transpose(1, 2)  # [B, video_len*h*w, num_feats]

        tar_feat_vec = tar_feats.view(B, num_feats, -1)  # [B, num_feats, h*w]

        affinity = torch.bmm(ref_feats_vec, tar_feat_vec)  # [B, video_len*h*w, h*w]

        # Softmax normalization for each ref frame
        affinity = affinity.view(B, video_len, h * w, h, w)
        affinity = F.softmax(affinity, dim=2)

        return affinity

    @staticmethod
    def _retain_topk(data, k, dim):
        top_values, top_indices = torch.topk(data, k, dim=dim, largest=True)
        output = torch.zeros_like(data)
        output.scatter_(dim=dim, index=top_indices, src=top_values)
        return output

    @staticmethod
    def _topk_bmm(m1, m2, k):
        """
        Top-k is performed along the Y axis of m1.

        :param m1: Shape: [B, X, Y]
        :param m2: Shape: [B, Y, Z]
        :param k:
        :return: [B, X, Z]
        """

        filtered_m1 = OriginalUvcLabelPropagation._retain_topk(m1, k=k, dim=2)
        return torch.bmm(filtered_m1, m2)

    @staticmethod
    def _normalize_labels(label_predictions):
        feat_h, feat_w, num_labels = label_predictions.size()

        label_predictions = label_predictions.view(-1, num_labels).clone()
        mask = label_predictions.sum(dim=0) != 0
        label_predictions[:, mask] -= label_predictions[:, mask].min(dim=0).values
        label_predictions[:, mask] /= label_predictions[:, mask].max(dim=0).values
        label_predictions = label_predictions.view(feat_h, feat_w, num_labels)
        return label_predictions

    @staticmethod
    def _propagate_labels_impl(affinity, ref_labels, k):
        """
        Propagates the given labels with the help of the given affinity matrix.

        :param affinity: Affinity of shape [video_len, h * w, h, w] Semantics: [ref, tar]
        :param ref_labels: Labels for the ref frames. Shape: [video_len, h, w, num_labels]
        :param k: number of pixels to use for the propagation
        :return: propagated label array. Shape: [h, w, num_labels]
        """

        vl, h, w, num_labels = ref_labels.shape
        device = affinity.device

        ref_labels = torch.from_numpy(ref_labels).to(dtype=torch.float32,
                                                     device=device)  # [video_len, h, w, num_labels]
        ref_labels = ref_labels.view(vl, h * w, num_labels)
        affinity = affinity.view(vl, h * w, h * w).transpose(1, 2)  # Shape [vl, hw, hw], Semantics: [tar, ref]

        # [vl, hw, hw] * [vl, hw, nl] -> [vl, hw, nl]
        label_predictions = OriginalUvcLabelPropagation._topk_bmm(affinity, ref_labels, k)

        # Average over all reference frames
        label_predictions = label_predictions.mean(dim=0)  # [h*w, num_labels]

        # Restore output size
        label_predictions = label_predictions.view(h, w, num_labels)

        return OriginalUvcLabelPropagation._normalize_labels(label_predictions).cpu().numpy()
