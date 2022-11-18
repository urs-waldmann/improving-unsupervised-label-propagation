import copy
import os
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from features import FeatureExtractor
from label_prop.label_propagation_base import AbstractLabelPropagation
from label_prop.video_io import LabelPropVideoIO
from label_prop.label_initializer import GroundTruthLabelInitializer, LabelInitializer
from utils.ring_buffer import RingBuffer


class LabelPropagationEvaluator:
    """
    Evaluation helper class that performs label propagation with a fixed number of context frames.

    A fixed number of preceding frames is used as the reference to predict the labels for the next frame.
    """

    def __init__(self,
                 feat_extractor: FeatureExtractor,
                 num_context: int,
                 label_prop: AbstractLabelPropagation,
                 recreate_labels: bool = False,
                 *,
                 verbose=True,
                 label_initializer: LabelInitializer = None,
                 use_first_frame_annotations=True):
        assert feat_extractor is not None
        assert num_context is not None
        assert num_context + (1 if use_first_frame_annotations else 0) > 0
        assert label_prop is not None

        self.feat_extractor = feat_extractor
        self.n_last_frames = num_context
        self.label_prop = label_prop
        self.recreate_labels = recreate_labels
        self.use_first_frame_annotations = use_first_frame_annotations

        self.device = self.feat_extractor.device

        self.label_initializer = GroundTruthLabelInitializer(
            self.device) if label_initializer is None else label_initializer

        self.verbose = verbose

    def extract_features(self, frame: np.ndarray):
        preprocessed_frame = self.feat_extractor.preprocess(frame)
        features = self.feat_extractor.extract(preprocessed_frame, flat=True)
        return features

    def eval_video(self, frame_source: LabelPropVideoIO):
        self.label_prop.reset_for_next_video()

        f1 = frame_source.frame_at(0)
        f1_feat = self.extract_features(f1).T  # [num_feats, h*w]
        f1_labels = self.label_initializer.initialize_labels(frame_source)

        frame_source.save_frame_result(0, f1, f1_labels)

        # The queue stores the n preceding frames
        queue: RingBuffer[Tuple[torch.Tensor, torch.Tensor]] = RingBuffer(self.n_last_frames)

        if not self.use_first_frame_annotations:
            queue.add((f1_feat, f1_labels))

        frame_iter = range(1, len(frame_source))
        if self.verbose:
            frame_iter = tqdm(frame_iter, total=len(frame_source))

        for frame_num in frame_iter:
            target_frame = frame_source.frame_at(frame_num)

            feat_tar = self.extract_features(target_frame)  # [h*w, num_feats]

            # we use the first segmentation and the n previous ones
            queue_contents = queue.get_contents()
            if len(queue_contents) == 0:
                used_frame_feats, used_segs = [], []
            else:
                used_frame_feats, used_segs = map(list, zip(*queue_contents))

            if self.use_first_frame_annotations:
                used_frame_feats = [f1_feat] + used_frame_feats
                used_segs = [f1_labels] + used_segs

            feat_refs = torch.stack(used_frame_feats)  # [num_ctx, num_feats, h*w]
            label_refs = torch.cat(used_segs)  # [num_ctx, num_labels, h, w]

            label_tar = self.label_prop(feat_tar, feat_refs, label_refs)

            if self.recreate_labels:
                label_tar = frame_source.label_codec.recode(label_tar)

            # Update the frame queue
            queue.add((feat_tar.T, copy.deepcopy(label_tar)))

            frame_source.save_frame_result(frame_num, target_frame, label_tar)


class CachedLabelPropagationEvaluator:
    """
    Hacky solution to load features from saved arrays instead of recomputing them every time.

    This implementation relies on the linear access pattern of the default evaluator. extract_features must be called
    exactly once per frame in temporal order, otherwise the results are wrong. Alternatively, we could compute hashes of
    the frame content and use it to match it to an index, but for our use case the simpler approach should suffice.

    This implementation reads the features for an entire video at once. For long videos this can easily lead to OOM
    errors.
    """

    def __init__(self, cache_dir: str, evaluator: LabelPropagationEvaluator):
        """
        :param cache_dir: Directory with feature cache files named <video_name>.pth
        :param evaluator: Evaluator to use for the actual evaluation. Instance is modified, don't reuse it elsewhere!
        """
        self.evaluator = evaluator

        # Change the evaluator implementation of the feature extraction to load features from the cache instead.
        self.evaluator.extract_features = self.extract_features

        self._cache_dir = cache_dir

        # We keep track of the used features in these variables.
        # Tracks the location in the current video.
        self._current_index = None
        # CPU array with features per frame.
        self._current_feats = None

    def extract_features(self, frame: np.ndarray):
        assert self._current_feats is not None and self._current_index is not None

        vid_len, h, w, num_feats = self._current_feats.shape
        assert self._current_index < vid_len

        # Extract the feature slice for this frame, reshape and move to GPU.
        feats = (self._current_feats[self._current_index, ...]
                 .view(h * w, num_feats)
                 .to(self.evaluator.feat_extractor.device))

        # Increment for the next usage. This is where the trouble begins if the access pattern is not linear.
        self._current_index += 1
        return feats

    def eval_video(self, frame_source: LabelPropVideoIO):
        # FIXME: This only works, because all current instances of VideoIO have a video variable, which is not
        #  guaranteed by the base class. Future implementations could break this. A refactoring of the LabelPropVideoIO
        #  and dataset iteration situation will probably solve this problem by using a common base class for all videos.
        #  This removes the need for specialized VideoIO classes.
        video_name = frame_source.video.video_name

        self._current_index = 0

        # Deleting the old feature array first prevents a memory usage spike during loading where. We cannot simply
        # reuse the same memory, because the videos don't necessarily have the same length and resolution.
        del self._current_feats

        # We map the features to the CPU memory, because the video memory is usually exhausted more easily.
        self._current_feats = torch.load(
            os.path.join(self._cache_dir, f'{video_name}.pth'), map_location='cpu')

        return self.evaluator.eval_video(frame_source)
