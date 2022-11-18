import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.davis import MultiObjectDavisDataset
from dataset.evaluation.davis_utils import f_measure
from dataset.mask_dataset import SegmentationResultReader, MaskDataset, MaskVideo, SegmentationVideoResult
from utils.mask_utils import binmask_iou


class SegmentationMetric(ABC):
    @property
    @abstractmethod
    def metric_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def compute(self, gt, pred) -> float:
        raise NotImplementedError()


class JaccardMetric(SegmentationMetric):
    @property
    def metric_name(self) -> str:
        return 'J'

    def compute(self, gt, pred) -> float:
        return binmask_iou(gt, pred)


class BoundaryMetric(SegmentationMetric):
    @property
    def metric_name(self) -> str:
        return 'F'

    def compute(self, gt, pred) -> float:
        return f_measure(gt, pred)


def _score_decay(score_array):
    ids = (np.round(np.linspace(1, len(score_array), 5) + 1e-10) - 1).astype(np.int32)
    d_bins = [score_array[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    decay = np.nanmean(d_bins[0]) - np.nanmean(d_bins[3])
    return decay


def _score_recall(score_array):
    recall = np.nanmean(score_array > 0.5)
    return recall


def _score_mean(score_array):
    mean = np.nanmean(score_array)
    return mean


class VideoStatistic(Enum):
    MEAN = ('mean', _score_mean)
    RECALL = ('recall', _score_recall)
    DECAY = ('decay', _score_decay)

    @property
    def stat_name(self):
        return self.value[0]

    def __call__(self, *args, **kwargs):
        return self.value[1](*args, **kwargs)


class SegmentationDatasetResult(NamedTuple):
    mean: pd.Series
    statistics: pd.DataFrame
    raw_scores: pd.DataFrame


class VideoSegmentationEvaluator:
    metrics: List[SegmentationMetric]
    multi_object: bool
    skip_first_frame: bool
    skip_last_frame: bool

    def __init__(self,
                 metrics: List[SegmentationMetric] = None,
                 statistics: List[VideoStatistic] = None,
                 multi_object=True,
                 skip_first_frame=True,
                 skip_last_frame=True,
                 verbose=True):
        if metrics is None:
            metrics = [JaccardMetric(), BoundaryMetric()]
        if statistics is None:
            statistics = [VideoStatistic.MEAN, VideoStatistic.RECALL, VideoStatistic.DECAY]

        self.metrics = metrics
        self.statistics = statistics
        self.multi_object = multi_object
        self.skip_first_frame = skip_first_frame
        self.skip_last_frame = skip_last_frame
        self.verbose = verbose

    def evaluate_video(self, video: MaskVideo, predictions: SegmentationVideoResult) -> pd.DataFrame:
        if self.multi_object:
            ids = sorted(np.unique(video.mask_at(0)))
            ids = ids[:-1] if ids[-1] == 255 else ids
            ids = ids if ids[0] else ids[1:]
        else:
            ids = [1]

        start = 1 if self.skip_first_frame else 0
        end = len(video) - 1 if self.skip_last_frame else len(video)

        result_rows = []
        for fn in range(start, end):
            gt_mask = video.mask_at(fn)
            pred_mask = predictions.mask_at(fn)

            if not self.multi_object:
                gt_mask = (gt_mask > 0).astype(np.uint8)
                pred_mask = (pred_mask > 0).astype(np.uint8)

            for obj_id in ids:
                for metric in self.metrics:
                    score = metric.compute(gt_mask == obj_id, pred_mask == obj_id)

                    result_rows.append(dict(
                        frame_number=fn,
                        video_name=video.video_name,
                        metric_name=metric.metric_name,
                        obj_id=obj_id,
                        score=score
                    ))

        sequence_df = pd.DataFrame(result_rows)
        return sequence_df

    def evaluate(self, dataset: MaskDataset, result_reader: SegmentationResultReader) -> SegmentationDatasetResult:
        raw_df = self.compute_raw_results(dataset, result_reader)
        statistics = self.compute_statistics(raw_df)
        dataset_mean = self.compute_dataset_mean(statistics)
        return SegmentationDatasetResult(mean=dataset_mean, statistics=statistics, raw_scores=raw_df)

    def compute_raw_results(self, dataset: MaskDataset, result_reader: SegmentationResultReader) -> pd.DataFrame:
        results = []
        for vn, video in tqdm(enumerate(dataset), total=len(dataset), disable=not self.verbose):
            video_predictions = result_reader[video.video_name]

            video_results = self.evaluate_video(video, video_predictions)
            results.append(video_results)
        raw_df = pd.concat(results, axis=0)
        return raw_df

    def compute_statistics(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        def stat_mapper(group):
            score_array = np.array(group['score'])
            # mean, recall, decay = db_statistics(score_array)
            return pd.DataFrame([{stat.stat_name: stat(score_array) for stat in self.statistics}])

        statistics = (raw_df.sort_values('frame_number')
                      .groupby(['video_name', 'metric_name', 'obj_id'])
                      .apply(stat_mapper)
                      .reset_index(level=[2, 3], drop=True)
                      .reset_index())
        return statistics

    def compute_dataset_mean(self, statistics, add_jf=True) -> pd.Series:
        dataset_mean = statistics.groupby(['metric_name']).mean()

        dataset_mean = dataset_mean.stack(0)
        dataset_mean.index = dataset_mean.index.map(lambda x: f'{x[0]}_{x[1]}')

        if add_jf:
            if 'J_mean' in dataset_mean and 'F_mean' in dataset_mean:
                dataset_mean['J_and_F'] = (dataset_mean['J_mean'] + dataset_mean['F_mean']) / 2
            else:
                warnings.warn('Could not add J_and_F score. J_mean or F_mean is missing.')

        # Using string keys ensures a consistent sorting if unknown metrics are used
        sort_keys = dict(J_and_F='0', J_mean='1', J_recall='2', J_decay='3', F_mean='4', F_recall='5', F_decay='6')
        dataset_mean.sort_index(inplace=True, key=lambda x: x.map(lambda y: sort_keys.get(y, y)))

        return dataset_mean


if __name__ == '__main__':
    dataset = MultiObjectDavisDataset(year='2017', split='val')
    result_reader = SegmentationResultReader(r'K:\tasks\DAVIS2017val-ablation\baseline')

    evaluator = VideoSegmentationEvaluator()
    result = evaluator.evaluate(dataset, result_reader)
    print(result.mean)

    pass
