import os

import numpy as np
import pandas as pd

from lib import datasets_path

from .pd_dataset import PandasDataset
from datetime import datetime
from ..utils import sample_mask

class SyntheticData(PandasDataset):
    def __init__(self, impute_nans=False, freq='1S', masked_sensors=None):
        self.eval_mask = None
        df, dist, mask = self.load(impute_nans=impute_nans, masked_sensors=masked_sensors)
        if masked_sensors is None:
            self.masked_sensors = list()
        else:
            self.masked_sensors = list(masked_sensors)
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='synthetic', freq=freq, aggr='nearest')


    def load_raw(self):
        path = os.path.join(datasets_path['electricity'], 'electricity.txt')
        eval_mask = None
        content = np.load(path)
        df = pd.DataFrame(content['col'])
        adj = pd.DataFrame(content['adj'])
        return df, adj, eval_mask

    def load(self, impute_nans=True, masked_sensors=None, missing_ratio=0.8):
        # load readings metadata
        df, adj, eval_mask = self.load_raw()

        # compute the masks
        mask = (~np.isnan(df.values)).astype('uint8')

        if masked_sensors is not None:
            eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)
        self.eval_mask = eval_mask # 1 if the value is grond-truth for imputation else 0

        # eventually replace nans with 0
        if impute_nans:
            df = df.fillna(value=0)

        return df, adj, mask

    def get_similarity(self, thr=0.1):
        adj = self.dist
        return adj.to_numpy()

    @property
    def mask(self):
        return self._mask


class MissingSynthetic(SyntheticData):

    def __init__(self, missing_ratio=0.8, seed=1):
        super(MissingSynthetic, self).__init__()
        self.rng = np.random.default_rng(seed)
        missing_mask = np.random.rand(self.numpy().shape[0], self.numpy().shape[1])
        missing_ratio_mask = missing_mask < missing_ratio
        self.eval_mask = (self._mask & missing_ratio_mask).astype('uint8')


    @property
    def training_mask(self):
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))


    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]

