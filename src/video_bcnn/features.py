"""Portable Phase-C feature-cache datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureCacheDataset(Dataset):
    def __init__(self, path, records):
        with np.load(path) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            datasets = np.asarray(payload["datasets"]).astype(str)
            paths = np.asarray(payload["paths"]).astype(str)
        if features.ndim != 2:
            raise ValueError("Cached features must be a [videos,dimensions] matrix.")
        lookup = {"{}:{}".format(dataset, video): index
                  for index, (dataset, video) in enumerate(zip(datasets, paths))}
        indices, missing = [], []
        for row in records:
            key = "{}:{}".format(row["dataset"], row["path"])
            if key not in lookup:
                missing.append(key)
            else:
                indices.append(lookup[key])
        if missing:
            raise ValueError("Feature cache misses {} requested videos; first: {}"
                             .format(len(missing), missing[0]))
        self.features = torch.from_numpy(features[indices])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"feature": self.features[index]}
