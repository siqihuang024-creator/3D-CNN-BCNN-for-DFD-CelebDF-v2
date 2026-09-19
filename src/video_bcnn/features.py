"""Portable per-clip Phase-C feature-cache datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureCacheDataset(Dataset):
    """Expose every cached clip row whose (dataset,path) is requested."""

    def __init__(self, path, records):
        with np.load(path) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            datasets = np.asarray(payload["datasets"]).astype(str)
            paths = np.asarray(payload["paths"]).astype(str)
            clip_indices = (np.asarray(payload["clip_indices"], dtype=np.int64)
                            if "clip_indices" in payload else np.zeros(len(features), dtype=np.int64))
        if features.ndim != 2 or len(features) != len(paths):
            raise ValueError("Feature cache arrays have inconsistent shapes.")
        requested = ["{}:{}".format(row["dataset"], row["path"])
                     for row in records]
        rows_by_video = {}
        for index, (dataset, video) in enumerate(zip(datasets, paths)):
            rows_by_video.setdefault("{}:{}".format(dataset, video), []).append(index)
        missing = sorted(set(requested) - set(rows_by_video))
        if missing:
            raise ValueError("Feature cache misses {} requested videos; first: {}"
                             .format(len(missing), missing[0]))
        # Keep manifest order while retaining every cached clip for a video.
        indices = [index for key in requested for index in rows_by_video[key]]
        self.features = torch.from_numpy(features[indices])
        self.clip_indices = torch.from_numpy(clip_indices[indices])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"feature": self.features[index],
                "clip_index": self.clip_indices[index]}
