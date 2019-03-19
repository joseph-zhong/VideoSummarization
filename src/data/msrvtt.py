"""
msrvtt.py
---

Defines MSRVTT dataset.
"""
import torch
import torch.utils.data as data

import src.utils.utility as _util


class MSRVTTDataset(data.Dataset):
    def __init__(self, dataset, mode) -> None:
        super().__init__()

        dataset_dir = _util.getDatasetByName(dataset, mode)
        self._features = _util.loadArray(dataset_dir, "frames")
        self._captions = _util.loadArray(dataset_dir, "captions")
        self._lengths = _util.loadArray(dataset_dir, "cap_lens")
        self._video_ids = _util.loadArray(dataset_dir, "video_ids")
        self._vid2Idx = {video_id: idx for idx, video_id in enumerate(sorted(set(self._video_ids)))}

    def __getitem__(self, index):
        video_id = self._video_ids[index]
        feature = torch.from_numpy(self._features[self._vid2Idx[video_id]])
        return feature, self._captions[index], self._lengths[index], video_id

    def __len__(self):
        return len(self._captions)

MSRVTTDataset("MSRVTT", "train")