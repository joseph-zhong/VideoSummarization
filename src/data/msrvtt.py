"""
msrvtt.py
---

Defines MSRVTT dataset.
"""

import numpy as np
import torch
import torch.utils.data as _data

import src.utils.utility as _util


class MSRVTTDataset(_data.Dataset):
    def __init__(self, dataset: str, mode: str) -> None:
        super().__init__()

        dataset_dir = _util.get_dataset_by_name(dataset, mode)
        self._features = _util.load_array(dataset_dir, "frames")
        self._captions = _util.load_array(dataset_dir, "captions")
        self._lengths = _util.load_array(dataset_dir, "cap_lens")
        self._video_ids = _util.load_array(dataset_dir, "video_ids")
        self._vid2idx = {video_id: idx for idx, video_id in enumerate(sorted(set(self._video_ids)))}

    def __getitem__(self, index):
        video_id = self._video_ids[index]
        feature = torch.from_numpy(self._features[self._vid2idx[video_id]])
        captions = torch.from_numpy(self._captions[index])
        return feature, captions, self._lengths[index], video_id

    def __len__(self):
        return len(self._captions)


class VideoDataset(_data.Dataset):
    def __init__(self, dataset: str, mode: str) -> None:
        super().__init__()

        dataset_dir = _util.get_dataset_by_name(dataset, mode)
        self._features = _util.load_array(dataset_dir, "frames")
        self._video_ids = _util.load_array(dataset_dir, "video_ids")
        self._vid2idx = {video_id: idx for idx, video_id in enumerate(sorted(set(self._video_ids)))}

    def __getitem__(self, index):
        video_id = self._video_ids[index]
        feature = torch.from_numpy(self._features[self._vid2idx[video_id]])
        return feature, video_id

    def __len__(self):
        return len(self._video_ids)


def train_collate_fn(data):
    # Sort data by cap_len.
    data.sort(key=lambda x: x[-2], reverse=False)

    videos, captions, lengths, video_ids = zip(*data)

    # Combine videos into 3D tensor.
    videos = torch.stack(videos, 0)

    # Merge captions together. Turns 1D sequence into 2D.
    captions = torch.stack(captions, dim=0)
    return videos, captions, lengths, video_ids


def eval_collate_fn(data):
    # Sort data by video_id.
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, video_ids = zip(*data)

    # Combine videos into 3D tensor.
    videos = torch.stack(videos, 0)

    return videos, video_ids


def get_train_dataloader(dataset: str, batch_size=10, shuffle=True, num_workers=3, pin_memory=True) -> _data.dataloader:
    return get_dataloader(dataset, "train", batch_size, shuffle, num_workers, pin_memory)


def get_dataloader(dataset: str, mode: str, batch_size=10, shuffle=True, num_workers=3, pin_memory=True) -> _data.dataloader:
    vtt = MSRVTTDataset(dataset, mode)
    return torch.utils.data.DataLoader(dataset=vtt,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=train_collate_fn,
                                       pin_memory=pin_memory)


def get_eval_dataloader(dataset: str, mode: str, batch_size=200, shuffle=False, num_workers=1, pin_memory=False) -> _data.dataloader:
    ds = VideoDataset(dataset, mode)
    return torch.utils.data.DataLoader(dataset=ds,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=eval_collate_fn,
                                       pin_memory=pin_memory)
