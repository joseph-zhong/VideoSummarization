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

def train_collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    # 根据video的长度对数据进行排序
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, captions, lengths, video_ids = zip(*data)

    # 把视频合并在一起（把2D Tensor的序列变成3D Tensor）
    videos = torch.stack(videos, 0)

    # 把caption合并在一起（把1D Tensor的序列变成一个2D Tensor）
    captions = torch.stack(captions, 0)
    return videos, captions, lengths, video_ids


def eval_collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, video_ids = zip(*data)

    # 把视频合并在一起（把2D Tensor的序列变成3D Tensor）
    videos = torch.stack(videos, 0)

    return videos, video_ids

MSRVTTDataset("MSRVTT", "train")
