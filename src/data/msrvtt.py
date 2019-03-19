import json
import os

import torch
import torch.utils.data as data

import src.utils.utility as _util

class MSRVTTDataset(data.Dataset):
    def __init__(self, dataset, mode) -> None:
        super().__init__()

        dataset_dir = _util.getDatasetByName(dataset, mode)
        self._features = _util.loadArray(dataset_dir, "frames")

        msrvtt_dir = _util.getRawDatasetByName("MSRVTT", mode)
        with open(os.path.join(msrvtt_dir, "annotations.json"), 'r') as f:
            j = json.load(f)

        splits = {}
        for video in j["videos"]:
            splits[video["video_id"]] = video["split"]

        video_ids = []
        captions = []
        lengths = []
        for sentence in j["sentences"]:
            video_ids.append(sentence["video_id"])
            captions.append(sentence["caption"])
            lengths.append(len(sentence["caption"]))

        vid2Idx = {video_id: idx for idx, video_id in enumerate(sorted(set(video_ids)))}
        for video_id in video_ids:
            assert splits[video_id] == mode, "{} should have split mode {} but has {}".format(video_id, splits[video_id], mode)
            assert video_id in vid2Idx, "Video {} not found in internal map".format(video_id)

        self._vid2Idx = vid2Idx
        self._video_ids = video_ids
        self._captions = captions
        self._lengths = lengths

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