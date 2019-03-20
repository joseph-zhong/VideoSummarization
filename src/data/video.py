#!/usr/bin/env python3
"""
video.py
---

Utilities for loading video datasets and building video feature vectors
"""
import os
from typing import Tuple, List

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import src.utils.cmd_line as _cmd
import src.utils.utility as _util
from src.model.models import MotionEncoder, AppearanceEncoder

_logger = _util.get_logger(__file__)


def sample_frames(video_path: str, num_frames: int) -> Tuple[List[np.ndarray], None]:
    assert os.path.isfile(video_path), "Could not find video at path: {}".format(video_path)

    vid = imageio.get_reader(video_path)
    vid_len = sum(1 for _ in vid)
    vid = imageio.get_reader(video_path)

    # fps = vid.get_meta_data()['fps']
    dims = vid.get_data(0).shape

    period = vid_len // num_frames
    frames = []
    # clips = []
    # clip_len = min(2 * period, 16)
    # clip_seq = deque(maxlen=clip_len)
    for i, frame in enumerate(vid):
        # clip_seq.append(frame)
        if i % period == 0:
            if len(frames) == num_frames:
                break
            assert frame.shape == dims, "Shape of frame {} ({}) does not match starting frame {}".format(i, frame.shape, dims)
            frames.append(frame)
        # if i > period and len(clip_seq) == clip_len and i % period == 0:
        #     clips.append(list(clip_seq))
        #     clip_seq.clear()

    assert len(frames) == num_frames, "Could not construct {} frames from {} which contains only {} frames" \
        .format(num_frames, os.path.basename(video_path), vid_len)
    return frames, None  # clips


def resize_frame(frame, dims):
    if frame.ndim == 2:
        # We are given an grayscale image, just tile.
        frame = np.tile(frame[..., np.newaxis], 3)

    new_width, new_height = dims
    height, width, channels = frame.shape
    if height == width:
        resized = cv2.resize(frame, dims)
    elif height < width:
        resized = cv2.resize(frame, (width * new_height // height, new_width))
        crop = (resized.shape[1] - new_height) // 2
        resized = resized[:, crop:resized.shape[1]-crop]
    else:
        resized = cv2.resize(frame, (new_height, height * new_width // width))
        crop = (resized.shape[0] - new_width) // 2
        resized = resized[crop:resized.shape[0]-crop]

    return cv2.resize(resized, dims)


def preprocess_frame(frame, dims=(224, 224)):
    frame = resize_frame(frame, dims)
    frame = frame.astype(np.float32)
    # ImageNet de-mean.
    frame /= 255
    frame -= np.array([0.485, 0.456, 0.406])
    frame /= np.array([0.229, 0.224, 0.225])
    return frame


def extract_features(raw, dataset, mode, max_frames=-1, overwrite=False, aencoder=AppearanceEncoder(), mencoder=MotionEncoder()):
    """
    Builds appearance and motion features for a list of videos.

    :param raw: Raw dataset of videos for which to extract features.
    :param dataset: Dataset in which to place the resultant features.
    :param mode: Dataset mode (train, val, test).
    :param max_frames: Maximum number of allowable frames in a given video.
    :param overwrite: Unless this flag is specified, will fail rather than overwrite existing cache.
    :param aencoder: Encoder used for appearance.
    :param mencoder: Encoder used for motion.
    :return: Numpy of features with shape [len(videos), max_frames, aencoder.feature_size() + mencoder.feature_size()]
    where index i corresponds to the ith video in sorted(videos).
    """
    assert isinstance(raw, str)
    assert isinstance(dataset, str)
    assert isinstance(mode, str) and mode == "train" or mode == "val" or mode == "test", \
        "Extraction mode must be train, val, or test got {}".format(mode)
    assert isinstance(max_frames, int) and max_frames > 0, "max_frame must be a positive integer"
    assert isinstance(aencoder, nn.Module)
    assert isinstance(mencoder, nn.Module)

    raw_dir = _util.get_raw_dataset_by_name(raw)
    dataset_dir = _util.get_dataset_by_name(dataset, mode=mode, create=True)

    video_ids = sorted(set(_util.load_array(dataset_dir, "video_ids")))
    videos = [os.path.join(raw_dir, mode, "{}.mp4".format(video_id)) for video_id in video_ids]

    for video_path in videos:
        assert os.path.exists(video_path), "Cannot find mp4 video @ {}".format(video_path)

    aencoder = aencoder.cuda(1)
    # mencoder = mencoder#.cuda(0)
    # num_features = aencoder.feature_size() + mencoder.feature_size()
    num_features = aencoder.feature_size()

    features = np.zeros((len(videos), max_frames, num_features), dtype=np.float32)
    for i, video_path in enumerate(tqdm(videos)):
        frames, _ = sample_frames(video_path, max_frames)

        frames = np.array([preprocess_frame(f) for f in frames])
        frames = frames.transpose(0, 3, 1, 2)
        frames = torch.from_numpy(frames).cuda(1)

        # clips = np.array([[preprocess_frame(f) for f in clip] for clip in clips])
        # clips = clips.transpose(0, 4, 1, 2, 3)
        # clips = torch.from_numpy(clips).cuda(0)

        af = aencoder.forward(frames)
        # mf = mencoder.forward(clips)

        af = af.cpu().detach().numpy()
        # mf = mf.cpu().detach().numpy()

        features[i, :frames.shape[0], :] = af  #np.concatenate([af, mf], axis=1)

    _util.dump_array(dataset_dir, "frames", 100, features, overwrite=overwrite)
    return features


def main():
    global _logger
    args = _cmd.parseArgsForClassOrScript(extract_features)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger = _util.get_logger(__file__, verbosity=verbosity)
    _logger.info("Passed arguments: '{}'".format(varsArgs))
    extract_features(**varsArgs)


if __name__ == '__main__':
    main()
