"""
eval.py
---

Evaluate on a dataset given a pretrained model
"""
import os

import matplotlib
import seaborn as sns
import torch
from torch import nn
from tqdm import tqdm

sns.set()
matplotlib.use('Agg')

import src.utils.utility as _util
import src.utils.cmd_line as _cmd
import src.data.msrvtt as _data
import src.model.models as _models
import src.train.train_test_utils as _train
from extern.coco_caption.pycocotools.coco import COCO
from src.data.caption import vocab

_logger = _util.get_logger(__file__)


def evaluate(raw: str, dataset: str, mode: str, weights_path: str, batch_size: int = 64, use_cuda: bool = False) -> None:
    dataset_dir = _util.get_dataset_by_name(dataset, mode)
    raw_dir = _util.get_raw_dataset_by_name(raw, mode)

    model, run, args, weights_path = _util.get_params_by_weights_path(weights_path)

    a_feature_size = int(args["a_feature_size"])
    projected_size = int(args["projected_size"])
    mid_size = int(args["mid_size"])
    hidden_size = int(args["hidden_size"])
    max_frames = int(args["max_frames"])
    max_words = int(args["max_words"])
    banet = _models.BANet(a_feature_size, projected_size, mid_size, hidden_size, max_frames, max_words, use_cuda=use_cuda)

    pretrained_path = os.path.join(weights_path, "weights.pth")
    weights = torch.load(pretrained_path)
    banet.load_state_dict(weights)
    if use_cuda:
        banet.cuda()

    print("Computing metrics...")
    eval_loader = _data.get_eval_dataloader(dataset, mode, batch_size=batch_size)
    test_reference_txt_path = os.path.join(dataset_dir, 'reference.json')
    test_prediction_txt_path = os.path.join(dataset_dir, 'prediction.txt')
    reference = COCO(test_reference_txt_path)

    _train.eval_step(eval_loader, banet, test_prediction_txt_path, reference, use_cuda=use_cuda)

    # Must switch to a new loder which provides captions.
    eval_loader = _data.get_dataloader(dataset, mode, batch_size=batch_size)
    for i, (videos, captions, cap_lens, video_ids) in tqdm(enumerate(eval_loader, start=1), total=len(eval_loader)):
        if use_cuda:
            videos = videos.cuda()

        video_encoded = banet.encoder(videos)
        tokens = banet.decoder.sample(video_encoded)

        # vid_paths = [os.path.join(raw_dir, "{}.mp4".format(video_id)) for video_id in video_ids]

        for j in range(len(tokens)):
            # vid = imageio.get_reader(vid_paths[j]).iter_data()

            print('[vid_id={}]'.format(video_ids[j]))
            print("gt  :", vocab().decode(captions[j]))
            print("pred:", vocab().decode(tokens.data[j].squeeze()))
            print()

            # First few frames are black sometimes
            # next(vid)
            # next(vid)
            # next(vid)
            # next(vid)
            # plt.imshow(next(vid))

def main():
    global _logger
    args = _cmd.parseArgsForClassOrScript(evaluate)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))
    evaluate(**varsArgs)

if __name__ == '__main__':
    main()
