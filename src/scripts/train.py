#!/usr/bin/env python3
"""
train.py
---

This is the primary training script.

Usage

"""

import os
import pickle
import shutil
import inspect

import torch
import numpy as np
import tensorboard_logger as _tb_logger

import src.data.msrvtt as _data
import src.model.models as _models
import src.train.train_test_utils as _train

import src.utils.utility as _util
import src.utils.cmd_line as _cmd

from extern.coco_caption.pycocotools.coco import COCO

_logger = _util.getLogger(__file__)

tmp_dir = 'tmp'
val_reference_txt_path = os.path.join(tmp_dir, 'msr-vtt_val_references.txt')
val_prediction_txt_path = os.path.join(tmp_dir, 'msr-vtt_val_predictions.txt')

test_reference_txt_path = os.path.join(tmp_dir, 'msr-vtt_test_references.txt')
test_prediction_txt_path = os.path.join(tmp_dir, 'msr-vtt_test_predictions.txt')


def train(
    dataset: str,
    num_epochs=100,
    batch_size=100,

    learning_rate=3e-4,
    ss_factor=24,
    max_ss=0.6,
    use_cuda=True,
    use_ckpt=False,

    projected_size=500,
    hidden_size=1024,  # Hidden size of the recurrent cells.
    mid_size=128,  # Dimension of the boundary detection layer.

    frame_shape=(3, 224, 224),  # Video frame shape.
    a_feature_size=2048,  # Appearance model feature-dimension size.

    # REVIEW josephz: Remove this?
    # m_feature_size=4096,  # Motion model feature-dimension size.

    frame_sample_rate=10,  # Sample rate of video frames.
    max_frames=100,  # Maximum length of the video-frame sequence.
    max_words=30,  # Maximum length of the caption-word sequence.
):
    # Prepare output paths.
    params = {arg_name: arg.default for arg_name, arg in inspect.signature(train).parameters.items()}
    ckpt_path = _util.getWeightsByParams(params=params)
    print("Saving checkpoints to '{ckpt_path}', you may visualize in tensorboard with the following: \n\t`tensorboard --logdir={ckpt_path}`".format(
        ckpt_path=ckpt_path))
    banet_pth_path = os.path.join(ckpt_path, 'msr-vtt_banet.pth')
    best_banet_pth_path = os.path.join(ckpt_path, 'msr-vtt_best_banet.pth')
    optimizer_pth_path = os.path.join(ckpt_path, 'msr-vtt_optimizer.pth')
    best_optimizer_pth_path = os.path.join(ckpt_path, 'msr-vtt_best_optimizer.pth')

    # Prepare dataset paths.
    video_root = './datasets/MSR-VTT/Video/'
    anno_json_path = './datasets/MSR-VTT/datainfo.json'
    video_sort_lambda = lambda x: int(x[5:-4])
    train_range = (0, 6512)
    val_range = (6513, 7009)
    test_range = (7010, 9999)

    feat_dir = 'feats'
    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)

    vocab_pkl_path = os.path.join(feat_dir, 'msr-vtt_vocab.pkl')
    caption_pkl_path = os.path.join(feat_dir, 'msr-vtt_captions.pkl')
    caption_pkl_base = os.path.join(feat_dir, 'msr-vtt_captions')
    train_caption_pkl_path = caption_pkl_base + '_train.pkl'
    val_caption_pkl_path = caption_pkl_base + '_val.pkl'
    test_caption_pkl_path = caption_pkl_base + '_test.pkl'

    feature_h5_path = os.path.join(feat_dir, 'msr-vtt_features.h5')
    feature_h5_feats = 'feats'
    feature_h5_lens = 'lens'

    # Load vocabulary for the particular dataset.
    with open(vocab_pkl_path, 'rb') as fin:
        vocab = pickle.load(fin)
    vocab_size = len(vocab)

    # Load Reference for COCO.
    reference_json_path = '{0}.json'.format(test_reference_txt_path)
    reference = COCO(reference_json_path)

    # Initialize the model.
    banet = _models.BANet(a_feature_size, projected_size, mid_size, hidden_size, max_frames, max_words, vocab)

    # Load model weights if possible.
    if os.path.exists(banet_pth_path) and use_ckpt:
        banet.load_state_dict(torch.load(banet_pth_path))
    if use_cuda:
        banet.cuda()

    # Initialize loss and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(banet.parameters(), lr=learning_rate)
    if os.path.exists(optimizer_pth_path) and use_ckpt:
        optimizer.load_state_dict(torch.load(optimizer_pth_path))

    # Initialize Dataloaders.
    train_loader = _data.get_train_loader(train_caption_pkl_path, feature_h5_path, batch_size)
    eval_loader = _data.get_eval_loader(val_range, feature_h5_path)

    num_train_steps = len(train_loader)
    num_eval_steps = len(eval_loader)

    # Begin Training Loop.
    print("Training Configuration:")
    print("\tLearning Rate: '{0:.4f}'".format(learning_rate))
    print("\tScheduled Sampling:")
    print("\t\tMax Teacher Forcing Rate: '{0:.4f}'".format(max_ss))
    print("\t\tScheduled Factor: '{0:.4f}'".format(ss_factor))
    print("\tBatch Size: '%d'".format(batch_size))
    print("\tEpochs: '%d'".format(num_epochs))
    print("\tDataset: '%s'".format(dataset))
    print("\tCheckpoint Path: '%d'".format(ckpt_path))

    best_meteor = 0
    loss_count = 0
    for epoch in range(num_epochs):
        epsilon = max(0.6, ss_factor / (ss_factor + np.exp(epoch / ss_factor)))
        print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        _tb_logger.log_value('epsilon', epsilon, epoch)

        for i, (videos, targets, cap_lens, video_ids) in enumerate(train_loader, start=1):
            if use_cuda:
                videos = videos.cuda()
                targets = targets.cuda()

            # Zero the gradients and run the encoder-decoder model.
            optimizer.zero_grad()
            outputs, video_encoded = banet(videos, targets, teacher_forcing_ratio=epsilon)

            # NOTE: Usually the last batch is less than the selected batch_size, so we dynamically
            #       compute the correct batch_size to use here, rather than throwing away the last
            #       training batch.
            bsz = len(targets)

            # Un-pad and flatten the outputs and labels.
            outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], dim=0)
            targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], dim=0)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            # Compute loss for back-propagation.
            loss = criterion(outputs, targets)
            # _tb_logger.log_value('loss', loss.data[0], epoch * num_train_steps + i)
            loss_count += loss.data[0]
            loss.backward()
            optimizer.step()

            # Report Training Progress metrics on loss and perplexity.
            if i % 10 == 0 or bsz < batch_size:
                loss_count /= 10 if bsz == batch_size else i % 10
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                      (epoch, num_epochs, i, num_train_steps, loss_count, np.exp(loss_count)))
                loss_count = 0
                tokens = banet.decoder.sample(video_encoded)
                tokens = tokens.data[0].squeeze()
                we = banet.decoder.decode_tokens(tokens)
                gt = banet.decoder.decode_tokens(targets[0].squeeze())
                print('[vid:%d]' % video_ids[0])
                print('WE: %s\nGT: %s' % (we, gt))

        # Finally, compute evaluation metrics and save the best models.
        banet.eval()
        print("Computing Metrics:...")
        metrics = _train.eval_step(eval_loader, banet, test_prediction_txt_path, reference, use_cuda=use_cuda)
        for k, v in metrics.items():
            # _tb_logger.log_value(k, v, epoch)
            print('\t%s: %.6f' % (k, v))
            if k == 'METEOR' and v > best_meteor:
                # Save the best model based on the METEOR metric.
                # For reference, see https://www.cs.cmu.edu/~alavie/papers/BanerjeeLavie2005-final.pdf
                shutil.copy2(banet_pth_path, best_banet_pth_path)
                shutil.copy2(optimizer_pth_path, best_optimizer_pth_path)
                best_meteor = v
        banet.train()

def main():
    global _logger
    args = _cmd.parseArgsForClassOrScript(train)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))
    train(**varsArgs)

if __name__ == '__main__':
    main()
