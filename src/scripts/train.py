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
import tqdm

import src.data.msrvtt as _data
import src.model.models as _models
import src.train.train_test_utils as _train

import src.utils.utility as _util
import src.utils.cmd_line as _cmd
from src.data.caption import Vocabulary, Token, vocab

from extern.coco_caption.pycocotools.coco import COCO

_logger = _util.get_logger(__file__)

def train(
    # General training hyperparameters.
    dataset: str,
    num_epochs: int=100,
    batch_size: int=128,

    # Learning rate schedulers.
    learning_rate: float=3e-4,
    ss_factor: int=24,
    min_ss: float=0.6,

    # Representation hyperparameters.
    projected_size: int=500,
    hidden_size: int=1024,  # Hidden size of the recurrent cells.
    mid_size: int=128,  # Dimension of the boundary detection layer.

    # REVIEW josephz: Remove this?
    # frame_shape: tuple=(3, 224, 224),  # Video frame shape.
    a_feature_size: int=2048,  # Appearance model feature-dimension size.
    # REVIEW josephz: Remove this?
    # m_feature_size=4096,  # Motion model feature-dimension size.

    # Maximum-size hyperparameters.
    # frame_sample_rate: int=10,  # Sample rate of video frames.
    max_frames: int=30,  # Maximum length of the video-frame sequence.
    max_words: int=30,  # Maximum length of the caption-word sequence.

    # Misc hyperparameters.
    ckpt_freq: int=3,
    use_cuda: bool=False,
    use_ckpt: bool=False,
    seed: int=0,
):
    """

    Args:
        dataset (str): Dataset to train on.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size to train with.

        learning_rate (float): Learning rate.
        ss_factor (int): Scheduled Sampling factor, to compute a teacher-forcing ratio.
        min_ss (float): Minimum teacher-forcing ratio.

        projected_size (int): Projection size for the Encoder-Decoder model.
        hidden_size (int): Hidden state size for the recurrent network in the encoder.
        mid_size (int): Hidden state size for the Boundary Detector network in the encoder.
        a_feature_size: Input feature size for the Encoder network.

        max_frames (int): Maximum length of the video-frame sequence.
        max_words (int): Maximum length of the caption-word sequence.

        ckpt_freq (int): Frequency to compute evaluation metrics and save checkpoint.
        use_cuda (bool): Flag whether to use CUDA devices.
        use_ckpt (bool): Flag on whether to load checkpoint if possible.
        seed (int): Random seed.

    Effects:
        We will have several outputs:
            - Checkpoints (model weights)
            - Logs (tensorboard logs)
    """
    # Set seeds.
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    # Prepare output paths.
    # REVIEW josephz: This is unbelievably hacky, but we want an easy way to allow the user to set and track
    #   hyperparameters using the cmd_line interface? This should probably be abstracted in utility.py.
    hparams = locals()
    params = {arg_name: hparams[arg_name] for arg_name in inspect.signature(train).parameters.keys()}

    ckpt_path = _util.get_weights_path_by_param(reuse=False, **params)
    print("Saving checkpoints to '{ckpt_path}', you may visualize in tensorboard with the following: \n\n\t`tensorboard --logdir={ckpt_path}`\n".format(
        ckpt_path=ckpt_path))

    # Setup logging paths.
    log_path = os.path.join(ckpt_path, 'logs')
    _util.mkdir(log_path)
    _tb_logger.configure(log_path, flush_secs=10)

    # REVIEW josephz: Todo, clean this up.
    banet_pth_path_fmt = os.path.join(ckpt_path, '{:04d}_{:04d}.pth')
    best_banet_pth_path = os.path.join(ckpt_path, 'weights.pth')
    optimizer_pth_path = os.path.join(ckpt_path, 'optimizer.pth')
    best_optimizer_pth_path = os.path.join(ckpt_path, 'best_optimizer.pth')

    # Load Vocabulary.
    vocab_size = len(vocab())

    # Load Reference for COCO.
    # val_dir = _util.get_dataset_by_name(dataset, mode='val')
    # val_reference_txt_path = os.path.join(val_dir, 'reference.json')
    # val_prediction_txt_path = os.path.join(val_dir, 'prediction.txt')
    # reference = COCO(val_reference_txt_path)

    eval_mode = 'val'
    eval_dir = _util.get_dataset_by_name(dataset, mode=eval_mode)
    test_reference_txt_path = os.path.join(eval_dir, 'reference.json')
    test_prediction_txt_path = os.path.join(eval_dir, 'prediction.txt')
    reference = COCO(test_reference_txt_path)
    print("Evaluating on '{}'".format(eval_dir))

    # Initialize the model.
    banet = _models.BANet(a_feature_size, projected_size, mid_size, hidden_size, max_frames, max_words, use_cuda=use_cuda)

    # Load model weights if possible.
    if use_ckpt:
        pretrained_path = os.path.join(_util.get_raw_dataset_by_name('MSRVTT'), 'pretrained_weights.pth')
        weights = torch.load(pretrained_path)

        # REVIEW josephz: Figure out how to do the decoder weights partially:
        #   https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/6
        del weights['decoder.word_embed.weight']
        del weights['decoder.word_restore.bias']
        del weights['decoder.word_restore.weight']
        banet.load_state_dict(weights, strict=False)
    if use_cuda:
        banet.cuda()

    # Initialize loss and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(banet.parameters(), lr=learning_rate)
    if os.path.exists(optimizer_pth_path) and use_ckpt:
        optimizer.load_state_dict(torch.load(optimizer_pth_path))

    # Initialize Dataloaders.
    train_loader = _data.get_train_dataloader(dataset, batch_size=batch_size)
    eval_loader = _data.get_eval_dataloader(dataset, eval_mode, batch_size=batch_size)

    num_train_steps = len(train_loader)
    num_eval_steps = len(eval_loader)

    # Begin Training Loop.
    print("Training Configuration:")
    print("\tLearning Rate: '{0:.4f}'".format(learning_rate))
    print("\tScheduled Sampling:")
    print("\t\tMax Teacher Forcing Rate: '{0:.4f}'".format(min_ss))
    print("\t\tScheduled Factor: '{0:.4f}'".format(ss_factor))
    print("\tBatch Size: '{}'".format(batch_size))
    print("\tEpochs: '{}'".format(num_epochs))
    print("\tDataset: '{}'".format(dataset))
    print("\tCheckpoint Path: '{}'".format(ckpt_path))

    best_meteor = 0
    loss_count = 0
    for epoch in range(num_epochs):
        epsilon = max(min_ss, ss_factor / (ss_factor + np.exp(epoch / ss_factor)))
        print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        _tb_logger.log_value('epsilon', epsilon, epoch)

        for i, (videos, captions, cap_lens, video_ids) in tqdm.tqdm(enumerate(train_loader, start=1), total=num_train_steps):
            if use_cuda:
                videos = videos.cuda()
                targets = captions.cuda()
            else:
                targets = captions

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
            # assert all(targets > 0) and all(outputs > 0)
            loss = criterion(outputs, targets)
            loss_val = loss.item()
            _tb_logger.log_value('loss', loss_val, epoch * num_train_steps + i)
            loss_count += loss_val
            # REVIEW josephz: Is there grad_norm?
            loss.backward()
            optimizer.step()

            eval_steps = 25
            if i % eval_steps == 0 or bsz < batch_size:
                loss_count /= eval_steps if bsz == batch_size else i % eval_steps
                perplexity = np.exp(loss_count)
                print('Epoch [%d/%d]:\n\tStep [%d/%d]\n\tLoss: %.4f\n\tPerplexity: %5.4f' %
                      (epoch, num_epochs, i, num_train_steps, loss_count, perplexity))
                _tb_logger.log_value('perplexity', perplexity, epoch * num_train_steps + i)
                loss_count = 0
                tokens = banet.decoder.sample(video_encoded)
                for j in range(5):
                    we = vocab().decode(tokens.data[j].squeeze())
                    gt = vocab().decode(captions[j].squeeze())
                    print('\t\t[vid_id={}]'.format(video_ids[j]))
                    print('\t\t\tWE: %s\n\t\t\tGT: %s' % (we, gt))

        # Finally, compute evaluation metrics and save the best models.
        if epoch % ckpt_freq == 0:
            # Save epoch checkpoint.
            banet_pth_path = banet_pth_path_fmt.format(epoch, num_epochs)
            print("Saving checkpoints to '{}'".format(banet_pth_path))
            torch.save(banet.state_dict(), banet_pth_path)
            torch.save(optimizer.state_dict(), optimizer_pth_path)

            # Compute evaluation.
            banet.eval()
            print("Computing Metrics:...")
            metrics = _train.eval_step(eval_loader, banet, test_prediction_txt_path, reference, use_cuda=use_cuda)
            for k, v in metrics.items():
                _tb_logger.log_value(k, v, epoch)
                if k == 'METEOR' and v > best_meteor:
                    # Save the best model based on the METEOR metric.
                    # For reference, see https://www.cs.cmu.edu/~alavie/papers/BanerjeeLavie2005-final.pdf
                    print("Saving best checkpoint of metric: '{}'".format(best_meteor))
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
