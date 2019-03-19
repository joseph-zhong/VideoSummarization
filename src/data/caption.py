"""
caption.py
---

Utilities for processing video captions.
"""

import json
import os
import pickle
from collections import Sequence
from enum import Enum
from functools import lru_cache
from typing import Union, List, Iterable

import nltk
import numpy as np
from tqdm import tqdm

import src.utils.cmd_line as _cmd
import src.utils.utility as _util
import src.utils.coco as _coco


class Token(Enum):
    """
    Simple Token enum for abstracting common meta tokens.
    """
    START = "<start>"
    END = "<end>"
    PAD = "<pad>"
    UNK = "<unk>"

    def __str__(self):
        return self.value


def vocab(dataset="MSRVTT"):
    if vocab.inst is None:
        if dataset is None:
            vocab.inst = Vocabulary()
        else:
            dataset_dir = _util.get_dataset_by_name(dataset)
            with open(os.path.join(dataset_dir, Vocabulary.PICKLE_FILE), 'rb') as f:
                vocab.inst = pickle.load(f)
            if not isinstance(vocab.inst, Vocabulary):
                raise TypeError('Pickled object @ {} not of type {}'.format(dataset_dir, Vocabulary))
    return vocab.inst
vocab.inst = None


class Vocabulary:
    PICKLE_FILE = "vocab.pkl"
    """
    Represents an NLP vocabulary. Given all words in a corpus, it will identify all unique words and
    will return a word's unique index on __getitem__. This is a singleton class.
    """
    def __init__(self, threshold: int = 3):
        """
        Creates a Vocabulary with the given threshold.
        :param threshold: Number of occurrences under which words will be ignored.
        """
        assert isinstance(threshold, int) and threshold > 0, "Unexpected vocabulary threshold: {}".format(threshold)

        self._threshold = threshold
        self._word2idx = {}
        self._word2count = {}
        self._idx2word = []
        self._nwords = 0

        self.add(Token.PAD, threshold)
        self.add(Token.START, threshold)
        self.add(Token.END, threshold)
        self.add(Token.UNK, threshold)

    def add(self, word: Union[str, Token], inc: int = 1) -> None:
        """
        Add a word to this vocabulary if it does not already exist.
        :param word: Word as a string or Token.
        :param inc: Number of times this word has appeared since last call.
        """
        assert isinstance(word, (str, Token))
        if isinstance(word, str):
            word = word.lower()

        count = self._word2count.get(word, 0)
        self._word2count[word] = count + inc

        if count < self._threshold <= count + inc:
            self._word2idx[word] = len(self._idx2word)
            self._idx2word.append(word)
            self._nwords += 1

    @lru_cache(maxsize=256)
    def __getitem__(self, word: Union[str, Token]) -> str:
        """
        Retrieve a word's index in this vocabulary.
        :param word: Word as a string or Token.
        """
        assert isinstance(word, (str, Token))
        if isinstance(word, str):
            word = word.lower()

        if word not in self._word2idx or self._word2count[word] < self._threshold:
            return self._word2idx[Token.UNK]
        return self._word2idx[word]

    def __len__(self) -> int:
        """
        :return: Number of tokens in this vocabulary excluding unk'd tokens.
        """
        return self._nwords

    @property
    def num_tokens(self) -> int:
        """
        :return: Number of tokens in this vocabulary including unk'd tokens.
        """
        return len(self._word2count)

    def decode(self, caption: Iterable[int]) -> str:
        """
        Decodes an iterable of token indices into a human-readable string.
        :param caption: iterable of token indices.
        :return: human-readable string form.
        """
        assert isinstance(caption, Iterable)

        # str conversion required as some tokens are Tokens, not str.
        return " ".join(str(self._idx2word[int(i)]) for i in caption)


def build_vocabulary(captions, threshold):
    assert isinstance(captions, Sequence) and all(isinstance(c, str) for c in captions)

    _logger.info("Building vocabulary from {} captions with unk threshold {}".format(len(captions), threshold))

    vocab = Vocabulary(threshold)
    for caption in tqdm(captions):
        tokens = nltk.tokenize.word_tokenize(caption)

        for token in tokens:
            vocab.add(token)

    _logger.info("Built vocabulary with {} tokens from {} unique tokens ({:.2f}% reduction)"
                 .format(len(vocab), vocab.num_tokens, 100 - 100. * len(vocab) / vocab.num_tokens))
    return vocab


def _build_cache(dataset: str, mode: str, sentences: List[str], video_ids: List[str], vocab: Vocabulary, max_words: int,
                 overwrite: bool) -> None:
    dataset_dir = _util.get_dataset_by_name(dataset, mode, create=True)

    captions = []
    cap_lens = []
    _logger.info("Building {} cache...".format(mode))
    for i, sentence in enumerate(tqdm(sentences)):
        caption = [vocab[Token.START]]

        caption += map(vocab.__getitem__, map(str.lower, nltk.tokenize.word_tokenize(sentence)))
        cap_lens.append(len(caption) + 1)  # plus one for Token.END

        if len(caption) >= max_words:
            _logger.warn("Truncating caption {} from {} words to {}".format(i, len(caption) + 1, max_words))
            caption = caption[:max_words - 1]
        caption.append(vocab[Token.END])

        caption += [vocab[Token.PAD]] * (max_words - len(caption))

        assert len(caption) == max_words
        captions.append(caption)

    captions = np.array(captions)
    cap_lens = np.array(cap_lens)
    video_ids = np.array(video_ids)

    _logger.info("Saving cache...")
    _util.dump_array(dataset_dir, "captions", 10000, captions, overwrite=overwrite)
    _util.dump_array(dataset_dir, "cap_lens", 100000, cap_lens, overwrite=overwrite)
    _util.dump_array(dataset_dir, "video_ids", 100000, video_ids, overwrite=overwrite)


def build_cache(raw: str, dataset: str, threshold: int, max_words: int, train_fraction: float = 1., overwrite: bool = False) -> None:
    """
    Builds caption cache files for a raw dataset based on annotations.json.

    :param raw: Raw dataset of videos for which to build cache.
    :param dataset: Dataset in which to place the resultant cache files.
    :param threshold: Number of occurrences under which a token will be unk'd.
    :param max_words: Maximum number of allowable words in a caption (will pad to this length).
    :param train_fraction: Amount of training annotations to use, defaults to 1.0 (all of train).
    :param overwrite: Unless this flag is specified, will fail rather than overwrite existing cache.
    """
    assert isinstance(raw, str)
    assert isinstance(dataset, str)
    assert isinstance(train_fraction, float) and 0 < train_fraction <= 1.
    assert isinstance(max_words, int) and max_words > 0, "max_words must be a positive integer"

    train_dir = _util.get_raw_dataset_by_name(raw, mode="train")
    val_dir = _util.get_raw_dataset_by_name(raw, mode="val")
    test_dir = _util.get_raw_dataset_by_name(raw, mode="test")

    train_ann = os.path.join(train_dir, "annotations.json")
    val_ann = os.path.join(val_dir, "annotations.json")
    test_ann = os.path.join(test_dir, "annotations.json")

    assert os.path.exists(train_ann), "Could not find train annotations.json in raw dataset {}".format(raw)
    assert os.path.exists(val_ann), "Could not find val annotations.json in raw dataset {}".format(raw)
    assert os.path.exists(test_ann), "Could not find test annotations.json in raw dataset {}".format(raw)

    with open(train_ann, 'r') as f:
        train_ann = json.load(f)
    with open(val_ann, 'r') as f:
        val_ann = json.load(f)
    with open(test_ann, 'r') as f:
        test_ann = json.load(f)

    train_sentences = []
    train_video_ids = []
    for sentence in train_ann["sentences"]:
        train_video_ids.append(sentence["video_id"])
        train_sentences.append(sentence["caption"])

    val_sentences = []
    val_video_ids = []
    for sentence in val_ann["sentences"]:
        val_video_ids.append(sentence["video_id"])
        val_sentences.append(sentence["caption"])

    test_sentences = []
    test_video_ids = []
    for sentence in test_ann["sentences"]:
        test_video_ids.append(sentence["video_id"])
        test_sentences.append(sentence["caption"])

    all_sentences = train_sentences + val_sentences + test_sentences
    vocab = build_vocabulary(all_sentences, threshold)

    last_train_idx = int(len(train_ann["sentences"]) * train_fraction)
    _logger.info("Using {} of {} total train sentences".format(last_train_idx, len(train_ann["sentences"])))

    _build_cache(dataset, "train", train_sentences[:last_train_idx], train_video_ids[:last_train_idx], vocab, max_words, overwrite)
    _build_cache(dataset, "val", val_sentences, val_video_ids, vocab, max_words, overwrite)
    _build_cache(dataset, "test", test_sentences, test_video_ids, vocab, max_words, overwrite)

    with open(os.path.join(_util.get_dataset_by_name(dataset), Vocabulary.PICKLE_FILE), 'wb') as f:
        pickle.dump(vocab, f)

    prepare_gt(os.path.join(_util.get_dataset_by_name(dataset, mode="val"), "reference.json"), val_sentences, val_video_ids)
    prepare_gt(os.path.join(_util.get_dataset_by_name(dataset, mode="test"), "reference.json"), test_sentences, test_video_ids)


def prepare_gt(path, sentences, video_ids):
    _logger.info("Outputting ground truth to {}".format(path))
    with open(path, 'w') as f_out:
        for sentence, video_id in zip(sentences, video_ids):
            video_id = int(video_id[5:])
            ground_truth = "{}\t{}\n".format(video_id, sentence)
            f_out.write(ground_truth)

    _coco.create_reference_json(path)

def main():
    global _logger
    args = _cmd.parseArgsForClassOrScript(build_cache)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger = _util.get_logger(__file__, verbosity=verbosity)
    _logger.info("Passed arguments: '{}'".format(varsArgs))
    build_cache(**varsArgs)


if __name__ == '__main__':
    main()

