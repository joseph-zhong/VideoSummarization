"""
Collection of common utility functions (ex., get weight path)
"""
import os
import glob
import shutil
import logging

from functools import lru_cache

from typing import Any

import numpy as np

DEFAULT_VERBOSITY = 4

def getLogger(name, level=logging.DEBUG, verbosity=DEFAULT_VERBOSITY):
    level = max(level, logging.CRITICAL - 10 * verbosity)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

_logger = getLogger(__file__)
_LOGGING_FORMAT = "[%(asctime)s %(levelname)5s %(filename)s %(funcName)s:%(lineno)s] %(message)s"
logging.basicConfig(format=_LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

def mkdir(path: str) -> None:
    if not os.path.exists(path):
        assert isinstance(path, str), "Expected string path, got {}".format(path)
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise FileExistsError("Cannot create directory '{}', file exists at this location".format(path))


def touch(path: str) -> None:
    if not os.path.exists(path):
        assert isinstance(path, str), "Expected string path, got {}".format(path)
        os.utime(path, None)
    elif not os.path.isdir(path):
        raise IsADirectoryError("Cannot touch file '{}', directory exists at this location".format(path))


def dumpArray(directory: str, name: str, batch: int, arr: np.ndarray, overwrite=False) -> None:
    assert isinstance(directory, str) and os.path.isdir(directory), "Could not find directory: {}".format(directory)

    path = os.path.join(directory, name)
    assert isinstance(name, str), "Invalid dump name: {}".format(name)

    if overwrite:
        _logger.warning("Deleting existing dump {}".format(path))
        shutil.rmtree(path)
    else:
        assert not os.path.exists(path), "Existing dump found for {} in {}, use overwrite".format(name, directory)
    mkdir(path)

    assert isinstance(arr, np.ndarray)
    assert isinstance(batch, int) and batch > 0

    _logger.info("Dumping array {} to {} with batch size {}".format(arr.shape, path, batch))

    num_batches = arr.shape[0] // batch
    for i in range(num_batches):
        _logger.info("\tbatch {}".format(i))
        np.save(os.path.join(path, "{0:04d}".format(i)), arr[i * batch:(i + 1) * batch])

    remainder = arr.shape[0] % (num_batches * batch)
    if remainder:
        _logger.info("\tbatch {}".format(num_batches))
        np.save(os.path.join(path, "{0:04d}".format(num_batches)), arr[-remainder:])


def loadArray(directory: str, name: str) -> np.ndarray:
    assert isinstance(directory, str) and os.path.isdir(directory), "Could not find directory: {}".format(directory)

    path = os.path.join(directory, name)
    assert isinstance(name, str), "Invald dump name: {}".format(name)

    assert os.path.isdir(path), "Existing dump not found for {} in {}".format(name, directory)

    files = sorted(glob.glob(os.path.join(path, "*.npy")))
    assert len(files) > 0, "Could not find any batch files XXXX.npy in {}".format(name, path)

    batches = []
    for file in files:
        batches.append(np.load(file))

    return np.concatenate(batches, axis=0)


def getRawDatasetByName(name: str, mode: str = None) -> str:
    assert isinstance(name, str) and len(name), "Expected dataset name as string got {}".format(name)
    assert mode is None or isinstance(mode, str) and mode == "train" or mode == "test" or mode == "val", \
        "mode must be train, val, or test got {}".format(mode)

    datasets = os.path.join(getWorkspace(), "data", "raw")
    dataset = os.path.join(datasets, name)
    if mode:
        dataset = os.path.join(dataset, mode)

    assert os.path.exists(dataset), \
        "Could not find dataset '{}' in path '{}'".format(name, datasets)

    if not os.path.isdir(dataset):
        raise FileExistsError("File exists at expected dataset location: {}".format(dataset))

    return dataset


def getDatasetByName(name: str, mode: str = None, create=False) -> str:
    assert isinstance(name, str) and len(name), "Expected dataset name as string got {}".format(name)
    assert mode is None or isinstance(mode, str) and mode == "train" or mode == "test" or mode == "val", \
        "mode must be train, val, or test got {}".format(mode)
    datasets = os.path.join(getWorkspace(), "data", "datasets")
    dataset = os.path.join(datasets, name)
    if mode:
        dataset = os.path.join(dataset, mode)

    if create and not os.path.exists(dataset):
        mkdir(dataset)

    assert os.path.exists(dataset), \
        "Could not find dataset '{}' in path '{}'".format(name, datasets)

    if not os.path.isdir(dataset):
        raise FileExistsError("File exists at expected dataset location: {}".format(dataset))

    return dataset


def getWeightsByParams(reuse=False, overwrite=False, **params: Any) -> str:
    """
    Get the weights directory for a model given it's parameters.

    :param reuse: If true and an existing weights directory is found, it will be used without modification, otherwise a
        new directory is created.
    :param overwrite: If true and an existing weights directory is found, it will be emptied and used otherwise a new
        directory is created.
    :param params: Model parameters to use in generating unique nested model path.
    :return: Directory in which to place model weights.
    """
    # Most importantly, we NEED to know which dataset these weights are for. We'll have this be the top-most
    # param in our nested directory structure.
    assert isinstance(params.get("dataset", None), str), "Dataset parameter is required to determine weights path"

    # It is required that parameters are sorted in alphabetical order. Otherwise, two sets of identical parameters could
    # produce two different weights paths. This also ensures all strings are lowered for consistency.
    keys = sorted(k for k in params.keys() if k != "dataset")
    values = (params[key] if not isinstance(params[key], str) else params[key].lower() for key in keys)
    dirs = ("{}={}".format(key.lower(), value) for key, value in zip(keys, values))

    # Build path and ensure exists.
    path = os.path.join(getWorkspace(), "data", "weights", params["dataset"].lower(), *dirs)
    mkdir(path)

    # If overwrite is not specified, return the most recent weights folder otherwise create a new one.
    previous = sorted(glob.glob(os.path.join(path, "*")))
    previous = int(os.path.basename(previous[-1])) if previous else -1

    if overwrite or reuse:
        new = os.path.join(path, "{:04d}".format(max(previous, 0)))
        if previous != -1 and not reuse:
            # Must empty the existing folder
            shutil.rmtree(new)
        mkdir(new)
        return new
    else:
        new = os.path.join(path, "{:04d}".format(previous + 1))
        mkdir(new)
        return new


@lru_cache(maxsize=1)
def getWorkspace() -> str:
    assert 'VS_WORKSPACE' in os.environ, "ENV variable 'VS_WORKSPACE' must be set to the repository root"
    return os.environ['VS_WORKSPACE']
