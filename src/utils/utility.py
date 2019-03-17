"""
Collection of common utility functions (ex., get weight path)
"""
import os
import glob
import shutil
import logging

from functools import lru_cache

from typing import Any

_logger = logging.getLogger(__file__)
_LOGGING_FORMAT = "[%(asctime)s %(levelname)5s %(filename)s %(funcName)s:%(lineno)s] %(message)s"
logging.basicConfig(format=_LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

DEFAULT_VERBOSITY = 4

def getLogger(name, level=logging.DEBUG, verbosity=DEFAULT_VERBOSITY):
    level = max(level, logging.CRITICAL - 10 * verbosity)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


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


def getDatasetByName(name: str) -> str:
    datasets = os.path.join(getWorkspace(), "data", "datasets")
    dataset = os.path.join(datasets, name)

    assert os.path.exists(dataset), \
        "Could not find dataset '{}' in path '{}'".format(name, datasets)

    if not os.path.isdir(dataset):
        raise FileExistsError("File exists at expected dataset location: {}".format(dataset))

    return dataset


def getWeightsByParams(overwrite=False, **params: Any) -> str:
    """
    Get the weights directory for a model given it's parameters.

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

    if overwrite:
        new = os.path.join(path, "{:04d}".format(max(previous, 0)))
        if previous != -1:
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
    assert 'VS_WORKSPACE' in os.environ
    return os.environ['VS_WORKSPACE']
