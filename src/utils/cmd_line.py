#!/usr/bin/env python
"""
cmd_line.py
---

CMD Line parsing utilities.

"""
import argparse
import inspect
from types import GeneratorType

import src.utils.utility as _util

_logger = _util.getLogger(__file__)


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def add_boolean_argument(parser, name, default=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name,
        nargs='?',
        default=default,
        const=True,
        type=_str_to_bool)
    group.add_argument('--no' + name,
        dest=name,
        action='store_false')


def parseArgsForClassOrScript(fn):
    assert inspect.isfunction(fn) or inspect.ismethod(fn)

    spec = inspect.getargspec(fn)

    parser = argparse.ArgumentParser()
    for i, arg in enumerate(spec.args):
        if arg == 'self' or arg == 'logger':
            continue

        # If index is greater than the last var with a default, it's required.
        numReq = len(spec.args) - len(spec.defaults)
        required = i < numReq
        default = spec.defaults[i - numReq] if not required else None
        # By default, args are parsed as strings if not otherwise specified.
        if isinstance(default, bool):
            # REVIEW josephz: This currently has a serious flaw in that clients may only set positive boolean flags.
            parser.add_argument("--" + arg, default=default, action='store_true')
        elif isinstance(default, (tuple, list, GeneratorType)):
            parser.add_argument("--" + arg, default=default, nargs="+", help="Tuple of " + arg, required=False)
        else:
            parser.add_argument("--" + arg, default=default, type=type(default) if default is not None else str)

    parser.add_argument("-v", "--verbosity",
        default=_util.DEFAULT_VERBOSITY,
        type=int,
        help="Verbosity mode. Default is 4. "
                 "Set as "
                 "0 for CRITICAL level logs only. "
                 "1 for ERROR and above level logs "
                 "2 for WARNING and above level logs "
                 "3 for INFO and above level logs "
                 "4 for DEBUG and above level logs")
    argv = parser.parse_args()
    argsToVals = vars(argv)

    if argv.verbosity > 0 or argv.help:
        docstr = inspect.getdoc(fn)
        assert docstr is not None, "Please write documentation :)"
        print()
        print(docstr.strip())
        print()
        print("Arguments and corresponding default or set values")
        for arg in spec.args:
            if arg == 'self' or arg == 'logger' or arg not in argsToVals:
                continue
            print("\t{}={}".format(arg, argsToVals[arg] if argsToVals[arg] is not None else ""))
        print()

    return argv
