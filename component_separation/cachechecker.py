import sys
import os
from enum import Enum


def alert_cached(func):
    def wrapped(*args):
        if os.path.isfile(args[0]):
            print('Output file {} already exists. Overwrite settings are set to {}'.format(args[0], args[1]))
            if args[1]:
                print('Overwriting cache')
            else:
                print('Exiting..')
                sys.exit()
        return func(*args)
    return wrapped


def iff_make_dir(outpath_name):
    if os.path.exists(outpath_name):
        pass
    else:
        os.makedirs(outpath_name)


class Asserter:
    info_component = ["noise", "foreground", "signal", "total"]
    info_combination = ["non-separated"]
    PLANCKMAPFREQS = ['030', '044', '070', '100', '143', '217', '353', '545', '857']


class Asserter_smica:
    info_component = ["noise", "foreground", "signal", "total"]
    info_combination = ["non-separated", "separated"]
    PLANCKMAPFREQS = ['030', '044', '070', '100', '143', '217', '353', '545', '857']