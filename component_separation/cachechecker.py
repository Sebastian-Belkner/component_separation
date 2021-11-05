import sys
import os

"""
TODO this cachecker could be used to generate metadata which will be stored
to each of the generated maps/spectra. Then, when loading a file, cachechecker
compares if metadata agrees with settings.
"""


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
    info_component = ["N", "F", "S", "T"]
    info_combination = ["non-separated"]
    PLANCKMAPFREQS = ['030', '044', '070', '100', '143', '217', '353', '545', '857']
    misc_type = ["w"]


class Asserter_smica:
    info_component = ["N", "F", "S", "T"]
    info_combination = ["non-separated", "separated", "combined"]
    PLANCKMAPFREQS = ['030', '044', '070', '100', '143', '217', '353', '545', '857']
    misc_type = ['cov', "cov4D", "CMB", "gal_mm", "gal", "w"]