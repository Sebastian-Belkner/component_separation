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