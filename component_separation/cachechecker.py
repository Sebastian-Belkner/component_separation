import sys
import os


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