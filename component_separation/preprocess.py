import logging
#%%
import os
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import functools
import healpy as hp
from logdecorator import log_on_end, log_on_error, log_on_start
from component_separation.cs_util import Planckf, Plancks, Planckr


def replace_undefnan(data):
    data = np.where(data==np.nan | data==np.inf  | data==-np.inf, 0.0, data)


def remove_brightsaturate(data):
    mean = np.mean(data)
    std = np.std(data)
    return np.where(np.abs(data)>mean+20*std, 0.0, data)


def subtract_mean(data):
    return data-np.mean(data)


def remove_monopole(data):
    return hp.remove_monopole(data, fitval=True)


def remove_dipole(data):
    """Healpy description suggests that this function removes both, the monopole and dipole

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    return hp.remove_dipole(data, fitval=True)