import logging
#%%
import os
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import functools
import healpy as hp
from logdecorator import log_on_end, log_on_error, log_on_start
from component_separation.cs_util import Planckf, Plancks, Planckr
import component_separation.spherelib.python.spherelib.astro as slhpastro


def preprocess_all(data):
    data_prep = data
    for freq, val in data.items():
        print(data_prep[freq].shape)
        data_prep[freq] = replace_undefnan(data_prep[freq])
        data_prep[freq] = subtract_mean(data_prep[freq])
        data_prep[freq] = remove_brightsaturate(data_prep[freq])
        data_prep[freq] = remove_dipole(data_prep[freq])
    return data

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def remove_unseen(tqumap: List[Dict]) -> List[Dict]:
    """Replaces UNSEEN pixels in the polarisation maps (Q, U) with 0.0. This is a quickfix for healpy `_sphtools.pyx`,
    as it throws errors when processing maps with UNSEEN pixels. reason being, one statement is ambigious: `if np.array([True, True])`.

    Args:
        tqumap (Dict): Data as coming from `pw.get_data()`

    Returns:
        Dict: The corrected data
    """
    UNSEEN = -1.6375e30
    UNSEEN_tol = 1.e-2 * 1.6375e30
    def count_bad(m):
        i = 0
        nbad = 0
        size = m.size
        for i in range(m.size):
            if np.abs(m[i] - UNSEEN) < UNSEEN_tol:
                nbad += 1
        return nbad

    def mkmask(m):
        nbad = 0
        size = m.size
        i = 0
        # first, count number of bad pixels, to see if allocating a mask is needed
        nbad = count_bad(m)
        mask = np.ndarray(shape=(1,), dtype=np.int8)
        #cdef np.ndarray[double, ndim=1] m_
        if nbad == 0:
            return False
        else:
            mask = np.zeros(size, dtype = np.int8)
            #m_ = m
            for i in range(size):
                if np.abs(m[i] - UNSEEN) < UNSEEN_tol:
                    mask[i] = 1
        mask.dtype = bool
        return mask
    
    if '100' in tqumap[0].keys():
        # only fixing q and u maps
        if len(tqumap)==2:
            maps = [tqumap[0]["100"]['map'], tqumap[1]["100"]['map']]
        else:
            maps = [tqumap[1]["100"]['map'], tqumap[2]["100"]['map']]
        maps_c = [np.ascontiguousarray(m, dtype=np.float64) for m in maps]

        masks = [np.array([False]) if count_bad(m) == 0 else mkmask(m) for m in maps_c]
        for idx, (m, mask) in enumerate(zip(maps_c, masks)):
            if mask.any():
                m[mask] = 0.0
            if len(tqumap)==2:
                tqumap[idx]["100"]['map'] = m
            else:
                tqumap[idx+1]["100"]['map'] = m
    return tqumap


def tcmb2trj(data: List[Dict]) -> List[Dict]:
    """Converts maps (which are presumably in K_CMB) to K_RJ scale.

    Args:
        data (Dict): Maps in K_CMB scale

    Returns:
        Dict: Converted maps in K_RJ scale
    """
    for idx, planckmap in enumerate(data):
        for freq, val in planckmap.items():
            factor = slhpastro.convfact(freq=int(freq)*1e9, fr=r'K_CMB',to=r'K_RJ')
            data[idx][freq]["map"] *= factor
    return data


def tcmb2trj_sc(freq) -> List[Dict]:
    """Converts maps (which are presumably in K_CMB) to K_RJ scale.

    Args:
        data (Dict): Maps in K_CMB scale

    Returns:
        Dict: Converted maps in K_RJ scale
    """

    factor = slhpastro.convfact(freq=int(freq)*1e9, fr=r'K_CMB',to=r'K_RJ')
    return factor


def replace_undefnan(data):
    treshold = 1e20
    buff = np.where(data==np.nan, 0.0, data)
    buff = np.where(buff < -treshold, 0.0, buff)
    buff = np.where(buff > treshold, 0.0, buff)
    return buff


def remove_brightsaturate(data):
    ret = np.zeros_like(data)
    for n in range(data.shape[0]):
        ret[n,:] = np.where(np.abs(data[n,:])>np.mean(data[n,:])+20*np.std(data[n,:]), 0.0, data[n,:])
    return ret


def subtract_mean(data):
    return (data.T-np.mean(data, axis=1)).T


def remove_dipole(data):
    """Healpy description suggests that this function removes both, the monopole and dipole

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    ret = np.zeros_like(data)
    for n in range(data.shape[0]):
        ret[n,:] = hp.remove_dipole(data[n,:], fitval=False)
    return ret


@deprecated
def remove_monopole(data):
    "DEPRECATED"
    return hp.remove_monopole(data, fitval=False)