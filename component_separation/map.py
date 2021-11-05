import healpy as hp
import numpy as np

from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

from component_separation.cs_util import Helperfunctions as hpf

import component_separation.spherelib.python.spherelib.astro as slhpastro
import component_separation.MSC.MSC.apodize as ap


def create_difference_map(data_hm1, data_hm2):
    def _difference(data1, data2):
        ret = dict()
        for freq, val in data1.items():
            ret.update({freq: (data1[freq] - data2[freq])/2.})
        return ret
    ret_data = _difference(data_hm1, data_hm2)
    return ret_data


def create_mapsyn(spectrum, cf, freqcomb):
    synmap = dict()
    for freqc in freqcomb:
        synmap.update({
            freqc: hp.synfast(
                cls = [
                    spectrum[freqc]["TT"],
                    spectrum[freqc]["EE"],
                    spectrum[freqc]["BB"],
                    spectrum[freqc]["TE"]],
                nside = cf.nside[0] if int(freqc.split("-")[0])<100 else cf.nside[1],
                new=True)})

    #TODO return only numpy
    return np.array([])


def apodize_mask(mask):
    retval = ap.apodize_mask(mask, cache_dir=False)
    return retval


def apodize_masks(masks):
    """
    TODO currently assuming all masks are the same. Perhaps this needs being changed in the future
    """
    key_buff = None
    for key, val in masks.items():
        key_buff = key
        break
    mask_buff = ap.apodize_mask(masks[key_buff], cache_dir=False)
    for freq, val in masks.items():
        masks[freq] = mask_buff

    return masks
    

@log_on_start(INFO, "Starting to process maps")
@log_on_end(DEBUG, "Maps processed successfully: '{result}' ")
def process_all(data):
    """
    Root function. Executes general processing for everydays usage 
    """
    for freq, val in data.items():
        data[freq] = replace_undefnan(data[freq])
        data[freq] = remove_brightsaturate(data[freq])
        data[freq] = subtract_mean(data[freq])
        data[freq] = remove_dipole(data[freq])
    return data


@log_on_start(INFO, "Starting to remove unseen pixels")
@log_on_end(DEBUG, "Unseen pixels removed successfully: '{result}' ")
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


@log_on_start(INFO, "Starting to convert temperature scale")
@log_on_end(DEBUG, "Tempscale converted successfully: '{result}' ")
def tcmb2trj(data: List[Dict], fr, to) -> List[Dict]:
    """Converts maps (which are presumably in K_CMB) to K_RJ scale.
    Args:
        data (Dict): Maps in K_CMB scale
    Returns:
        Dict: Converted maps in K_RJ scale
    """
    for idx, planckmap in enumerate(data):
        for freq, val in planckmap.items():
            factor = slhpastro.convfact(freq=int(freq)*1e9, fr=fr,to=to)
            data[idx][freq]["map"] *= factor
    return data


@log_on_start(INFO, "Starting to calculate conversion factor")
@log_on_end(DEBUG, "Converison factor calculated successfully: '{result}' ")
def tcmb2trj_sc(freq, fr, to) -> List[Dict]:
    """Converts maps (which are presumably in K_CMB) to K_RJ scale.
    Args:
        freq: detector to be scaled
    Returns:
        float: Scaling factor
    """
    factor = slhpastro.convfact(freq=int(freq)*1e9, fr=fr,to=to)
    return factor


@log_on_start(INFO, "Starting to replace undef/nan values")
@log_on_end(DEBUG, "Undef/nan values replaced successfully: '{result}' ")
def replace_undefnan(data):
    treshold = 1e20
    buff = np.where(data==np.nan, 0.0, data)
    buff = np.where(buff < -treshold, 0.0, buff)
    buff = np.where(buff > treshold, 0.0, buff)
    return buff


@log_on_start(INFO, "Starting to remove Bright/saturated pixels")
@log_on_end(DEBUG, "Bright/saturated pixels removed successfully: '{result}' ")
def remove_brightsaturate(data):
    ret = np.zeros_like(data)
    for n in range(data.shape[0]):
        ret[n,:] = np.where(np.abs(data[n,:])>np.mean(data[n,:])+10*np.std(data[n,:]), 0.0, data[n,:])
    return ret


@log_on_start(INFO, "Starting to subtract mean")
@log_on_end(DEBUG, "Mean subtracted successfully: '{result}' ")
def subtract_mean(data):
    return (data.T-np.mean(data, axis=1)).T


@log_on_start(INFO, "Starting to remove monopole and dipole")
@log_on_end(DEBUG, "Monopole and dipole removed successfully: '{result}' ")
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


@hpf.deprecated
def remove_monopole(data):
    "DEPRECATED"
    return hp.remove_monopole(data, fitval=False)