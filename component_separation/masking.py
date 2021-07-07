#!/usr/local/bin/python
"""
synmaps.py: script for generating masks. Not yet used but eventually needed
"""
import json
import logging
import logging.handlers
import os
import platform
import sys
import numpy as np
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks
import component_separation.preprocess as prep
import component_separation
compath = os.path.dirname(component_separation.__file__)[:-21]

with open('{}/config.json'.format(compath), "r") as f:
    cf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"
mask_path = cf[mch]['outdir']



def hitsvar2mask(hitsmap, tresh_low, tresh_up):
    """Generates a map mask based on the hits-count of the planck scanning strategy. It uses the histogram of the hitscount to derive a sky-area, which
    has the same noise level.

    Args:
        hitsmap (np.ndarray): [description]
        tresh_low: float): the lower limit of the noise variance. E.g. :math:`tresh_low=0` returns the complete map,
                            whereas :math:`upper_noise=0.5` returns the areas which have noise levels up to 50% of the minimum
                            noise level.
        tresh_up (float): the upper limit of the noise variance. E.g. :math:`tresh_up=1` returns the complete map,
                            whereas :math:`tresh_up=0.5` returns the areas which have noise levels up to 50% of the maximum
                            noise level.
    """

    
    mask = dict()
    for FREQ, mp in hitsmap.items():
        # mp_prep = preprocess_map(mp)
        mean = np.mean(mp)
        std = np.std(mp)
        a = np.where(np.abs(mp)>mean+20*std, 0.0, mp)
        print(FREQ)
        mn, mx = np.min(a), np.max(a)
        print(mn, mx)
        # hist, bin_edges = np.histogram(mp, bins=100)#, range=(tresh_low, tresh_up))
        noise_low = mn + (mx - mn) * tresh_low
        noise_up = mn + (mx - mn) * tresh_up
        print(noise_low, noise_up)
        print(len(a[a<=noise_up]))
        mask.update({FREQ: np.where((a<=noise_up)&(a>=noise_low), True, False)})
    return mask


def buildandsave_masks():
    
    # load
    tresh_low, tresh_up = 0.0, 0.5
    freqdset = cf['pa']['freqdset'] # DX12 or NERSC
    hitsvar = io.load_hitsvar(cf["pa"])

    # build
    masks = hitsvar2mask(hitsvar, tresh_low, tresh_up)

    # store
    for FREQ, mask in masks.items():
        fname = '{freqdset}-freq_{freq}-{tresh_low}to{tresh_up}-split_{split}.hitshist.npy'.format(
            freqdset = freqdset,
            freq = FREQ,
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"],
            tresh_low = tresh_low,
            tresh_up = tresh_up)
        path_name = mask_path + 'mask/hitsvar/'+ fname
        io.save_mask(mask, path_name)


if __name__ == '__main__':
    buildandsave_masks()