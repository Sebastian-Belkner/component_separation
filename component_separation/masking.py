#!/usr/local/bin/python
"""
synmaps.py: script for generating masks
"""
import json
import logging
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks


with open('config.json', "r") as f:
    cf = json.load(f)

mask_path = cf[mch]['outdir']

def buildandsave_masks():
    tresh_low, tresh_up = 0.0, 0.1
    freqdset = cf['pa']['freqdset'] # DX12 or NERSC
    
    hitsmap = io.load_hitsmaps(cf["pa"])
    masks = pw.get_mask_hitshist(hitsmap, tresh_low, tresh_up)
    for FREQ, mask in masks.items():
        fname = '{freqdset}-freq_{freq}-{tresh_low}to{tresh_up}-split_{split}.hitshist.npy'.format(
            freqdset = freqdset,
            freq = FREQ,
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"],
            tresh_low = tresh_low,
            tresh_up = tresh_up)
        path_name = mask_path + 'mask/hitscount/'+ fname
        io.save_mask(mask, path_name)


if __name__ == '__main__':
    buildandsave_masks()