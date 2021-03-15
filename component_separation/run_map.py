#!/usr/local/bin/python
"""
run_map.py: script for generating maps

"""

__author__ = "S. Belkner"


import json
import logging
import logging.handlers
import os
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks

with open('config.json', "r") as f:
    cf = json.load(f)


uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir_spectrum']
map_path = cf[mch]['outdir_map']
indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]


def preprocess_map(data):
    data_prep = data
    for freq, val in data.items():
        print(data_prep[freq].shape)
        data_prep[freq] = prep.replace_undefnan(data_prep[freq])
        data_prep[freq] = prep.subtract_mean(data_prep[freq])
        data_prep[freq] = prep.remove_brightsaturate(data_prep[freq])
        data_prep[freq] = prep.remove_dipole(data_prep[freq])
    return data


def create_difference_map(FREQ):
    def _difference(data1, data2):
        ret = dict()
        for freq, val in data1.items():
            ret.update({freq: (data1[freq] - data2[freq])/2.})
        return ret
    freqfilter =  [
        '030',
        '044',
        '070',
        '100',
        '143',
        '217',
        '353',
        '545',
        '857',
    ]
    freqf = [f for f in freqfilter if f != FREQ]
    cf['pa']["freqfilter"] = freqf
    print(freqf)
    cf['pa']["freqdset"] = "DX12-split1"
    data_hm1 = io.load_plamap_new(cf['pa'])
    cf['pa']["freqdset"] = "DX12-split2"
    data_hm2 = io.load_plamap_new(cf['pa'])

    ret_data = _difference(data_hm1, data_hm2)
    ret_data = preprocess_map(ret_data)

    return ret_data


if __name__ == '__main__':
    print(60*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(60*"$")

    empiric_noisemap = True

    if empiric_noisemap:
        for FREQ in PLANCKMAPFREQ[:-2]:
            data_diff = create_difference_map(FREQ)
            filename = "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full-eohd.fits"\
                .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
                .replace("{freq}", FREQ)\
                .replace("{nside}", str(1024) if int(FREQ)<100 else str(2048))\
                .replace("{00/1}", "00" if int(FREQ)<100 else "01")
            io.save_map(data_diff[FREQ], map_path+filename)
            del data_diff