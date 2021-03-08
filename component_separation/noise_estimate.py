#!/usr/local/bin/python
"""
noise_estimate.py: script for estimating noise using planck maps

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
weight_path = cf[mch]['outdir_weight']
indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


def map2spec(maps, freqcomb):
    # tqumap_hpcorrected = tqumap
    if len(maps) == 3:
        spectrum = pw.tqupowerspec(maps, lmax, lmax_mask, freqcomb, specfilter)
    elif len(maps) == 2:
        spectrum = pw.qupowerspec(maps, lmax, lmax_mask, freqcomb, specfilter)
    elif len(maps) == 1:
        print("Only TT spectrum caluclation requested. This is currently not supported.")
    return spectrum


def preprocess_map(data):
    # if data[0] == None:
    #     data = data[1:]
    # elif data[1] == None:
    #     data = [data[0]]
    # data = prep.remove_unseen(data)
    data_prep = data
    for idx, IQU in enumerate(data_prep):
        for key, val in IQU.items():
            data_prep[idx][key]["map"] = prep.replace_undefnan(data_prep[idx][key]["map"])
            data_prep[idx][key]["map"] = prep.subtract_mean(data_prep[idx][key]["map"])
            data_prep[idx][key]["map"] = prep.remove_brightsaturate(data_prep[idx][key]["map"])
            data_prep[idx][key]["map"] = prep.remove_dipole(data_prep[idx][key]["map"])
    return data


def postprocess_spectrum(data, freqcomb):
    spec_sc = pw.apply_scale(data, llp1=llp1)
    if bf:
        beamf = io.load_beamf(freqcomb=freqcomb)
        spec_scbf = pw.apply_beamfunction(spec_sc, beamf, lmax, specfilter)
    else:
        spec_scbf = spec_sc
    return spec_scbf


def difference(data1, data2):
    ret = [None, None, None]
    for idx, IQU in enumerate(data1):
        for key, val in IQU.items():
            ret[idx] = {key: data1[idx][key] - data2[idx][key]}
    return ret


if __name__ == '__main__':
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(40*"$")

    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

    filename = io.make_filenamestring(cf)

    cf['pa']["freqdset"] = "DX12-split1"
    data_hm1 = io.load_plamap(cf['pa'])
    cf['pa']["freqdset"] = "DX12-split2"
    data_hm2 = io.load_plamap(cf['pa'])

    data_diff = difference(data_hm1, data_hm2)

    data_diff = preprocess_map(data_diff)

    spectrum = map2spec(data_diff, freqcomb)
    io.save_data(spectrum, spec_path+'unscaled-hm'+filename)

    spectrum_scaled = postprocess_spectrum(spectrum, freqcomb)
    io.save_data(spectrum_scaled, spec_path+'scaled-hm'+filename)