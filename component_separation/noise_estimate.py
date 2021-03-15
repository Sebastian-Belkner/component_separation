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
map_path = cf[mch]['outdir_map']
indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


def postprocess_spectrum(data, freqcomb):
    spec_sc = pw.apply_scale(data, llp1=llp1)
    if bf:
        beamf = io.load_beamf(freqcomb=freqcomb)
        spec_scbf = pw.apply_beamfunction(spec_sc, beamf, lmax, specfilter)
    else:
        spec_scbf = spec_sc
    return spec_scbf





def map2spec(freqcomb, filename):
    data_diff = io.load_plamap(cf['pa'])
    if len(data_diff) == 3:
        spectrum = pw.tqupowerspec(data_diff, lmax, lmax_mask, freqcomb, specfilter)
    elif len(data_diff) == 2:
        spectrum = pw.qupowerspec(data_diff, lmax, lmax_mask, freqcomb, specfilter)
    elif len(data_diff) == 1:
        print("Only TT spectrum caluclation requested. This is currently not supported.")
    io.save_data(spectrum, spec_path+'unscaled'+filename)
    return spectrum


def spec2specsc(freqcomb, filename):
    spectrum = io.load_spectrum(spec_path+'unscaled'+filename)
    spectrum_scaled = postprocess_spectrum(spectrum, freqcomb)
    io.save_data(spectrum_scaled, spec_path+'scaled'+filename)


if __name__ == '__main__':
    print(60*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(60*"$")
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    filename = io.make_filenamestring(cf)


    
    # map2spec(freqcomb, filename)

    spec2specsc(freqcomb, filename)