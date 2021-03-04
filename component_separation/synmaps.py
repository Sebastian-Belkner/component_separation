#!/usr/local/bin/python
"""
synmaps.py: script for generating synthetic maps

"""

import json
import logging
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from functools import reduce
import numpy as np

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks


PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]


uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir']

indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


with open('config.json', "r") as f:
    cf = json.load(f)

if __name__ == '__main__':
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)==int(FREQ))]

    start = 0
    if cf['pa']["run_sim"]:
        for i in range(start, num_sim):
            print("Starting simulation {} of {}.".format(i+1, num_sim))
            path_name = spec_path + 'spectrum/unscaled' + filename
            spectrum = io.load_spectrum(path_name=path_name)

            synmap = spec2synmap(spectrum, freqcomb)
            # io.save_map(synmap, spec_path, "syn/unscaled-"+str(i)+"_synmap-"+filename)

            syn_spectrum = map2spec(synmap, freqcomb)
            io.save_spectrum(syn_spectrum, spec_path, "syn/unscaled-"+str(i)+"_synspec-"+filename)

            syn_spectrum_scaled = spec2specsc(syn_spectrum, freqcomb)
            io.save_spectrum(syn_spectrum_scaled, spec_path, "syn/scaled-"+str(i)+"_synmap-"+filename)
    

        syn_spectrum_avgsc = synmaps2average(filename)
        io.save_spectrum(syn_spectrum_avgsc, spec_path, "syn/scaled-" + "synavg-"+ filename)
        
    # weights = specsc2weights(syn_spectrum_avg, False)
    # io.save_weights(weights, spec_path, "syn/"+"SYNweights"+filename)