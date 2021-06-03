#!/usr/local/bin/python
"""
synmaps.py: script for generating synthetic maps from spectra

"""

import json
import logging
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

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

import component_separation
with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

num_sim = cf['pa']["num_sim"]
lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


if __name__ == '__main__':
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)==int(FREQ))]

    start = 0
    C_ltot = io.load_data(path_name=io.spec_sc_path_name)
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    for i in range(start, num_sim):
        print("Starting simulation {} of {}.".format(i+1, num_sim))
        syn_map = pw.create_synmap(C_ltot, cf, mch, freqcomb, specfilter) 
        io.save_data(syn_map, io.synmap_sc_path_name)

        #     syn_spectrum = pw.tqupowerspec(syn_map, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
        #     io.save_data(syn_spectrum, io.out_spec_pathspec_path, "syn/unscaled-"+str(i)+"_synspec-"+filename)

        #     syn_spectrum_scaled = spec2specsc(syn_spectrum, freqcomb)
        #     io.save_spectrum(syn_spectrum_scaled, spec_path, "syn/scaled-"+str(i)+"_synmap-"+filename)
    

        # syn_spectrum_avgsc = synmaps2average(filename)
        # io.save_spectrum(syn_spectrum_avgsc, spec_path, "syn/scaled-" + "synavg-"+ filename)
        
    # weights = specsc2weights(syn_spectrum_avg, False)
    # io.save_weights(weights, spec_path, "syn/"+"SYNweights"+filename)