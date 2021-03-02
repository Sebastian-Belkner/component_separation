#!/usr/local/bin/python
"""
run.py: script for executing main functionality of component_separation

"""

__author__ = "S. Belkner"


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
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks


with open('config.json', "r") as f:
    cf = json.load(f)

LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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

spec_path = cf[mch]['outdir']

indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


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

    if cf['pa']['new_spectrum']:
        data = io.load_plamap(cf['pa'])
        data = preprocess_map(data)
        spectrum = map2spec(data, freqcomb)
        io.save_spectrum(spectrum, spec_path, 'unscaled'+filename)
    else:
        path_name = spec_path + 'spectrum/unscaled' + filename
        spectrum = io.load_spectrum(path_name=path_name)
    if spectrum is None:
        print("couldn't find spectrum with given specifications at {}. Exiting..".format(path_name))
        sys.exit()

    spectrum_scaled = postprocess_spectrum(spectrum, freqcomb)
    io.save_spectrum(spectrum_scaled, spec_path, 'scaled'+filename)

    weights = specsc2weights(spectrum_scaled, cf['pa']["Tscale"])
    io.save_weights(weights, spec_path, 'weights'+cf['pa']["Tscale"]+filename)

    # weighted_spec = spec_weight2weighted_spec(spectrum, weights)