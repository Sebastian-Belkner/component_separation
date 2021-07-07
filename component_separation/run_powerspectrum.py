"""
run_powerspectrum.py: script for executing main functionality of component_separation

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

import numpy as np
from healpy.sphtfunc import smoothing

import component_separation
import component_separation.io as io
import component_separation.powspec as pw
import component_separation.transform_map as trsf_m
import component_separation.transform_spec as trsf_s
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Planckf, Plancks

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

# LOGFILE = 'data/tmp/logging/messages.log'
# logger = logging.getLogger("")
# handler = logging.handlers.RotatingFileHandler(
#         LOGFILE, maxBytes=(1048576*5), backupCount=0
# )
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]
nside_out = cf['pa']["nside_out"]


def set_logger(loglevel=logging.INFO):
    logger.setLevel(loglevel)
    logging.StreamHandler(sys.stdout)


def map2spec(data, tmask, pmask):
    # tqumap_hpcorrected = tqumap
    # if len(data) == 3:
    #     spectrum = pw.tqupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # elif len(data) == 2:
    #     spectrum = pw.qupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # elif len(data) == 1:
    #     print("Only TT spectrum calculation requested. This is currently not supported.")
    spectrum = pw.tqupowerspec(data, tmask, pmask, lmax, lmax_mask)
    return spectrum


def specsc2weights(spectrum, Tscale):
    cov = pw.build_covmatrices(spectrum, lmax, freqfilter, specfilter, Tscale)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter, Tscale)
    return weights


def spec_weight2weighted_spec(spectrum, weights):
    alms = pw.spec2alms(spectrum)
    alms_w = pw.alms2almsxweight(alms, weights)
    spec = pw.alms2cls(alms_w)
    return spec


if __name__ == '__main__':
    filename_raw = io.total_filename_raw
    filename = io.total_filename
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated raw filename(s) for this session: {}".format(filename_raw))
    print("Generated filename(s) for this session: {}".format(filename))
    print(filename)
    print(40*"$")
    # set_logger(DEBUG)
    run_map2spec = True
    run_alm2spec = True

    if run_map2spec:
        # run this for noise data, and full data. check if files exist
        for data_desc in ['noise', 'full', 'signal']:
            prepare_cf(data_desc)
            maps = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
            maps = trsf_m.process_all(maps)
            tmask, pmask, pmask = io.load_one_mask_forallfreq()

            C_ltot_unsc = map2spec(maps, tmask, pmask)
            io.save_data(C_ltot_unsc, io.spec_unsc_path_name)
            C_ltot = trsf_s.process_all(C_ltot_unsc, csu.freqcomb, cf['pa']['smoothing_window'], cf['pa']['max_polynom'])
            io.save_data(C_ltot, io.spec_sc_path_name)

        # spectrum = io.load_data(path_name=io.spec_unsc_path_name)
