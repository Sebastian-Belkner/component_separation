#!/usr/local/bin/python
"""
run_powerspectrum.py: script for executing main functionality of component_separation


"""

#TODO check single frequency transferfunctions

__author__ = "S. Belkner"

import os
os.environ["OMP_NUM_THREADS"] = "8"
import json
import logging
from astropy.io import fits
from scipy import interpolate
import logging.handlers
import component_separation.interface as cslib

import numpy as np

import platform
import healpy as hp
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple
import component_separation.MSC.MSC.pospace as ps
import pandas as pd
import smica

import component_separation
import matplotlib.pyplot as plt
import component_separation.io as io
import component_separation.powspec as pw
import component_separation.transform_map as trsf_m
from component_separation.cs_util import Config
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf

with open(os.path.dirname(component_separation.__file__)+'/config_ps.json', "r") as f:
    cf = json.load(f)
csu = Config(cf)

# LOGFILE = 'data/tmp/logging/messages.log'
# logger = logging.getLogger("")
# handler = logging.handlers.RotatingFileHandler(
#         LOGFILE, maxBytes=(1048576*5), backupCount=0
# )
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

filename = io.total_filename

@io.alert_cached(io.weight_path_name)
def run_weight():
    """
    Calculates weights derived from data defined by freqdset attribute of config.
    No SMICA, straightforward weight derivation.
    Needed for combining maps without SMICA.
    """
    def specsc2weights(spectrum):
        print(spectrum.shape)
        cov = pw.build_covmatrices(spectrum, cf['pa']['Tscale'])
        print(cov.shape)
        cov_inv_l = pw.invert_covmatrices(cov)
        print(cov_inv_l.shape)
        weights = pw.calculate_weights(cov_inv_l, cf['pa']['Tscale'])
        return weights
    C_ltot = cslib.load_powerspectra('full')
    weights_tot = specsc2weights(C_ltot)
    print(weights_tot.shape)
    io.save_data(weights_tot, io.weight_path_name)


@io.alert_cached(io.out_misc_path+"tf_{}".format(cf['pa']['binname']) + "_" + filename)
@io.alert_cached(io.out_misc_path+"crosscov_{}".format(cf['pa']['binname']) + "_" + filename)
def run_tf():
    lmax = cf['pa']["lmax"]
    lmax_mask = cf['pa']["lmax_mask"]
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    CMB = dict()
    CMB["TQU"] = dict()
    C_lS_EE = io.load_data(io.signal_sc_path_name)[0,1]
    CMB["TQU"]["in"] = io.load_data(io.map_cmb_sc_path_name)
    CMB["TQU"]['out'] = io.load_data(io.cmbmap_smica_path_name)

    # any mask will do here
    crosscov = ps.map2cl_spin(
        qumap=CMB["TQU"]["in"][1:3],
        spin=2,
        mask=pmask['100'],
        lmax=lmax,
        lmax_mask=lmax_mask,
        qumap2=CMB["TQU"]['out'][1:3],
        mask2=pmask['100']
    )
    transferfunction = np.sqrt(crosscov[0][:lmax]/C_lS_EE[:lmax])
    io.save_data(transferfunction, io.out_misc_path+"tf_{}".format(cf['pa']['binname']) + "_" + filename)
    io.save_data(crosscov, io.out_misc_path+"crosscov_{}".format(cf['pa']['binname']) + "_" + filename)


if __name__ == '__main__':
    # set_logger(DEBUG)
    bool_weight = True
    bool_tf = False

    if bool_weight:
        run_weight()

    if bool_tf:
        run_tf()