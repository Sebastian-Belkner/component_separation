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

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

nside_out = cf['pa']['nside_out'] if cf['pa']['nside_out'] is not None else cf['pa']['nside_desc_map']
num_sim = cf['pa']["num_sim"]

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
detectors = csu.PLANCKMAPFREQ_f
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]
tmask, pmask, pmask = io.load_mask_per_freq(nside_out[0]) #io.load_one_mask_forallfreq(nside_out=nside_out)


def spec_weight2weighted_spec(spectrum, weights):
    alms = pw.spec2alms(spectrum)
    alms_w = pw.alms2almsxweight(alms, weights)
    spec = pw.alms2cls(alms_w)
    return spec


def specsc2weights(spectrum):
    print(spectrum.shape)
    cov = pw.build_covmatrices(spectrum, cf['pa']['Tscale'])
    print(cov.shape)
    cov_inv_l = pw.invert_covmatrices(cov)
    print(cov_inv_l.shape)
    weights = pw.calculate_weights(cov_inv_l, cf['pa']['Tscale'])
    return weights


if __name__ == '__main__':
    # set_logger(DEBUG)
    run_weight = True
    run_tf = True

    CMB = dict()
    CMB["TQU"] = dict()
    almT, almE, almB = dict(), dict(), dict()

    filename_raw = io.total_filename_raw
    filename = io.total_filename
    ndet = len(detectors)
    bins =  const.SMICA_lowell_bins    #const.linear_equisized_bins_10 #const.linear_equisized_bins_1
    freqdset = cf['pa']['freqdset']
    sim_id = cf[mch][freqdset]['sim_id']

    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated filename(s) for this session: {}".format(filename_raw))
    print(filename)
    print(40*"$")

    if run_weight:
        """
        Calculates weights derived from data defined by freqdset attribute of config.
        No SMICA, straightforward weight derivation.
        Needed for combining maps without SMICA.
        """
        C_ltot = cslib.load_powerspectra('full')
        weights_tot = specsc2weights(C_ltot)
        print(weights_tot.shape)
        io.save_data(weights_tot, io.weight_path_name)

    if run_tf:
        #### State: deriving a map from the cmb spectrum of smica turns out to be difficult, as either synalm or synfast needs to be used. then smica uses binned data, so the data
        #### needs to be interpolated -> deriving crosscovariance between pure cmb in and pure cmb out is difficult

        # CMB["TT"] = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_0200.fits', hdu=1)[:281625]
        # buff = io.load_data("/global/cscratch1/sd/sebibel/smica/C_lS_out.npy")[0][0]
        # xnew = np.arange(0,lmax,1)
        # buffinterp = interpolate.interp1d(np.mean(bins, axis=1), buff, bounds_error = False, fill_value='extrapolate')
        # CMB["EEinterp"] = hp.synalm(buffinterp(xnew))
        # CMB["BB"] = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_0200.fits', hdu=3)[:281625]
        # print(CMB["TT"].shape, CMB["EEinterp"].shape , CMB["BB"].shape, hp.Alm.getlmax(len(CMB["BB"])))
        # CMB["TQU"]['out'] = hp.alm2map([CMB["TT"], CMB["EEinterp"] , CMB["BB"]], nside_out)

        # another way would be to do crosscovariance pure cmb in with minimum variance map derived from smica. that is easy and works.

        C_lS_EE = io.load_data("/global/cscratch1/sd/sebibel/misc/C_lS_in.npy")[0][0]
        # io.save_data(CMB["TQU"]['out'], "/global/cscratch1/sd/sebibel/misc/cmboutmap.npy")

        # crosscovariance between cmb input and what smica gives
        # CMB["TQU"]["in"] = io.load_data("/global/cscratch1/sd/sebibel/misc/C_lS_in.npy")
        CMB["TQU"]["in"] = io.load_data("/global/cscratch1/sd/sebibel/misc/cmbinmap.npy")
        CMB["TQU"]['out'] = io.load_data("/global/cscratch1/sd/sebibel/misc/smicaminvarmap.npy")
        # CMB["TQU"]['out'] = io.load_data("/global/cscratch1/sd/sebibel/misc/C_lS_out.npy")

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
        io.save_data(transferfunction, "/global/cscratch1/sd/sebibel/misc/tf.npy")
        io.save_data(crosscov, "/global/cscratch1/sd/sebibel/misc/crosscov.npy")