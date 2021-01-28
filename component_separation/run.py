"""
run.py: script for executing main functionality of component_separation

"""

__author__ = "S. Belkner"

# analytic expression for weight estimates
# TODO add LFI beam window functions to calculation
# compare to planck cmb simulations data
# use, in addition to the current datasets, cross and diff datasets
# serialise cov_matrix results and weighting results (to allow for combined plots)
# remove pandas usage

import json
import logging
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

with open('config.json', "r") as f:
    cf = json.load(f)


def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)


def spec2synmap(spectrum):
    return pw.create_synmap(spectrum, cf, mch, lmax, lmax_mask, freqfilter, specfilter) 


def map2spec(tqumap):
    if tqumap[0] == None:
        tqumap = tqumap[1:]
    elif tqumap[1] == None:
        tqumap = [tqumap[0]]
    tqumap_hpcorrected = pw.hphack(tqumap)
    # tqumap_hpcorrected = tqumap
    if len(tqumap) == 3:
        spectrum = pw.tqupowerspec(tqumap_hpcorrected, cf, lmax, lmax_mask, freqfilter, specfilter)
    elif len(tqumap) == 2:
        spectrum = pw.qupowerspec(tqumap_hpcorrected, cf, lmax, lmax_mask, freqfilter, specfilter)
    elif len(tqumap) == 1:
        print("Only TT spectrum caluclation requested. This is currently not supported.")
    return spectrum


def spec2specsc(spectrum, bf, llp1):
    # df = pw.create_df(spectrum, cf["pa"]["offdiag"], freqfilter, specfilter)
    df_sc = pw.apply_scale(spectrum, specfilter, llp1=llp1)
    if bf:
        beamf = io.load_beamf(freqcomb=freqcomb)
        df_scbf = pw.apply_beamfunction(df_sc, beamf, lmax, specfilter) #df_sc #
    else:
        df_scbf = df_sc

    return df_scbf


def specsc2weights(spec_sc):
    cov = pw.build_covmatrices(spec_sc, cf["pa"]["offdiag"], lmax, freqfilter, specfilter)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)
    return weights


if __name__ == '__main__':
    mskset = cf['pa']['mskset'] # smica or lens
    freqdset = cf['pa']['freqdset'] # DX12 or NERSC
    set_logger(DEBUG)

    lmax = cf['pa']["lmax"]
    lmax_mask = cf['pa']["lmax_mask"]
    llp1 = cf['pa']["llp1"]
    bf = cf['pa']["bf"]

    spec_path = cf[mch]['outdir']
    indir_path = cf[mch]['indir']
    freqfilter = cf['pa']["freqfilter"]
    specfilter = cf['pa']["specfilter"]
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

    spec_filename = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    if cf['pa']['new_spectrum']:
        spectrum = map2spec(io.load_tqumap())
        io.save_spectrum(spectrum, spec_path, 'unscaled'+spec_filename)
    else:
        spectrum = io.load_spectrum(spec_path, 'unscaled'+spec_filename)

    spectrum_scaled = spec2specsc(spectrum, bf=bf, llp1=llp1)
    io.save_spectrum(spectrum_scaled, spec_path, 'scaled'+spec_filename)

    weights = specsc2weights(spectrum_scaled)
    io.save_weights(weights)
    
    synmaps = spec2synmap(spectrum)
    io.save_map(synmaps)

    spec_filename = 'SYN-{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    syn_spectrum = map2spec(synmaps)
    io.save_spectrum(syn_spectrum, spec_path, spec_filename)

    syn_spectrum_scaled = spec2specsc(syn_spectrum, bf, llp1)
    io.save_spectrum(syn_spectrum_scaled, spec_path, spec_filename)

    weights = specsc2weights(syn_spectrum_scaled)
    io.save_weights(weights)