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

import component_separation.MSC.MSC.pospace as ps
from component_separation.cs_util import Planckf, Plancks
import logging
import logging.handlers
from logging import DEBUG, ERROR, INFO, CRITICAL
import os, sys
import component_separation.powspec as pw
import component_separation.io as io
from typing import Dict, List, Optional, Tuple
import json
import platform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open('config.json', "r") as f:
    cf = json.load(f)

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


def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)


def plot_maps(fname):
    maps = io.load_weights(spec_path, fname)

    plotsubtitle = 'maps-{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    io.plotsave_maps(
        maps,
        cf,
        specfilter,
        plotsubtitle=plotsubtitle,
        plotfilename=fname)


def plot_weights(fname):
    weights = io.load_weights(spec_path, fname)

    plotsubtitle = 'weights-{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    io.plotsave_weights_binned(
        weights,
        cf,
        specfilter,
        plotsubtitle=plotsubtitle,
        plotfilename=fname)


def plot_spectrum(fname):
    spectrum = io.load_spectrum(spec_path, fname)
    
    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    io.plotsave_powspec_binned(
        spectrum,
        cf,
        truthfile=cf[mch]['powspec_truthfile'],
        plotsubtitle=plotsubtitle,
        plotfilename=fname)


if __name__ == '__main__':
    fnamesuf = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    plot_spectrum(fname = "unscaled"+fnamesuf)
    plot_weights(fname = 'SYNweights'+fnamesuf)
    plot_maps(fname = 'map'+fnamesuf)

    

    # name = 'SYNscaled'+fnamesuf
    # syn_spectrum = io.load_spectrum(spec_path, name)
    # import copy
    # diff_spectrum = copy.deepcopy(syn_spectrum)
    # for freqc, val in diff_spectrum.items():
    #     for spec, va in diff_spectrum[freqc].items():
    #         print(diff_spectrum[freqc][spec])
    #         print(30*"@")
    #         print(spectrum[freqc][spec])
    #         print(30*"@")
    #         print(diff_spectrum[freqc][spec]-spectrum[freqc][spec])
    #         print(30*"@")
    #         print(30*"@")
    #         print(30*"@")
    #         diff_spectrum[freqc][spec] = (diff_spectrum[freqc][spec]-spectrum[freqc][spec])/diff_spectrum[freqc][spec]

    # plotfilename = 'DIFFERENCE-{freqdset}_lmax-{lmax}_lmaxmsk-{lmax_mask}_msk-{mskset}_{freqs}_{spec}_{split}'.format(
    #     freqdset = freqdset,
    #     lmax = lmax,
    #     lmax_mask = lmax_mask,
    #     mskset = mskset,
    #     spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
    #     freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
    #     split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    # plotsubtitle = 'DIFFERENCE-{freqdset}"{split}" dataset - {mskset} masks'.format(
    #     mskset = mskset,
    #     freqdset = freqdset,
    #     split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])


    # io.plotsave_weights(
    #     weights,
    #     plotsubtitle=plotsubtitle,
    #     plotfilename=plotfilename)

    # io.plotsave_powspec(
    #     df_scbf,
    #     specfilter,
    #     truthfile=cf[mch]['powspec_truthfile'],
    #     plotsubtitle=plotsubtitle,
    #     plotfilename=plotfilename)