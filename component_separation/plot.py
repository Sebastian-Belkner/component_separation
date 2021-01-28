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

def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    plotfilename = '{freqdset}_lmax-{lmax}_lmaxmsk-{lmax_mask}_msk-{mskset}_{freqs}_{spec}_{split}'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    io.plotsave_powspec_binned(
        df_scbf,
        cf,
        specfilter,
        truthfile=cf[mch]['powspec_truthfile'],
        plotsubtitle=plotsubtitle,
        plotfilename=plotfilename)

    io.plotsave_weights_binned(
        weights,
        cf,
        specfilter,
        plotsubtitle=plotsubtitle,
        plotfilename=plotfilename)


    df = dict()
    for key, val in df_scbf.items():
        df.update({key:(df_scbf[key]-syn_df_scbf[key]).div(syn_df_scbf[key])})
    plotfilename = 'DIFFERENCE-{freqdset}_lmax-{lmax}_lmaxmsk-{lmax_mask}_msk-{mskset}_{freqs}_{spec}_{split}'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    plotsubtitle = 'DIFFERENCE-{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    io.plotsave_powspec_binned(
        df,
        cf,
        specfilter,
        truthfile=cf[mch]['powspec_truthfile'],
        plotsubtitle=plotsubtitle,
        plotfilename=plotfilename,
        loglog=False)


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