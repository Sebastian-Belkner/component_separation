"""
run.py: script for executing main functionality of component_separation

"""

__author__ = "S. Belkner"

import component_separation.MSC.MSC.pospace as ps
from component_separation.cs_util import Planckf, Plancks
import logging
import logging.handlers
from logging import DEBUG, ERROR, INFO, CRITICAL
import os, sys
import matplotlib.gridspec as gridspec
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
    if "syn" in fname:
        maps = io.load_map(spec_path, fname)
    else:
        maps = io.load_tqumap()

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    io.plotsave_map(
        data=maps,
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

def plot_spectrum_difference(fname):
    syn_spectrum = io.load_spectrum(spec_path, "SYNscaled"+fname)
    spectrum = io.load_spectrum(spec_path, "scaled"+fname)
    import copy
    plotsubtitle = 'DIFFERENCE-{freqdset}"{split}" dataset - {mskset} masks - average over 10 simulations'.format(
            mskset = mskset,
            freqdset = freqdset,
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    diff_spectrum = copy.deepcopy(syn_spectrum)
    for freqc, val in diff_spectrum.items():
        for spec, va in diff_spectrum[freqc].items():
            diff_spectrum[freqc][spec] = (diff_spectrum[freqc][spec]-spectrum[freqc][spec])/diff_spectrum[freqc][spec]

    koi = next(iter(syn_spectrum.keys()))
    specs = list(syn_spectrum[koi].keys())
    for spec in specs:
        plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        ax1 = plt.subplot(gs[0])
        io.plot_compare_powspec_binned(
            plt,
            spectrum,
            syn_spectrum,
            cf,
            truthfile=cf[mch]['powspec_truthfile'],
            spec=spec,
            plotsubtitle=plotsubtitle,
            plotfilename="compare"+fname,
            loglog=True)
        
        ax2 = plt.subplot(gs[1])
        io.plotsave_powspec_binned(
            plt,
            diff_spectrum,
            cf,
            truthfile=cf[mch]['powspec_truthfile'],
            spec=spec,
            plotsubtitle=plotsubtitle,
            plotfilename="DIFFERENCE"+fname,
            loglog=False,
            color = ['r', 'g', 'b', 'y'],
            alttext='Rel. difference')

        # io.plotsave_powspec_combined_binned(
        #     ax1,
        #     ax2,
        #     spec,
        #     plotsubtitle=plotsubtitle,
        #     plotfilename="combined"+fname)

        plt.savefig('vis/spectrum/{}_spectrum/{}_binned--{}.jpg'.format(spec, spec, "SYNscaled_combined"+fname))
        plt.close()


def plot_spectrum(fname):
    spectrum = io.load_spectrum(spec_path, fname)
    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    io.plotsave_powspec_binned(
        plt,
        spectrum,
        cf,
        truthfile=cf[mch]['powspec_truthfile'],
        spec=None,
        plotsubtitle=plotsubtitle,
        plotfilename=fname
        )


if __name__ == '__main__':
    set_logger(DEBUG)
    fnamesuf = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
# "synmaps"+
    # plot_maps(fname = fnamesuf)
    # plot_spectrum(fname = "scaled"+fnamesuf)
    # plot_spectrum(fname = "SYNscaled"+fnamesuf)
    # plot_spectrum_difference(fname = fnamesuf)
    plot_weights(fname = 'weights'+fnamesuf)
