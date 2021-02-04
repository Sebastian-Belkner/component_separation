"""
run.py: script for executing main functionality of component_separation.draw module

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

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from component_separation.cs_util import Planckf, Plancks
from component_separation.draw import plot as cplt

with open('config.json', "r") as f:
    cf = json.load(f)

with open('component_separation/draw/draw.json', "r") as f:
    dcf = json.load(f)

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

    cplt.plotsave_map(
        data=maps,
        plotsubtitle=plotsubtitle,
        plotfilename=fname)


def plot_weights(fname):
    weights = io.load_weights(spec_path, fname)

    plotsubtitle = 'weights-{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    cplt.plotsave_weights_binned(
        weights,
        cf,
        specfilter,
        plotsubtitle=plotsubtitle,
        plotfilename=fname)


def plot_beamwindowfunctions():
    freqfilter= [
            "545",
            "857"
            ]
    set_logger(DEBUG)
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

    filename = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    #%%
    beamf = io.load_beamf(freqcomb=freqcomb)
    freq = ['030', '044', '070', "100", "143", "217", "353"]

    cplt.plot_beamwindowfunction()


def plot_spectrum_difference(fname):
    desc = "0_SYNscaled"#+"_avg_"
    syn_spectrum = io.load_spectrum(spec_path, "syn/"+desc+fname)
    spectrum = io.load_spectrum(spec_path, "scaled"+fname)
    if "avg" in desc:
        plotsubtitle = 'DIFFERENCE-{freqdset}"{split}" dataset - {mskset} masks - average over 6-10 simulation(s)'.format(
            mskset = mskset,
            freqdset = freqdset,
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    else:
        plotsubtitle = 'DIFFERENCE-{freqdset}"{split}" dataset - {mskset} masks - average over 1 simulation(s)'.format(
            mskset = mskset,
            freqdset = freqdset,
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    diff_spectrum = dict()
    for freqc, val in syn_spectrum.items():
        diff_spectrum.update({freqc: {}})
        for spec, va in syn_spectrum[freqc].items():
            diff_spectrum[freqc].update({spec: np.abs(syn_spectrum[freqc][spec]-spectrum[freqc][spec])/spectrum[freqc][spec]})

    koi = next(iter(syn_spectrum.keys()))
    specs = list(syn_spectrum[koi].keys())
    for spec in specs:
        plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        ax1 = plt.subplot(gs[0])
        cplt.plot_compare_powspec_binned(
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
        cplt.plotsave_powspec_diff_binned(
            plt,
            diff_spectrum,
            cf,
            truthfile=cf[mch]['powspec_truthfile'],
            spec=spec,
            plotsubtitle=plotsubtitle,
            plotfilename="DIFFERENCE"+fname,
            loglog=True,
            color = ['r', 'g', 'b', 'y'],
            alttext='Rel. difference')

        # cplt.plotsave_powspec_combined_binned(
        #     ax1,
        #     ax2,
        #     spec,
        #     plotsubtitle=plotsubtitle,
        #     plotfilename="combined"+fname)

        plt.savefig('vis/spectrum/{}_spectrum/{}_binned_combined--{}.jpg'.format(spec, spec, desc+fname))
        plt.close()


def plot_spectrum(fname):
    spectrum = io.load_spectrum(spec_path, fname)
    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    cplt.plotsave_powspec_binned(
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
    # plot_spectrum(fname = "syn/0_SYNunscaled"+fnamesuf)
    # plot_spectrum(fname = "SYNscaled"+fnamesuf)
    plot_spectrum_difference(fname = fnamesuf)
    # plot_weights(fname = 'weights'+fnamesuf)
