#!/usr/local/bin/python
"""
smcia_fit.py: script for using SMICA. It expects scaled powerspectra for each instrument and a noise estimate.
Both is fed to SMICA, which uses these for generating a model, which eventually gives estimates for foregrounds and CMB signal.

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

import matplotlib.pyplot as plt
import numpy as np

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks

import smica

with open('config.json', "r") as f:
    cf = json.load(f)

LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=0
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

spec_path = cf[mch]['outdir_spectrum']
weight_path = cf[mch]['outdir_weight']
indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)
    logging.StreamHandler(sys.stdout)

def build_smica_model():
    cmb = smica.Source1D (nmap, Q, name='cmb')
    acmb = np.ones((nmap,1)) # if cov in cmb unit
    cmb.set_mixmat(acmb, fixed='all')
    cmb.set_powspec(cmbcq) # where cmbcq is a starting point for cmbcq like binned lcdm
    cmb.set_powspec(cmbcq*0, fixed='all') # B modes fit

    gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6

    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_powspec(N, fixed="all") # where N is a (nmap, Q) array with noise spectra
    noise.set_ampl(np.ones((nmap,1)), fixed="all")

    model = smica.Model(complist=[cmb, gal, noise])
    return model, gal


def fit_mixing_matrix():
    gal.fix_mixmat("all")

def calc_nmodes():
    pass
    # for each bin q
    #     rg = np.arange(bins[q,0],bins[q,1]+1)
    #     nmode[q] = np.sum(2*rg+1, axis=0)
    #     fsky = np.mean(mask**2)
    #     nmode *= fsky
    # return nmode


if __name__ == '__main__':
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(40*"$")
    model = None
    stats = None
    nmodes = None

    # smica.fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_fix=False, qmin=0, qmax=None)

    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

    set_logger(DEBUG)

    fname = io.make_filenamestring(cf)
    inpath_name = cf[mch]["outdir_spectrum"] + 'scaled' + fname
    spectrum = io.load_spectrum(inpath_name, fname)

    fname = io.make_filenamestring(cf)
    inpath_name = cf[mch]["outdir_spectrum"] + "scaled-difference" + fname
    noise = io.load_spectrum(inpath_name, fname)

    nmodes = calc_nmodes()

    smica_model = build_smica_model()

    fit_mixing_matrix()










