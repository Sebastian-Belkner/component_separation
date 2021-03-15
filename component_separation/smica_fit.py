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


def fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_fix=False, noise_template=None, afix=None, qmin=0, async=None, logger=None, qmax=None, no_starting_point=False):
    """ Fit the model to empirical covariance.
    
    """
    cg_maxiter = 1
    cg_eps = 1e-20
    Q = len(nmodes)
    if qmax is None:
        qmax = Q
    nmap = stats.shape[0]
    cmb,gal,noise = model._complist

    # find a starting point
    acmb = model.get_comp_by_name("cmb").mixmat()
    afix = 1-cmb._mixmat.get_mask() #0:free 1:fixed
    cmbcq = np.squeeze(model.get_comp_by_name("cmb").powspec())
    cfix = 1-cmb._powspec.get_mask() #0:free 1:fixed
    polar = True if acmb.shape[1]==2 else False
    if not is_mixmat_fixed(model) and not no_starting_point:
        model.ortho_subspace(stats, nmodes, acmb, qmin=qmin, qmax=qmax)
        cmb.set_mixmat(acmb, fixed=afix)
        if async is not None:
            agfix = 1-gal._mixmat.get_mask()
            ag = gal.mixmat()
            agfix[:,-2:] = 1
            ag[0:nmap/2,-2] = async
            ag[nmap/2:,-1]  = async
            gal.set_mixmat(ag, fixed=agfix)
            
    model.quasi_newton(stats, nmodes)
    cmb.set_powspec (cmbcq, fixed=cfix)
    model.close_form(stats)
    cmb.set_powspec (cmbcq, fixed=cfix)
    mm = model.mismatch(stats, nmodes, exact=True)
    mmG = model.mismatch(stats, nmodes, exact=True)

    # start CG/close form
    for i in range(maxiter):
        # fit mixing matrix
        gal.fix_powspec("all")
        cmbfix = "all" if polar else cfix 
        cmb.fix_powspec(cmbfix)
        model.conjugate_gradient (stats, nmodes,maxiter=cg_maxiter, avextol=cg_eps)
        if 0:#logger is not None:
            np.set_printoptions(precision=5)
            logger.info(str(cmb.mixmat()/acmb))
        # fit power spectra
        gal.fix_powspec("null")
        if mmG!=model.mismatch(stats, nmodes, exact=True):
            cmbfix = cfix if polar else "all" 
            cmb.fix_powspec(cmbfix)
        if i==maxiter/2 and not noise_fix: # fit also noise at some point
            Nt = noise.powspec()
            noise = smica.NoiseDiag(nmap, Q, name='noise')
            model = smica.Model([cmb, gal, noise])
            if noise_template is not None:
                noise.set_powspec(Nt, fixed=noise_template)
            else:
                noise.set_powspec(Nt)
            cmb.set_powspec (cmbcq)
        model.close_form(stats)

        # compute new mismatch 
        mm2 = model.mismatch(stats, nmodes, exact=True)
        mm2G = model.mismatch(stats, nmodes)
        gain = np.real(mmG-mm2G)
        if gain==0 and i>maxiter/2.0:
            break
        strtoprint = "iter= % 4i mismatch = %10.5f  gain= %7.5f " % (i, np.real(mm2), gain)
        if logger is not None:
            logger.info(strtoprint)
        else:
            print strtoprint
        mm = mm2
        mmG = mm2G

    cmb.fix_powspec(cfix)
    gal.fix_powspec("null")
    return model



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










