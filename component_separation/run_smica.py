#!/usr/local/bin/python

# %%
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
import functools
import pandas as pd
from component_separation.cs_util import Planckf, Planckr, Plancks
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks

import smica

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/config.json', "r") as f:
    cf = json.load(f)

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
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]

bins = np.array([
    [5.000000000000000000e+00, 9.000000000000000000e+00],
    [1.000000000000000000e+01, 1.900000000000000000e+01],
    [2.000000000000000000e+01, 2.900000000000000000e+01],
    [3.000000000000000000e+01, 3.900000000000000000e+01],
    [4.000000000000000000e+01, 5.900000000000000000e+01],
    [6.000000000000000000e+01, 7.900000000000000000e+01],
    [8.000000000000000000e+01, 9.900000000000000000e+01],
    [1.000000000000000000e+02, 1.190000000000000000e+02],
    [1.200000000000000000e+02, 1.390000000000000000e+02],
    [1.400000000000000000e+02, 1.590000000000000000e+02]])

def _multi(a,b):
    return a*b
def _read(mask_path, mask_filename):
    return {FREQ: hp.read_map(
            '{path}{mask_path}{mask_filename}'
            .format(
                path = indir_path,
                mask_path = mask_path,
                mask_filename = mask_filename
                    .replace("{freqdset}", freqdset)
                    .replace("{freq}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                ), dtype=np.bool)
                for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
            }

def bin_it(data):
    ret = np.ones((*data.shape[:-1], len(bins)))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(bins.shape[0]):
                ret[i,j,k] = np.mean(np.nan_to_num(data[i,j,int(bins[k][0]):int(bins[k][1])]))

            # histbins=np.concatenate(([0],bins[:,1]))
            # # print(np.nan_to_num(data[i,j][:int(bins[:,1][-1])]))
            # ret[i,j] = np.histogram(
            #     np.nan_to_num(data[i,j][:int(bins[:,1][-1])]),
            #     histbins,
            #     weights=np.nan_to_num(data[i,j][:int(bins[:,1][-1])]))[0] / np.histogram(
            #         np.nan_to_num(data[i,j][:int(bins[:,1][-1])]),
            #         histbins)[0]

    return np.nan_to_num(ret)


def fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_fix=False, noise_template=None, afix=None, qmin=0, asyn=None, logger=None, qmax=None, no_starting_point=False):
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
    fixed = False
    if fixed:
        afix = 0-cmb._mixmat.get_mask() #0:free 1:fixed
        cfix = 0-cmb._powspec.get_mask() #0:free 1:fixed
    else:
        afix = 1-cmb._mixmat.get_mask() #0:free 1:fixed
        cfix = 1-cmb._powspec.get_mask() #0:free 1:fixed

    cmbcq = np.squeeze(model.get_comp_by_name("cmb").powspec())
    polar = True if acmb.shape[1]==2 else False

    if not fixed: # TODO; did i interpret this right?  if not is_mixmat_fixed(model)
        if not no_starting_point:
            model.ortho_subspace(stats, nmodes, acmb, qmin=qmin, qmax=qmax)
            cmb.set_mixmat(acmb, fixed=afix)
            if asyn is not None:
                agfix = 1-gal._mixmat.get_mask()
                ag = gal.mixmat()
                agfix[:,-2:] = 1
                ag[0:nmap/2,-2] = asyn
                ag[nmap/2:,-1]  = asyn
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
            print(strtoprint)
        mm = mm2
        mmG = mm2G

    cmb.fix_powspec(cfix)
    gal.fix_powspec("null")
    return model


def build_smica_model(nmap, Q, N):
    N_cov = pw.build_covmatrices(N, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    print(N_cov["EE"].shape)
    N_cov_bn = np.diagonal(bin_it(N_cov["EE"]), offset=0, axis1=0, axis2=1).T
    
    plt.title("Noise spectrum")
    plt.plot(np.mean(bins, axis=1), N_cov_bn[0,:])
    plt.plot(N_cov["EE"][0,0,:159])
    plt.show()
    # print("noise cov: {}".format(N_cov_bn))
    cmb = smica.Source1D (nmap, Q, name='cmb')
    acmb = np.ones((nmap,1)) # if cov in cmb unit
    cmb.set_mixmat(acmb, fixed='all')
    signal = pd.read_csv(
        "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)
    spectrum_trth = signal["Planck-"+"EE"].to_numpy()
    C_lS_bn =  bin_it(np.ones((7,7,3001))* spectrum_trth[:3001])

    plt.plot(np.mean(bins, axis=1), C_lS_bn[0,0,:])
    plt.title("Signal spectrum")
    plt.show()
    # print("C_lS cov: {}".format(C_lS))
    cmbcq = C_lS_bn.T
    cmb.set_powspec(cmbcq[:,0,0]) # where cmbcq is a starting point for cmbcq like binned lcdm

    # cmb.set_powspec(cmbcq[:,0,0]*0, fixed='all') # B modes fit
    dim = 6
    gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6
    gal.fix_mixmat("all")
    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_powspec(N_cov_bn, fixed="all") # where N is a (nmap, Q) array with noise spectra
    noise.set_ampl(np.ones((nmap,1)), fixed="all")

    model = smica.Model(complist=[cmb, gal, noise])
    return model, gal, N_cov_bn


def calc_nmodes(bins, mask):
    nmode = np.ones((bins.shape[0]))
    for idx,q in enumerate(bins):
        rg = np.arange(q[0],q[1]+1) #rg = np.arange(bins[q,0],bins[q,1]+1) correct interpretation?
        nmode[idx] = np.sum(2*rg+1, axis=0)
        fsky = np.mean(mask**2)
        nmode *= fsky
    return nmode


def _reorder_spectrum_dict(spectrum):
    spec_data = dict()
    for f in spectrum.keys():
        for s in spectrum[f].keys():
            if s in spec_data:
                spec_data[s].update({
                    f: spectrum[f][s]})
            else:
                spec_data.update({s:{}})
                spec_data[s].update({
                    f: spectrum[f][s]
                })
    return spec_data

# %%
# smica.fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_fix=False, qmin=0, qmax=None)

freqcomb =  [
    "{}-{}".format(FREQ,FREQ2)
        for FREQ in PLANCKMAPFREQ
        if FREQ not in freqfilter
        for FREQ2 in PLANCKMAPFREQ
        if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]


cf["pa"]['freqdset'] = "DX12-diff"
fname = io.make_filenamestring(cf)
inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/" + cf[mch]["outdir_spectrum"] + "scaled" + fname
noise = io.load_spectrum(inpath_name, fname)


cf["pa"]['freqdset'] = "DX12"
freqdset = cf["pa"]['freqdset']
fname = io.make_filenamestring(cf)
inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+ cf[mch]["outdir_spectrum"] + 'scaled' + fname
spectrum = io.load_spectrum(inpath_name, fname)
cov = pw.build_covmatrices(spectrum, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)["EE"]

# spectrum = _reorder_spectrum_dict(spectrum)

print("empirical cov: {} {}".format(cov.shape, np.nan_to_num(cov[:,:,2])))
cov_bn = bin_it(cov)
print("binned empirical cov: {}".format(cov_bn[:,:,2]))


# %%
plt.plot(np.mean(bins, axis=1), cov_bn[0,0,:])
plt.plot(cov[0,0,:159])
plt.show()

# %%
# replace the following lines with: mask = io.load_mask(cf)
mskset = cf["pa"]['mskset']
pmask_filename = cf[mch][mskset]['pmask']['filename']
indir_path = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/" + cf[mch]['indir'] 
pmask_path = "mask/"
pmasks = [_read(pmask_path, a) for a in pmask_filename]
mask = np.array([functools.reduce(_multi, [a[FREQ] for a in pmasks])
    for FREQ in PLANCKMAPFREQ
    if FREQ not in freqfilter
])

nmodes = calc_nmodes(bins, mask)
print("nmodes: {}".format(nmodes))
Q = len(nmodes)


# %%
# noise = _reorder_spectrum_dict(noise)
smica_model, gal, N_cov_bn = build_smica_model(cov_bn.shape[0], Q, noise)

# %%

fit_model_to_cov(
    smica_model,
    cov_bn,
    nmodes,
    maxiter=50,
    noise_fix=True,
    noise_template=N_cov_bn,
    afix=None, qmin=0,
    asyn=None,
    logger=None,
    qmax=None,
    no_starting_point=False)






# %%
