#!/usr/local/bin/python

# %% Load packages, and mostly nonchanging parameters
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

freqdset = cf["pa"]['freqdset']
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

mskset = cf["pa"]['mskset']
pmask_filename = cf[mch][mskset]['pmask']['filename']
indir_path = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/" + cf[mch]['indir'] 
pmask_path = "mask/"
pmasks = [_read(pmask_path, a) for a in pmask_filename]
mask = np.array([functools.reduce(_multi, [a[FREQ] for a in pmasks])
    for FREQ in PLANCKMAPFREQ
    if FREQ not in freqfilter
])

freqcomb =  [
    "{}-{}".format(FREQ,FREQ2)
        for FREQ in PLANCKMAPFREQ
        if FREQ not in freqfilter
        for FREQ2 in PLANCKMAPFREQ
        if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

freqdset = cf["pa"]['freqdset']
cf["pa"]['freqdset'] = "DX12-diff"
fname = io.make_filenamestring(cf)
# inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/" + cf[mch]["outdir_spectrum"] + "scaled" + fname
def _inpathname(freqc,spec):
    return  "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/tmp/spectrum/"+spec+freqc+"-"+"scaled"+fname
speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
noise_spec = {freqc: {
    spec: np.array(io.load_cl(_inpathname(freqc,spec)))
    for spec in speccs}  
    for freqc in freqcomb}

cf["pa"]['freqdset'] = "DX12"
fname = io.make_filenamestring(cf)
inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+ cf[mch]["outdir_spectrum"] + 'scaled' + fname

speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
spectrum = {freqc: {
    spec: np.array(io.load_cl(_inpathname(freqc,spec)))
    for spec in speccs}  
    for freqc in freqcomb}
cov = pw.build_covmatrices(spectrum, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)["EE"]


# %% Load changing parameters and functions

offst = 0
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

bins = np.array([
# [2.000000000000000000e+00, 9.000000000000000000e+00],
# [1.000000000000000000e+01, 1.900000000000000000e+01],
# [2.000000000000000000e+01, 2.900000000000000000e+01],
# [3.000000000000000000e+01, 3.900000000000000000e+01],
# [4.000000000000000000e+01, 5.900000000000000000e+01],
# [6.000000000000000000e+01, 7.900000000000000000e+01],
# [8.000000000000000000e+01, 9.900000000000000000e+01],
# [1.000000000000000000e+02, 1.190000000000000000e+02],
# [1.200000000000000000e+02, 1.390000000000000000e+02],
# [1.400000000000000000e+02, 1.590000000000000000e+02],
# [1.600000000000000000e+02, 1.790000000000000000e+02],
# [1.800000000000000000e+02, 1.990000000000000000e+02],
# [2.000000000000000000e+02, 2.190000000000000000e+02],
[2.200000000000000000e+02, 2.390000000000000000e+02],
[2.400000000000000000e+02, 2.590000000000000000e+02],
[2.600000000000000000e+02, 2.790000000000000000e+02],
[2.800000000000000000e+02, 2.990000000000000000e+02],
[3.000000000000000000e+02, 3.190000000000000000e+02],
[3.200000000000000000e+02, 3.390000000000000000e+02],
[3.400000000000000000e+02, 3.590000000000000000e+02],
[3.600000000000000000e+02, 3.790000000000000000e+02],
[3.800000000000000000e+02, 3.990000000000000000e+02],
[4.000000000000000000e+02, 4.190000000000000000e+02],
[4.200000000000000000e+02, 4.390000000000000000e+02],
[4.400000000000000000e+02, 4.590000000000000000e+02],
[4.600000000000000000e+02, 4.790000000000000000e+02],
[4.800000000000000000e+02, 4.990000000000000000e+02],
[5.000000000000000000e+02, 5.490000000000000000e+02],
[5.500000000000000000e+02, 5.990000000000000000e+02],
[6.000000000000000000e+02, 6.490000000000000000e+02],
[6.500000000000000000e+02, 6.990000000000000000e+02],
[7.000000000000000000e+02, 7.490000000000000000e+02],
[7.500000000000000000e+02, 7.990000000000000000e+02],
[8.000000000000000000e+02, 8.490000000000000000e+02],
[8.500000000000000000e+02, 8.990000000000000000e+02],
[9.000000000000000000e+02, 9.490000000000000000e+02],
[9.500000000000000000e+02, 9.990000000000000000e+02]
])



def calc_nmodes(bins, mask):
    nmodesflag = True
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


def bin_it(data):
    ret = np.ones((*data.shape[:-1], len(bins)))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(bins.shape[0]):
                ret[i,j,k] = np.mean(np.nan_to_num(data[i,j,offst+int(bins[k][0]):offst+int(bins[k][1])]))
    return np.nan_to_num(ret)


def build_smica_model(nmap, Q, N):
    # Noise part
    N_cov = pw.build_covmatrices(N, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    N_cov_bn = np.diagonal(bin_it(N_cov["EE"]), offset=0, axis1=0, axis2=1).T
    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_ampl(np.ones((nmap,1)), fixed="all")
    noise.set_powspec(np.nan_to_num(N_cov_bn), fixed="all") # where N is a (nmap, Q) array with noise spectra
    # print("noise cov: {}".format(N_cov_bn))


    # CMB part
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

    # print("C_lS cov: {}".format(C_lS))
    cmbcq = C_lS_bn[0,0,:]
    cmb.set_powspec(cmbcq) # where cmbcq is a starting point for cmbcq like binned lcdm

    # Galactic foreground part
    # cmb.set_powspec(cmbcq*0, fixed='all') # B modes fit
    dim = 6
    gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6
    gal.fix_mixmat("all")

    model = smica.Model(complist=[cmb, gal, noise])
    return model, gal, N_cov_bn, C_lS_bn


def fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_fix=False, noise_template=None, afix=None, qmin=0, asyn=None, logger=None, qmax=None, no_starting_point=False):
    """ Fit the model to empirical covariance.
    
    """
    cg_maxiter = 1
    cg_eps = 1e-20
    nmap = stats.shape[0]
    cmb,gal,noise = model._complist

    # find a starting point
    acmb = model.get_comp_by_name("cmb").mixmat()
    print(30*"#")
    print(acmb)
    print(30*"#")
    fixed = True
    if fixed:
        afix = 1-cmb._mixmat.get_mask() #0:free 1:fixed
        cfix = 1-cmb._powspec.get_mask() #0:free 1:fixed
    else:
        afix = 0-cmb._mixmat.get_mask() #0:free 1:fixed
        cfix = 0-cmb._powspec.get_mask() #0:free 1:fixed

    cmbcq = np.squeeze(model.get_comp_by_name("cmb").powspec())
    polar = True if acmb.shape[1]==2 else False

    if fixed: # TODO; did i interpret this right?  if not is_mixmat_fixed(model)
        if not no_starting_point:
            model.ortho_subspace(np.abs(stats), nmodes, acmb, qmin=qmin, qmax=qmax)
            cmb.set_mixmat(acmb, fixed=afix)
            if asyn is not None:
                agfix = 1-gal._mixmat.get_mask()
                ag = gal.mixmat()
                agfix[:,-2:] = 1
                ag[0:int(nmap/2),-2] = asyn
                ag[int(nmap/2):,-1]  = asyn
                gal.set_mixmat(ag, fixed=agfix)
            
    model.quasi_newton(np.abs(stats), nmodes)
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
        if i==int(maxiter/2) and not noise_fix: # fit also noise at some point
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


# %% Calculate nmodes if necessary
nmodes = calc_nmodes(bins, mask)

# spectrum = _reorder_spectrum_dict(spectrum)

# %% Bin cov matrix
cov_bn = bin_it(cov)


# %% Plot empirical cov matrix
plt.yscale('log')
for var in range(cov_bn.shape[0]):
    plt.plot(np.mean(bins, axis=1), cov_bn[var,var,:])
    # plt.plot(cov[var,var,offst:int(bins[-1][1])+offst])
plt.show()


# %%
# noise = _reorder_spectrum_dict(noise)
smica_model, gal, N_cov_bn, C_lS_bn = build_smica_model(cov_bn.shape[0], len(nmodes), noise_spec)

# %%
print(cov_bn.shape)
print(N_cov_bn.shape)
print(smica_model.dim)
# print(smica_model.dim())
print(smica_model.get_dim())
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
    qmax=len(nmodes),
    no_starting_point=False)

# %%
label = ["030", "044", "070", "100", "143", "217", "353"]
plt.title('Empiric EE-Powerspectrum (noise + signal + foreground)')
plt.yscale('log')
for var1 in range(C_lS_bn.shape[0]):
    for var2 in range(C_lS_bn.shape[0]):
        if var1==var2:
    # plt.plot(np.mean(bins, axis=1), C_lS_bn[var, var, :] + N_cov_bn[var], label=var)
            plt.plot(np.mean(bins, axis=1), np.abs(cov_bn[var1,var2,:]), label="{}-{}".format(label[var1], label[var2]))
plt.xlabel('Multipole')
plt.legend()
plt.ylabel('Powerspectrum')
plt.savefig("Empiric_EE-Spectra.jpg")
# %%
plt.yscale('log')
plt.title('EE-Noise from diffmap + CMB EE-Signal from Planck-best-fit')
for var1 in range(C_lS_bn.shape[0]):
    for var2 in range(C_lS_bn.shape[0]):
        
            plt.plot(np.mean(bins, axis=1), C_lS_bn[var1, var2, :] + N_cov_bn[var1], label="{}-{}".format(label[var1], label[var2]))
        # else:
        #     plt.plot(np.mean(bins, axis=1), C_lS_bn[var1, var2, :], label="{}-{}".format(label[var1], label[var2]))
    # plt.plot(np.mean(bins, axis=1), cov_bn[var,var,:])
plt.xlabel('Multipole')
plt.legend()
plt.ylabel('Powerspectrum')
# %%

N_covt = pw.build_covmatrices(noise_spec, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
N_cov_bnt = bin_it(N_covt["EE"]).T

for var in range(cov_bn.shape[2]):
    np.linalg.inv(cov_bn[:,:,var])
    if not np.all(np.linalg.eigvals(cov_bn[:,:,var]) > 0):
        print(np.all(np.linalg.eigvals(cov_bn[:,:,var]) > 0))
    # print(np.all(np.linalg.eigvals(C_lS_bn.T + N_cov_bnt) > 0))

# %%
