#!/usr/local/bin/python

# %% Load packages, and mostly nonchanging parameters
"""
smcia_fit.py: script for using SMICA. It expects scaled powerspectra for each instrument and a noise estimate.
Both is fed to SMICA, which uses these for generating a model, which eventually gives estimates for foregrounds and CMB signal.

# TODO 
 - check empirical transfer function at low ell. derive by cross spectra  between input and synmap. use testing suit on NERSC
simulation of cmb sky + noise + residual foreground « /global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20%s_sim/%04d/npipe6v20%s_%03d_map.fits
Noisefix maps « /global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20%s_sim/%04d/noisefix/noisefix_%03d%s_%04d.fits
account for low noise coming from simulation maps.
 - smooth spectra, and see if spikes remain, and if weights change
"""
__author__ = "S. Belkner"

import json
import platform
import functools
import pandas as pd
from component_separation.cs_util import Planckf, Planckr, Plancks

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import os

from component_separation.cs_util import Config as csu, Constants as const, Helperfunctions as hpf
import component_separation.io as io
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks
import smica
import component_separation

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPFREQ_f = [FREQ for FREQ in PLANCKMAPFREQ
                    if FREQ not in cf['pa']["freqfilter"]]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
PLANCKSPECTRUM_f = [SPEC for SPEC in PLANCKSPECTRUM
                    if SPEC not in cf['pa']["specfilter"]]
lmax = cf['pa']["lmax"]

freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]

_, mask, _ = io.load_one_mask_forallfreq()

noise_inpath_name = os.path.dirname(component_separation.__file__) +"/"+ io.noise_sc_path_name
noise_spec = io.load_data(noise_inpath_name)
spectrum = io.load_data(io.spec_sc_path_name)

cov = pw.build_covmatrices(spectrum, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)["EE"]
bins = const.SMICA_lowell_bins
offset = 0


# %% Load changing parameters and functions
def calc_nmodes(bins, mask):
    nmode = np.ones((bins.shape[0]))
    for idx,q in enumerate(bins):
        rg = np.arange(q[0],q[1]+1) #rg = np.arange(bins[q,0],bins[q,1]+1) correct interpretation?
        nmode[idx] = np.sum(2*rg+1, axis=0)
        fsky = np.mean(mask**2)
        nmode *= fsky
    print('nmodes: {}, fsky: {}'.format(nmode, fsky))
    return nmode


def build_smica_model(nmap, Q, N):
    # Noise part
    N_cov = pw.build_covmatrices(N, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    N_cov_bn = np.diagonal(hpf.bin_it(N_cov["EE"], bins=bins, offset=offset), offset=offset, axis1=0, axis2=1).T
    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_ampl(np.ones((nmap,1)), fixed="null")
    noise.set_powspec(np.nan_to_num(N_cov_bn), fixed="null") # where N is a (nmap, Q) array with noise spectra
    # print("noise cov: {}".format(N_cov_bn))

    # CMB part
    cmb = smica.Source1D (nmap, Q, name='cmb')
    acmb = np.ones((nmap,1)) # if cov in cmb unit
    cmb.set_mixmat(acmb, fixed='null')
    signal = pd.read_csv(
        os.path.dirname(component_separation.__file__) +"/"+ +cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)
    spectrum_trth = signal["Planck-"+"EE"].to_numpy()
    C_lS_bn =  hpf.bin_it(np.ones((7,7,3001))* spectrum_trth[:3001]/(hpf.ll*(hpf.ll+1))*2*np.pi, bins=bins, offset=offset)

    cmbcq = C_lS_bn[0,0,:]
    cmb.set_powspec(cmbcq) # where cmbcq is a starting point for cmbcq like binned lcdm

    # Galactic foreground part
    # cmb.set_powspec(cmbcq*0, fixed='all') # B modes fit
    dim = 6
    gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6
    gal.fix_mixmat("null")

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
    fixed = False
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
            noise = smica.NoiseDiag(nmap, len(nmodes), name='noise')
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


# %% Bin cov matrix
nmodes = calc_nmodes(bins, mask)
cov_bn = hpf.bin_it(cov, bins=bins, offset=offset)

# %% Plot empirical cov matrix
plt.yscale('log')
for var in range(cov_bn.shape[0]):
    plt.plot(np.mean(bins, axis=1), cov_bn[var,var,:])
    # plt.plot(cov[var,var,offst:int(bins[-1][1])+offst])
plt.show()


# %%
smica_model, gal, N_cov_bn, C_lS_bn = build_smica_model(cov_bn.shape[0], len(nmodes), noise_spec)


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
    qmax=len(nmodes),
    no_starting_point=False)


# %%
label = ["030", "044", "070", "100", "143", "217", "353"]
plt.title('Empiric EE-Powerspectrum (noise + signal + foreground)')
plt.yscale('log')
for var1 in range(C_lS_bn.shape[0]):
    for var2 in range(C_lS_bn.shape[0]):
        if var1==var2:
            plt.plot(np.mean(bins, axis=1), np.abs(cov_bn[var1,var2,:]), label="{}-{}".format(label[var1], label[var2]))
plt.plot(np.mean(bins, axis=1), smica_model.get_comp_by_name('cmb').powspec()[0][0], label= 'smica CMB')
plt.plot(np.mean(bins, axis=1), C_lS_bn[0, 0, :], label='EE Planck best estimate')
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