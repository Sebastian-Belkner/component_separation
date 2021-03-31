#!/usr/local/bin/python

"""
run.py: 
    Script for calculating the gain of the optimal powerspectrum. 
    Section III takes no data.
"""

# %% Load packages and helper functions

__author__ = "S. Belkner"


import json
import pandas as pd
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple
from numpy import inf
import component_separation.io as io

import component_separation.powspec as pw
import healpy as hp
import numpy.ma as ma

import matplotlib.pyplot as plt
import numpy as np

from component_separation.cs_util import Planckf, Plancks

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"
lmax = 3000
n_cha = 7
shape = (lmax+1, n_cha, n_cha)

PLANCKSPECTRUM = [p.value for p in list(Plancks)]
PLANCKMAPFREQ = [p.value for p in list(Planckf)]


base = 10
nbins = 150

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/config.json', "r") as f:
    cf = json.load(f)


signal = pd.read_csv(
        "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)
spectrum_trth = signal["Planck-"+"EE"][:shape[0]].to_numpy()


def std_dev_binned(d, bins):
    print(d.shape)
    val = np.nan_to_num(d.compressed())
    linsp = np.where(d.mask==False)[0]
    n, _ = np.histogram(
        linsp,
        bins=bins)
    sy, _ = np.histogram(
        linsp,
        bins=bins,
        weights=val)
    sy2, _ = np.histogram(
        linsp,
        bins=bins,
        weights=val * val)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    return mean, std, _
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def create_noisespectrum(dp, fwhm):
    """Analog to gauss_beam

    Args:
        arcmin ([type]): [description]
        fwhm ([type], optional): [description]. Defaults to np.array([2,10,27]).

    Returns:
        [type]: [description]
    """   
    rad = dp[np.newaxis].T * np.ones(lmax+1)  * 0.000290888
    ll = np.arange(0,lmax+1,1)
    # TODO AMPLITUDE SHOULD be noise level in muK arcmin
    def _spec(rad):
        return (dp)**2 * np.exp(ll*(ll+1)*fwhm**2/(8*np.log(2)))
        # return (2*0.000290888/(2.728*1e6))**2 * np.exp(ll*(ll+1)*rad**2/(8*np.log(2)))
    return _spec(rad)


def gauss_beam(fwhm):
    return (fwhm/(2.728))**2 * 1/hp.gauss_beam(fwhm, lmax=lmax, pol=True)


def create_covmatrix(spectrum):
    """Creates auto and cross covariance matrix, with noise only, so no cross.

    Args:
        spectrum ([type]): [description]

    Returns:
        [type]: [description]
    """    

    row, col = np.diag_indices(spectrum.shape[0])
    C = np.zeros(shape, float)
    for l in range(C.shape[0]):
        C[l,row,col] = spectrum[:,l]
    return C


def calculate_weights(cov):
    elaw = np.ones(cov.shape[1])
    weights = (cov @ elaw) / (elaw @ cov @ elaw)[np.newaxis].T
    return weights
    

def calculate_minimalcov(C_lN: np.ndarray, C_lS: np.ndarray = 0, C_lF: np.ndarray = 0) -> np.array:
    """Returns the minimal covariance using inverse variance weighting, i.e. calculates
    :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

    Args:
        C_lS (np.ndarray): An array of auto- and cross-covariance matrices of the CMB for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lF (np.ndarray): An array of auto- and cross-covariance matrices of the superposition of Foregrounds for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lN (np.ndarray): An array of auto- and cross-covariance matrices of the Noise for all instruments, its dimension is [Nspec,Nspec,lmax]
    """
    elaw = np.ones(C_lN.shape[1])
    cov_minimal = (elaw @ np.linalg.inv(C_lS + C_lF + C_lN) @ elaw)
    return 1/cov_minimal

def calculate_minimalcov2(C_l) -> np.array:
    """Returns the minimal covariance using inverse variance weighting, i.e. calculates
    :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

    Args:
        C_lS (np.ndarray): An array of auto- and cross-covariance matrices of the CMB for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lF (np.ndarray): An array of auto- and cross-covariance matrices of the superposition of Foregrounds for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lN (np.ndarray): An array of auto- and cross-covariance matrices of the Noise for all instruments, its dimension is [Nspec,Nspec,lmax]
    """
    elaw = np.ones(C_l.shape[1])
    cov_minimal = (elaw @ np.linalg.inv(C_l) @ elaw)
    return 1/cov_minimal


def calculate_minimalcov3(C_l) -> np.array:
    """Returns the minimal covariance using inverse variance weighting, i.e. calculates
    :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

    Args:
        C_lS (np.ndarray): An array of auto- and cross-covariance matrices of the CMB for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lF (np.ndarray): An array of auto- and cross-covariance matrices of the superposition of Foregrounds for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lN (np.ndarray): An array of auto- and cross-covariance matrices of the Noise for all instruments, its dimension is [Nspec,Nspec,lmax]
    """
    elaw = np.ones(C_l.shape[-1])
    cov_minimal = (elaw @ np.linalg.inv(C_l) @ elaw)
    return 1/cov_minimal


def load_empiric_noise_aac_covmatrix():
    """loads auto and cross (aac) covariance matrix from diff planck maps

    Returns:
        [type]: [description]
    """    
    cf["pa"]["mskset"] =  "smica"
    cf["pa"]["freqdset"] =  "DX12-diff"
    fname = io.make_filenamestring(cf)
    inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/tmp/spectrum/scaled"+fname
    spectrum = io.load_spectrum(inpath_name, fname)
    return spectrum


# %% Set 'fixed' parameters

lmax = cf['pa']["lmax"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]

a = np.zeros(shape, float)
C_lF = np.zeros(shape, float)
C_lN = np.zeros(shape, float)

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
noise_spectrum = load_empiric_noise_aac_covmatrix()
C_lN  = pw.build_covmatrices(noise_spectrum, lmax, freqfilter, specfilter)


# %% Build toy-cl
C_lS = np.zeros_like(C_lN["EE"].T, float)
row, col = np.diag_indices(C_lS[0,:,:].shape[0])
C_lS =  (np.ones((7,7,3001))* spectrum_trth).T
C_lF = np.zeros_like(C_lN["EE"].T, float)


"""
                                    ##########################################
                                    ##########################################
                                                    SECTION 1
                                    ##########################################
                                    ##########################################
""" 




# %% Create Noisespectrum from gaussbeam
npatch=3
m = np.array([list(range(1, n_cha+1)) for patch in range(npatch)]).T
#TODO take realistic numbers
noiseamp_patch = np.array(list(range(1,npatch+1)))*10+30
noiselevel = (m * noiseamp_patch).T
fwhm = (m * noiseamp_patch*0.0001).T
ll = np.arange(0,lmax+1,1)
noise = np.array(list(map(create_noisespectrum, noiselevel.flatten()*0.000290888, fwhm.flatten())))[:,:].reshape(*noiselevel.shape, lmax+1)*ll*(ll+1)/(2*np.pi)

fsky = np.zeros((npatch, npatch), float)
np.fill_diagonal(fsky, (0.23,0.71, 0.04))


# %%
C_lN = np.array(list(map(create_covmatrix, noise)))
C_lN_fullsky = (C_lN.T @ np.diag(fsky)).T
C_p = C_lS + C_lN
C_pNS = 0 * C_lS + C_lN
C_pNN = C_lS + 0.000001 * C_lN
C_pFS = C_lS + C_lN_fullsky
C_pNS_FS = 0 * C_lS + C_lN_fullsky
cov_min = np.concatenate((np.zeros(shape=(C_p.shape[0],1)),np.nan_to_num(calculate_minimalcov3(C_p[:,1:]))),axis=1)
cov_min_NS = np.concatenate((np.zeros(shape=(C_p.shape[0],1)),np.nan_to_num(calculate_minimalcov3(C_pNS[:,1:]))),axis=1)
cov_min_NN = np.concatenate((np.zeros(shape=(C_p.shape[0],1)),np.nan_to_num(calculate_minimalcov3(C_pNN[:,1:]))),axis=1)
cov_min_FS = np.concatenate((np.zeros(shape=(1)),np.nan_to_num(calculate_minimalcov3(C_pFS[1:]))),axis=0)
cov_min_NS_FS = np.concatenate((np.zeros(shape=(1)),np.nan_to_num(calculate_minimalcov3(C_pNS_FS[1:]))),axis=0)

# %%
var_patches = np.zeros((lmax+1,C_p.shape[0],C_p.shape[0]), float)
var_patches = np.array([2*np.outer(cov_min[:,l], cov_min[:,l]) for l in range(var_patches.shape[0])])/np.outer((2*ll+1),fsky).reshape(lmax+1,C_p.shape[0],C_p.shape[0])
var_patches_NS = np.array([2*np.outer(cov_min_NS[:,l], cov_min_NS[:,l]) for l in range(var_patches.shape[0])])/np.outer((2*ll+1),fsky).reshape(lmax+1,C_p.shape[0],C_pNS.shape[0])
var_patches_NN = np.array([2*np.outer(cov_min_NN[:,l], cov_min_NN[:,l]) for l in range(var_patches.shape[0])])/np.outer((2*ll+1),fsky).reshape(lmax+1,C_p.shape[0],C_pNN.shape[0])

var_patches[var_patches == inf] = 0
var_patches_NS[var_patches_NS == inf] = 0
var_patches_NN[var_patches_NN == inf] = 0

var_FS = np.array(
    [(2*cov_min_FS[l] * cov_min_FS[l])/((2*l+1)*0.96)
    for l in range(cov_min_FS.shape[0])])
var_FSma = ma.masked_array(var_FS, mask=np.where(var_FS<=0, True, False))

# %% no-noise combination, cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_NN = np.zeros((var_patches_NN.shape[0]))
for l in range(var_patches_NN.shape[0]):
    try:
        opt_NN[l] = np.nan_to_num(
            calculate_minimalcov2(np.array([
                [cov_min_NN[l], 0],
                [0, cov_min_NN[l]]])))
    except:
        pass
opt_NNma = ma.masked_array(opt_NN, mask=np.where(opt_NN<=0, True, False))


# %% Idea 3.3: no signal all sky, cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_NS = np.zeros((var_patches_NS.shape[0]))
for l in range(var_patches_NS.shape[0]):
    try:
        opt_NS[l] = np.nan_to_num(
            calculate_minimalcov2(np.array([
                [cov_min_NS[0][l], 0],
                [0, cov_min_NS[l]]])))
    except:
        pass
opt_NSma = ma.masked_array(opt_NS, mask=np.where(opt_NS<=0, True, False))


# %% cov_{min} =  calculate_minimalcov3(cosmic_variance(Cp1_l, Cp2_l))
opt = np.zeros((var_patches.shape[0]))
for l in range(var_patches.shape[0]):
    try:
        opt[l] = np.nan_to_num(
            calculate_minimalcov3(var_patches[l]))
    except:
        pass
    opt_ma = ma.masked_array(opt, mask=np.where(opt<=0, True, False))


# %% Plot
plt.figure(figsize=(10,8))

plt.plot(np.sqrt(var_patches[:,0,0]), label= "sqrt(\"cosmic+noise variance\"), low noise patch", lw=2, ls="-", color=CB_color_cycle[0])
plt.plot(np.sqrt(var_patches[:,1,1]), label= "sqrt(\"cosmic+noise variance\"), high noise patch", lw=2,ls="-", color=CB_color_cycle[1])
plt.plot(np.sqrt(opt_ma), label='sqrt(minimalcov(\"cosmic+noise variance\"))', alpha=0.5, lw=2, ls="-", color=CB_color_cycle[2])
plt.plot(np.sqrt(var_FSma), label="variance, full sky", lw=2,ls="-", color=CB_color_cycle[3])


plt.plot(var_patches_NS[:,0,0], label = "noise, low noise patch", lw=2, ls="--", color=CB_color_cycle[0])
plt.plot(var_patches_NS[:,1,1], label = "noise, high noise patch", lw=2, ls="--", color=CB_color_cycle[1])
plt.plot(opt_NSma, label='noise only, minimalcov(minimalpowerspectra)', alpha=0.5, lw=2, ls="--", color=CB_color_cycle[2])
plt.plot(cov_min_NS_FS, label='noise only, full sky', lw=2, ls="--", color=CB_color_cycle[3])
plt.xscale('log')
plt.yscale('log')


plt.plot(spectrum_trth, label="Planck EE spectrum", lw=1, ls="-.", color='black')
plt.plot(cov_min[0], label = "D_l, low noise patch", lw=1, ls="-.", color=CB_color_cycle[0])
plt.plot(cov_min[1], label = "D_l, high noise patch", lw=1, ls="-.", color=CB_color_cycle[1])
# plt.plot(var_FSma, label = "D_l, all sky patch", lw=1, ls="-.", color=CB_color_cycle[3])
# plt.plot(opt_3ma, label='D_l, minimalcov(minimalpowerspectra)', alpha=0.5, lw=3, color=CB_color_cycle[2])


plt.xlabel("Multipole")
plt.ylabel("Powerspectrum")
plt.xlim((4e1,3e3))
plt.ylim((1e-1,1e8))


plt.legend(loc='upper left')
# plt.text(9e2, 1e2, "Noise+signal", fontsize=20)
plt.title("Combining patches which have Noise + Signal")
plt.savefig('analyticpatching-noise+signal.jpg')


# %%
plt.plot((var_FSma-opt_ma)/opt_ma, label='Improvement when patching', lw=2, ls="-", color=CB_color_cycle[0])
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left')
# plt.xlim((1e3,2.5e3))
plt.ylim((-1,2e1))


# %%
"""
                                    ##########################################
                                    ##########################################
                                                    Backup
                                    ##########################################
                                    ##########################################
"""


# %% Build toy-patches

C_p1 = C_lS + C_lNana1
C_p2 = C_lS + C_lNana2
C_p3 = C_lS + C_lNana3

cov_min_1ana = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_p1[1:]))))
cov_min_2ana = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_p2[1:]))))
cov_min_3ana = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_p3[1:]))))


cov_min_1anaNONOISE = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_lS[1:] + 0.00001*C_lNana1[1:]))))
cov_min_2anaNONOISE = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_lS[1:] + 0.00001*C_lNana2[1:]))))

cov_min_1anaNOSIGNAL = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_lNana1[1:]))))
cov_min_2anaNOSIGNAL = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_lNana2[1:]))))
cov_min_3anaNOSIGNAL = np.concatenate(([0],np.nan_to_num(calculate_minimalcov2(C_lNana3[1:]))))

C_fullana = np.zeros((lmax+1,2,2), float)
C_fullana[:,0,0] = np.array([(2*cov_min_1ana[l] * cov_min_1ana[l])/((2*l+1)*fsky_1) for l in range(C_fullana.shape[0])])
C_fullana[:,1,1] = np.array([(2*cov_min_2ana[l] * cov_min_2ana[l])/((2*l+1)*fsky_2) for l in range(C_fullana.shape[0])])

C_allsky = np.array(
    [(2*cov_min_3ana[l] * cov_min_3ana[l])/((2*l+1)*0.96)
    for l in range(C_fullana.shape[0])])

VarC_allsky_ma = ma.masked_array(C_allsky, mask=np.where(C_allsky<=0, True, False))


# %% Idea 1: patched cov_{min}_l = sum_i weights_il * C_{full}_l, with weights_il = cov(c,c)^{-1}1/(1^Tcov(c,c)^{-1}1)
weights_1 = np.zeros(C_fullana.shape[:-1])
for l in range(C_fullana.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_fullana[l,:,:]))
    except:
        pass
opt_1 = np.array(
    [weights_1[l] @ (np.array([cov_min_1ana, cov_min_2ana]))[:,l] for l in range(C_fullana.shape[0])])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Combine on alm level
import healpy as hp
alm1 = hp.synalm(cov_min_1ana)
alm2 = hp.synalm(cov_min_2ana)
weights_1 = np.zeros((C_fullana.shape[0], C_fullana.shape[1]))

# %%
for l in range(C_fullana.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_fullana[l,:,:]))
    except:
        pass
alm_opt_1 = np.zeros_like(alm1)
for idx in range(alm_opt_1.shape[0]):
    l, m = hp.Alm.getlm(C_fullana.shape[0]-1, idx)
    alm_opt_1[idx] = weights_1[l,0]*alm1[idx] + weights_1[l,1]*alm2[idx]

# %%
opt_12 = hp.alm2cl(alm_opt_1)
opt_12ma = ma.masked_array(opt_12, mask=np.where(opt_12<=0, True, False))

# %%

nalm = hp.synalm(C_lNana1[:,0,0])
nmap = hp.alm2map(nalm,1024)
hp.mollview(nmap)

# %%
plt.hist(nmap, bins=100)
plt.show()

# %%
raw = np.nan_to_num(C_lN["EE"].T[2:] + C_lS[2:] + C_lF[2:])
cosvar = 2*(raw * raw)[:,5,5]*[1/(2*l+1) for l in range(1, 3000)]
print(cosvar.shape)
plt.plot(cosvar)
plt.plot(raw[:,5,5])
plt.xscale('log')
plt.yscale('log')
noisevarmask = np.where(noise_level<1*1e-9,True, False)


# %%
hp.mollview(noise* pmask["217"] * noisevarmask, norm='hist')



# %%
gb = gauss_beam(noiselevel[0]*0.000290888, lmax)
ll = np.arange(0,lmax+1,1)
plt.plot(gb[:,1]* ll*(ll+1))


# %% Covmin



for l in range(lmax+1):
    # C_lS[row,col,l] = 400*np.array([1,1,1])*np.sin(l*np.pi*2/10)**2
    C_lF[l,row,col] = [
        1*np.random.randint(2000-60*(l+1), 2000-59*l),
        2*np.random.randint(2000-60*(l+1), 2000-59*l),
        3*np.random.randint(2000-60*(l+1), 2000-59*l)]

    a = 1*np.random.randint(2000-60*(l+1), 2000-59*l)
    C_lF[l,0,1] = a
    C_lF[l,1,0] = a

    a = 2*np.random.randint(2000-60*(l+1), 2000-59*l)
    C_lF[l,0,2] = a
    C_lF[l,2,0] = a

    a = 6*np.random.randint(2000-60*(l+1), 2000-59*l)
    C_lF[l,1,2] = a
    C_lF[l,2,1] = a

    C_lN[l,row,col] = [
        6*np.random.randint(10+l*(l+1+l)*1.2, 20+l*(l+1+l)*1.5),
        1*np.random.randint(10+l*(l+1+l)*1.2, 20+l*(l+1+l)*1.5),
        1*np.random.randint(10+l*(l+1+l)*1.2, 20+l*(l+1+l)*1.5)]

    # np.fill_diagonal(a[:,:,l], np.random.randint(1,10))
cov_min = calculate_minimalcov(C_lS, C_lF, C_lN)
idx=0
plt.plot(C_lS[:,idx,idx] + C_lF[:,idx,idx] + C_lN[:,idx,idx], label="data0")
idx=1
plt.plot(C_lS[:,idx,idx] + C_lF[:,idx,idx] + C_lN[:,idx,idx], label="data1")
idx=2
plt.plot(C_lS[:,idx,idx] + C_lF[:,idx,idx] + C_lN[:,idx,idx], label="data2")

plt.plot(cov_min, label='minimal')
plt.plot(C_lS[:,0,0], label='signal0')
plt.plot(C_lF[:,0,0], label='foreground')
plt.plot(C_lN[:,0,0], label='noise')
plt.legend()

plt.show()

# %%
f = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/map/frequency/HFI_SkyMap_217-field_2048_R3.01_full.fits"
import healpy as hp
noise_level = hp.read_map(f, field=7)

# %%
import functools
def _read(mask_path, mask_filename):
    return {FREQ: hp.read_map(
            '{mask_path}{mask_filename}'
            .format(
                mask_path = mask_path,
                mask_filename = mask_filename))
                for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
            }
def _multi(a,b):
    return a*b
pmask_filename = ["gmaskP_apodized_0_2048.fits.gz", "psmaskP_2048.fits.gz"]
pmasks = [_read("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/mask/", a) for a in pmask_filename]
pmask = {FREQ: functools.reduce(_multi, [a[FREQ] for a in pmasks])
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
        }


# %%
hp.mollview(pmask["217"], norm='hist')
np.sum(pmask["217"]/len(pmask["217"]))
# %%
hp.mollview(noise_level* pmask["217"], norm='hist')
plt.show()

# %%
print(
    np.mean(noise_level), "1" ,"\n",
    np.mean(noise_level* pmask["217"]), np.sum(pmask["217"]/len(pmask["217"])), "\n",
    np.mean(noise_level* pmask["217"] * noisevarmask), np.sum((pmask["217"]*noisevarmask)/len(pmask["217"])), "\n"
)