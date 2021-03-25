#!/usr/local/bin/python

"""
run.py: 
    Script for calculating the gain of the optimal powerspectrum. 
    Section I takes real data,
    Section II takes fake data,
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
n_cha = 3
shape = (lmax, n_cha, n_cha)

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
spectrum_trth = signal["Planck-"+"EE"][:shape[0]+1].to_numpy()


def std_dev_binned(d, bins):
    print(d.shape)
    val = np.nan_to_num(d.compressed())
    linsp = np.where(d.mask==False)[0]
    print(linsp.shape, val.shape)
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

def create_noisespectrum(arcmin, fwhm=np.array([2,10,27])):
    """Analog to gauss_beam

    Args:
        arcmin ([type]): [description]
        fwhm ([type], optional): [description]. Defaults to np.array([2,10,27]).

    Returns:
        [type]: [description]
    """    
    fwhm = 0.000290888 * fwhm
    rad = arcmin[np.newaxis].T * np.ones(lmax)  * 0.000290888
    ll = np.arange(0,lmax,1)
    def _spec(rad):
        return (2*0.000290888/(2.728*1e6))**2 * np.exp(ll*(ll+1)*rad**2/(8*np.log(2)))
    return _spec(rad)


def gauss_beam(fwhm, lmax):
    return (fwhm/(2.728))**2* 1/ hp.gauss_beam(fwhm, lmax=lmax, pol=True)


def create_covmatrix(spectrum):
    """Creates auto and cross covariance matrix, with noise only, so no cross.

    Args:
        spectrum ([type]): [description]

    Returns:
        [type]: [description]
    """    
    row, col = np.diag_indices(a[0,:,:].shape[0])
    C_lN = np.zeros(shape, float)
    for l in range(lmax):
        C_lN[l,row,col] = spectrum[:,l]
    return C_lN


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

n_cha = 7
shape = (lmax, n_cha,n_cha)
a = np.zeros(shape, float)
C_lF = np.zeros(shape, float)
C_lN = np.zeros(shape, float)

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
noise_spectrum = load_empiric_noise_aac_covmatrix()
C_lN  = pw.build_covmatrices(noise_spectrum, lmax, freqfilter, specfilter)



"""
SECTION 1: Take real data
""" 

# %% Load Spectra, currently this includes "smallpatch", "largepatch", and fullsky. All maps are however smica-masked.
with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/component_separation/draw/draw.json', "r") as f:
    dcf = json.load(f)
freqfilter = dcf['pa']["freqfilter"]
specfilter = dcf['pa']["specfilter"]
freqcomb =  [
    "{}-{}".format(FREQ,FREQ2)
        for FREQ in PLANCKMAPFREQ
        if FREQ not in freqfilter
        for FREQ2 in PLANCKMAPFREQ
        if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]

dc = dcf["plot"]["spectrum"]
def _inpathname(freqc,spec, fname):
    return  "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+dc["indir_root"]+dc["indir_rel"]+spec+freqc+"-"+dc["in_desc"]+fname

dcf["pa"]["mskset"] =  "mask-spatch-smica"
fname = io.make_filenamestring(dcf)
speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
sspectrum = {freqc: {
    spec: np.array(io.load_cl(_inpathname(freqc,spec, fname)))
    for spec in speccs}  
    for freqc in freqcomb}

dcf["pa"]["mskset"] =  "mask-lpatch-smica"
fname = io.make_filenamestring(dcf)
lspectrum = {freqc: {
    spec: np.array(io.load_cl(_inpathname(freqc,spec, fname)))
    for spec in speccs}  
    for freqc in freqcomb}

dcf["pa"]["mskset"] =  "smica"
fname = io.make_filenamestring(dcf)
dc = dcf["plot"]["spectrum"]
inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
allspectrum = io.load_spectrum(inpath_name, fname)
allC_l  = pw.build_covmatrices(allspectrum, lmax, freqfilter, specfilter)


# %% Build Auto and Cross covariance Matrix
sC_l = np.nan_to_num(pw.build_covmatrices(sspectrum, lmax, freqfilter, specfilter)["EE"].T)[2:2000]
lC_l = np.nan_to_num(pw.build_covmatrices(lspectrum, lmax, freqfilter, specfilter)["EE"].T)[2:2000]
allC_l = np.nan_to_num(pw.build_covmatrices(allspectrum, lmax, freqfilter, specfilter)["EE"].T)[2:2000]


# %% Calculate Optimal Powerspectrum
cov_min_1 = np.nan_to_num(calculate_minimalcov2(sC_l))
cov_min_2 = np.nan_to_num(calculate_minimalcov2(lC_l))
cov_min_3 = np.nan_to_num(calculate_minimalcov2(allC_l))

cov_min_1ma = ma.masked_array(cov_min_1, mask=np.where(cov_min_1<=0, True, False))
cov_min_2ma = ma.masked_array(cov_min_2, mask=np.where(cov_min_2<=0, True, False))


# %% Build weights for Inverse Variance Weighting
C_full = np.zeros((allC_l.shape[0],2,2), float)
C_full[:,0,0] = np.array(
    [(2*cov_min_1[l] * cov_min_1[l])/((2*l+1)*0.23)
    for l in range(allC_l.shape[0])])
C_full[:,1,1] = np.array(
    [(2*cov_min_2[l] * cov_min_2[l])/((2*l+1)*0.71) for l in range(allC_l.shape[0])])
C_full[:,1,0] = 0*np.array([(2*cov_min_1[l] * cov_min_2[l])/((2*l+1)*np.sqrt(0.23*0.71))
        for l in range(allC_l.shape[0])])
C_full[:,0,1] = 0*np.array([(2*cov_min_2[l] * cov_min_1[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(allC_l.shape[0])])
print(C_full[10])


# %% Calculate combined powerspectrum via inverse variance weighting
weights_1 = np.zeros((C_full.shape[0], C_full.shape[1]))
for l in range(C_full.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_full[l,:,:]))
    except:
        pass
opt_1 = np.array([weights_1[l] @ np.array([cov_min_1, cov_min_2])[:,l] for l in range(C_full.shape[0])])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Calculate optimal Powerspectrum
opt_2 = np.zeros((C_full.shape[0]))
for l in range(C_full.shape[0]):
    try:
        opt_2[l] = np.nan_to_num(
            calculate_minimalcov2(np.cov(np.array([
                [cov_min_1[l], 0],
                [0, cov_min_2[l]]]))))
    except:
        pass
opt_2ma = ma.masked_array(opt_2, mask=np.where(opt_2<=0, True, False))


# %% Plot
bins = np.logspace(np.log(1)/np.log(base), np.log(C_full.shape[0]+1)/np.log(base), nbins, base=base)
plt.figure(figsize=(8,6))
print("opt1_ma:")
mean, std, _ = std_dev_binned(opt_1ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="Combined spectrum, weights*powerspectra",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    markersize=10,
    # color=CB_color_cycle[idx],
    alpha=0.9)

# print("opt2_ma:")
# mean, std, _ = std_dev_binned(opt_2ma, bins)
# print(mean)
# plt.errorbar(
#     (_[1:] + _[:-1])/2,
#     mean,
#     yerr=std,
#     label="Combined spectrum using minmalcov",# + freqc,
#     capsize=3,
#     elinewidth=2,
#     fmt='x',
#     markersize=10,
#     # color=CB_color_cycle[idx],
#     alpha=0.9)

print("cov_min_1ma")
mean, std, _ = std_dev_binned(cov_min_1ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="low noise patch",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    # color=CB_color_cycle[idx],
    alpha=0.9)

print("cov_min_2ma")
mean, std, _ = std_dev_binned(cov_min_2ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="high noise patch",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    # color=CB_color_cycle[idx],
    alpha=0.9)

plt.plot(spectrum_trth, label="Planck EE spectrum")
# plt.plot(cov_min_3, label = "all sky", lw=2, alpha=0.7)
# plt.plot(C_p1[:,5,5])
# plt.plot(C_p2[:,5,5]


plt.xscale("log", nonpositive='clip')
plt.yscale("log", nonpositive='clip')
plt.xlabel("Multipole l")
plt.ylabel("Powerspectrum")
plt.grid(which='both', axis='x')
plt.grid(which='major', axis='y')

plt.xlim((3e1,2e3))
plt.ylim((1e-1,1e2))

plt.legend()
plt.savefig('skypatches.jpg')
plt.show()


# %% Look at Variance of the powerspectra
plt.figure(figsize=(8,6))
print("opt1_ma:")
mean, std, _ = std_dev_binned(opt_1ma, bins)
plt.plot(
    (_[1:] + _[:-1])/2,
    std,
    label="Combined spectrum, weights*powerspectra",# + freqc,
    markersize=10,
    alpha=0.9)

print("opt2_ma:")
mean, std, _ = std_dev_binned(opt_2ma, bins)
print(mean)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="Combined spectrum using minmalcov",# + freqc,
    # color=CB_color_cycle[idx],
    alpha=0.9)

print("cov_min_1ma")
mean, std, _ = std_dev_binned(cov_min_1ma, bins)
plt.plot(
    (_[1:] + _[:-1])/2,
    std,
    label="low noise patch",# + freqc,
    # color=CB_color_cycle[idx],
    alpha=0.9)

print("cov_min_2ma")
mean, std, _ = std_dev_binned(cov_min_2ma, bins)
plt.plot(
    (_[1:] + _[:-1])/2,
    std,
    label="high noise patch",# + freqc,
    # color=CB_color_cycle[idx],
    alpha=0.9)

plt.xscale("log", nonpositive='clip')
plt.yscale("log", nonpositive='clip')
plt.xlabel("Multipole l")
plt.ylabel("Variance / C_l")
plt.grid(which='both', axis='x')
plt.grid(which='major', axis='y')

plt.xlim((3e1,2e3))
plt.ylim((1e-1,1e2))
plt.legend()
plt.show()

"""
SECTION 2: Take fake CMB data and noise data from diff maps
""" 


# %% Build toy-cl
C_lS = np.zeros_like(C_lN["EE"].T, float)
row, col = np.diag_indices(C_lS[0,:,:].shape[0])
c = np.ones((3001,7,7))
C_lS =  (np.ones((7,7,3001))* spectrum_trth).T
C_lF = np.zeros_like(C_lN["EE"].T, float)


# %% plot noise+signal
plt.figure(figsize=(8,6))
for n in range(C_lN["EE"].shape[1]):
    plt.plot(C_lN["EE"].T[:,n,n], label='Noise {} Channel'.format(PLANCKMAPFREQ[n]))

plt.plot(C_lS[:,0,0])
plt.title("EE noise andd signal spectra")
plt.xscale('log')
plt.yscale('log')
plt.xlim((10,lmax))
plt.legend()
# plt.ylim((1e-2,1e6))
plt.show()




# %% Build toy-patches
C_lN1 = C_lN["EE"].T[2:] * 1./3.
C_lN2 = C_lN["EE"].T[2:] * 2./3.
C_lN3 = C_lN["EE"].T[2:]
C_p1 = C_lS[2:] + C_lF[2:] + C_lN1
C_p2 = C_lS[2:] + C_lF[2:] + C_lN2
C_p3 = C_lS[2:] + C_lF[2:] + C_lN3
cov_min_1ana = np.nan_to_num(calculate_minimalcov(C_lN1, C_lS[2:], C_lF[2:]))
cov_min_2ana = np.nan_to_num(calculate_minimalcov(C_lN2, C_lS[2:], C_lF[2:]))
cov_min_3ana = np.nan_to_num(calculate_minimalcov(C_lN3, C_lS[2:], C_lF[2:]))

C_fullana = np.zeros((lmax-1,2,2), float)
C_fullana[:,0,0] = np.array([(2*cov_min_1ana[l] * cov_min_1ana[l])/((2*l+1)*0.23) for l in range(C_fullana.shape[0])])
C_fullana[:,1,1] = np.array([(2*cov_min_2ana[l] * cov_min_2ana[l])/((2*l+1)*0.71) for l in range(C_fullana.shape[0])])
C_fullana[:,1,0] = 0*np.array([(2*cov_min_1ana[l] * cov_min_2ana[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(C_fullana.shape[0])])
C_fullana[:,0,1] = 0*np.array([(2*cov_min_2ana[l] * cov_min_1ana[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(C_fullana.shape[0])])



# %% Idea 1: patched cov_{min}_l = sum_i weights_il * C_{full}_l, with weights_il = cov(c,c)^{-1}1/(1^Tcov(c,c)^{-1}1)
weights_1 = np.zeros((C_fullana.shape[0], C_fullana.shape[1]))
for l in range(C_fullana.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_fullana[l,:,:]))
    except:
        pass
opt_1 = np.array([weights_1[l] @ np.array([cov_min_1ana, cov_min_2ana])[:,l] for l in range(C_fullana.shape[0])])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Idea 3: cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_3 = np.zeros((C_fullana.shape[0]))
for l in range(C_fullana.shape[0]):
    try:
        opt_3[l] = np.nan_to_num(
            calculate_minimalcov2(np.array([
                [cov_min_1ana[l], 0],
                [0, cov_min_2ana[l]]])))
    except:
        pass
opt_3ma = ma.masked_array(opt_3, mask=np.where(opt_3<=0, True, False))


# %% Idea 4: cov_{min} =  calculate_minimalcov2(cosmic_variance(Cp1_l, Cp2_l))
opt_4 = np.zeros((C_fullana.shape[0]))
for l in range(C_fullana.shape[0]):
    try:
        opt_4[l] = np.nan_to_num(
            calculate_minimalcov2(C_fullana[l]))
    except:
        pass
opt_4ma = ma.masked_array(opt_4, mask=np.where(opt_4<=0, True, False))



# %% plot the combination
# plt.plot(opt_1ma, label = 'Combined spectrum, weights*powerspectra', alpha=0.5, lw=3)
plt.plot(opt_3ma, label='minimalcov(cosmic_variance)', alpha=0.5, lw=3)
plt.plot(C_fullana[:,0,0], label= "cosmic variance low noise patch")
plt.plot(C_fullana[:,1,1], label= "cosmic variance high noise patch")

plt.plot(opt_4ma, label='minimalcov(powerspectra)', alpha=0.5, lw=3)
plt.plot(spectrum_trth, label="Planck EE spectrum")
plt.plot(cov_min_1ana, label = "Powerspectrum low noise patch", lw=1)
plt.plot(cov_min_2ana, label = "Powerspectrum high noise patch", lw=1)
# plt.plot(cov_min_3ana, label = "all noise", lw=1)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1e-4,1e3))
plt.xlim((2e1,3e3))
plt.legend()


# %% plot the variance of the Spectra

plt.plot(opt_1ma, label = 'Combined spectrum, weights*powerspectra', alpha=0.5, lw=3)
# plt.plot(opt_2ma, label='minimalcov(minimalpowerspectra)', alpha=0.5, ls='--', lw=3)
plt.plot(spectrum_trth, label="Planck EE spectrum")
# plt.plot(cov_min_patched, label = 'patched 4/5 + 1/5 noise', alpha=0.5)
plt.plot(np.var(cov_min_1ana), label = "low noise patch", lw=1)
plt.plot(cov_min_2ana, label = "high noise patch", lw=1)
plt.plot(cov_min_3ana, label = "all noise", lw=1)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1e-1,1e2))
plt.xlim((2e1,2e3))
plt.legend()


"""
SECTION 3: Take no data
""" 


# %% Create Noisespectrum from gaussbeam
noiselevel = np.array([1,2,3,4,5,6,7])
ll = np.arange(1,lmax+2,1)

noise = np.array(
    [gauss_beam(noiselevel[0]*0.000290888, lmax)[:,1]*ll*(ll+1),
    gauss_beam(noiselevel[1]*0.000290888, lmax)[:,1]*ll*(ll+1),
    gauss_beam(noiselevel[2]*0.000290888, lmax)[:,1]*ll*(ll+1)])
noise = create_noisespectrum(arcmin=noiselevel)
C_lNana1 = create_covmatrix(noise)*1e20


noiselevel = np.array([3,4,5,6,7,8,9])
ll = np.arange(1,lmax+2,1)
noise = np.array(
    [gauss_beam(noiselevel[0]*0.000290888, lmax)[:,1]*ll*(ll+1),
    gauss_beam(noiselevel[1]*0.000290888, lmax)[:,1]*ll*(ll+1),
    gauss_beam(noiselevel[2]*0.000290888, lmax)[:,1]*ll*(ll+1)])
noise = create_noisespectrum(arcmin=noiselevel)
C_lNana2 = create_covmatrix(noise)*1e20

# %% Plot Noisespectrum
plt.plot(noise[0,:])
plt.plot(noise[1,:])
plt.plot(noise[2,:])


# %% Build toy-patches
C_lN1 = C_lNana1[:-1]
C_lN2 = C_lNana2[:-1]
C_lN3 = C_lN["EE"].T[2:]
C_p1 = C_lS[2:] + C_lF[2:] + C_lN1
C_p2 = C_lS[2:] + C_lF[2:] + C_lN2
C_p3 = C_lS[2:] + C_lF[2:] + C_lN3
cov_min_1ana = np.nan_to_num(calculate_minimalcov2(C_p1))
cov_min_2ana = np.nan_to_num(calculate_minimalcov2(C_p2))
cov_min_3ana = np.nan_to_num(calculate_minimalcov2(C_lN3))

cov_min_1anaNONOISE = np.nan_to_num(calculate_minimalcov2(C_lS[2:] + C_lF[2:]))
cov_min_2anaNONOISE = np.nan_to_num(calculate_minimalcov2(C_lS[2:] + C_lF[2:]))

C_fullana = np.zeros((lmax-1,2,2), float)
C_fullana[:,0,0] = np.array([(2*cov_min_1ana[l] * cov_min_1ana[l])/((2*l+1)*0.23) for l in range(C_fullana.shape[0])])
C_fullana[:,1,1] = np.array([(2*cov_min_2ana[l] * cov_min_2ana[l])/((2*l+1)*0.71) for l in range(C_fullana.shape[0])])
C_fullana[:,1,0] = 0*np.array([(2*cov_min_1ana[l] * cov_min_2ana[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(C_fullana.shape[0])])
C_fullana[:,0,1] = 0*np.array([(2*cov_min_2ana[l] * cov_min_1ana[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(C_fullana.shape[0])])


# %% Idea 1: patched cov_{min}_l = sum_i weights_il * C_{full}_l, with weights_il = cov(c,c)^{-1}1/(1^Tcov(c,c)^{-1}1)
weights_1 = np.zeros((C_fullana.shape[0], C_fullana.shape[1]))
for l in range(C_fullana.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_fullana[l,:,:]))
    except:
        pass
opt_1 = np.array([weights_1[l] @ np.array([cov_min_1ana, cov_min_2ana])[:,l] for l in range(C_fullana.shape[0])])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Idea 3: cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_3 = np.zeros((C_fullana.shape[0]))
for l in range(C_fullana.shape[0]):
    try:
        opt_3[l] = np.nan_to_num(
            calculate_minimalcov2(np.array([
                [cov_min_1ana[l], cov_min_1anaNONOISE],
                [cov_min_1anaNONOISE, cov_min_2ana[l]]])))
    except:
        print(np.nan_to_num(
            calculate_minimalcov2(np.array([
                [cov_min_1ana[l], cov_min_1anaNONOISE],
                [cov_min_1anaNONOISE, cov_min_2ana[l]]]))))
        pass
opt_3ma = ma.masked_array(opt_3, mask=np.where(opt_3<=0, True, False))


# %% Idea 4: cov_{min} =  calculate_minimalcov2(cosmic_variance(Cp1_l, Cp2_l))
opt_4 = np.zeros((C_fullana.shape[0]))
for l in range(C_fullana.shape[0]):
    try:
        opt_4[l] = np.nan_to_num(
            calculate_minimalcov2(C_fullana[l]))
    except:
        pass
opt_4ma = ma.masked_array(opt_4, mask=np.where(opt_4<=0, True, False))

# %% Plot

plt.figure(figsize=(8,6))
# plt.plot(opt_1ma, label = 'Combined spectrum, weights*powerspectra', alpha=0.5, lw=3)


plt.plot(opt_4ma, label='minimalcov(cosmic_variance)', alpha=0.5, lw=3)
plt.plot(C_fullana[:,0,0], label= "cosmic variance low noise patch")
plt.plot(C_fullana[:,1,1], label= "cosmic variance high noise patch")

plt.plot(spectrum_trth, label="Planck EE spectrum")
# plt.plot(cov_min_patched, label = 'patched 4/5 + 1/5 noise', alpha=0.5)
plt.plot(cov_min_1ana, label = "low noise patch", lw=1)
plt.plot(cov_min_2ana, label = "high noise patch", lw=1)
plt.plot(opt_3ma, label='minimalcov(minimalpowerspectra)', alpha=0.5, lw=3)
# plt.plot(cov_min_3ana, label = "all noise", lw=1)
plt.xscale('log')
plt.yscale('log')
# plt.xlim((6e1,3e3))
# plt.ylim((1e-1,5e1))

plt.legend()

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



"""
Backup
"""

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