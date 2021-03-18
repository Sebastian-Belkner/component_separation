#!/usr/local/bin/python

# %% Load packages and helper functions
"""
run.py: script for executing main functionality of component_separation

"""

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

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/config.json', "r") as f:
    cf = json.load(f)


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
    
    fname = io.make_filenamestring(cf)
    inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/tmp/spectrum/scaled"+fname
    spectrum = io.load_spectrum(inpath_name, fname)
    return spectrum


# %% set 'fixed' parameters

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

# %% load truth signal
signal = pd.read_csv(
        "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)
spectrum_trth = signal["Planck-"+"EE"][:shape[0]+1].to_numpy()
plt.xlabel('multipole')
plt.ylabel('powerspectrum')
plt.plot(spectrum_trth, label= 'Planck EE-spectrum')
plt.legend()

# %% Build toy-cl
C_lS = np.zeros_like(C_lN["EE"].T, float)
row, col = np.diag_indices(C_lS[0,:,:].shape[0])
c = np.ones((3001,7,7))
C_lS =  (np.ones((7,7,3001))* spectrum_trth).T
C_lF = np.zeros_like(C_lN["EE"].T, float)


# %% plot noise+signal
plt.figure(figsize=(8,6))
for n in range(C_lN["EE"].shape[1]):
    plt.plot(C_lN["EE"].T[:,n,n] + C_lS[:,n,n], label='{} Channel'.format(PLANCKMAPFREQ[n]))
plt.title("Noise + signal spectra")
plt.xscale('log')
plt.yscale('log')
plt.xlim((10,lmax))
plt.legend()
plt.ylim((0,1e7))
plt.show()


# %%
C_lN1 = C_lN["EE"].T[2:] * 1./3.
C_lN2 = C_lN["EE"].T[2:] * 2./3.
C_lN3 = C_lN["EE"].T[2:]
C_p1 = C_lS[2:] + C_lF[2:] + C_lN1
C_p2 = C_lS[2:] + C_lF[2:] + C_lN2
C_p3 = C_lS[2:] + C_lF[2:] + C_lN3
cov_min_1 = np.nan_to_num(calculate_minimalcov(C_lN1, C_lS[2:], C_lF[2:]))
cov_min_2 = np.nan_to_num(calculate_minimalcov(C_lN2, C_lS[2:], C_lF[2:]))
cov_min_3 = np.nan_to_num(calculate_minimalcov(C_lN3, C_lS[2:], C_lF[2:]))

C_full = np.zeros((lmax-1,2,2), float)
C_full[:,0,0] = cov_min_1
C_full[:,1,1] = cov_min_2
C_full[:,1,0] = C_lS[2:,0,0]
C_full[:,0,1] = C_lS[2:,0,0]


# %% Idea 1: patched cov_{min}_l = sum_i weights_il * C_{full}_l, with weights_il = cov(c,c)^{-1}1/(1^Tcov(c,c)^{-1}1)
weights_1 = np.zeros((C_full.shape[0], C_full.shape[1]))
for l in range(C_full.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(np.cov(C_full[l,:,:])))
    except:
        pass
opt_1 = np.array([weights_1[l] @ np.array([cov_min_1, cov_min_2])[:,l] for l in range(lmax-1)])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Idea 2: cov_{min} =  calculate_minimalcov(cov(Cp1_l, Cp2_l))
opt_2 = np.zeros((C_full.shape[0]))
for l in range(C_full.shape[0]):
    try:
        opt_2[l] = np.nan_to_num(
            calculate_minimalcov2(C_full[l,:,:]))
    except:
        pass
opt_2ma = ma.masked_array(opt_2, mask=np.where(opt_2<=0, True, False))

# %% Idea 3: use var(c) = 2*C_l^2/(nmodes*f_sky) as weights
weights_3 = np.zeros((C_full.shape[0], C_full.shape[1]))
for l in range(C_full.shape[0]):
    try:
        # np.linalg.inv()
        weights_3[l] = calculate_weights(
            np.linalg.inv(
                2* C_full[l,:,:] @ C_full[l,:,:])/(2*(l+1)*np.array([0.25,0.75])))
    except:
        pass
opt_3 = np.array([weights_3[l] @ np.array([cov_min_1, cov_min_2])[:,l] for l in range(lmax-1)])
opt_3ma = ma.masked_array(opt_3, mask=np.where(opt_3<=0, True, False))


# %% Idea 4: calculate_minimalcov(2*C_l^2/(nmodes*f_sky))

opt_4 = np.zeros((C_full.shape[0]))
for l in range(C_full.shape[0]):
    try:
        opt_4[l] = np.nan_to_num(
            calculate_minimalcov2(
                ((2* C_full[l] @ C_full[l])/(2*(l+1))*np.array([0.25,0.75])),
               ))
    except:
        pass
opt_4ma = ma.masked_array(opt_4, mask=np.where(opt_4<=0, True, False))


# %% plot the combinations
plt.figure(figsize=(10,8))
plt.plot(opt_1ma, label = 'Idea1, weights*powerspectra', alpha=0.5, lw=3)
plt.plot(opt_2ma, label='Idea2, minimalcov(minimalpowerspectra)', alpha=0.5, ls='--', lw=3)
plt.plot(opt_3ma, label='Idea3, 2*C_l^2/(nmodes*f_sky) as weights', alpha=0.5, lw=3)
plt.plot(opt_4ma, label='Idea4, minimalcov(2*C_l^2/(nmodes*f_sky))', alpha=0.5, ls='-.', lw=3)
plt.plot(spectrum_trth, label="Planck EE spectrum")
# plt.plot(cov_min_patched, label = 'patched 4/5 + 1/5 noise', alpha=0.5)
plt.plot(cov_min_1, label = "low noise patch", lw=1)
plt.plot(cov_min_2, label = "high noise patch", lw=1)
plt.plot(cov_min_3, label = "all noise", lw=1)
plt.xscale('log')
plt.yscale('log')
# plt.plot(C_p1[:,5,5])
# plt.plot(C_p2[:,5,5]
# plt.ylim((1e1,1e2))
plt.xlim((6e2,2e3))
plt.legend()



# %%
plt.plot(opt_4ma)

# %% calculate minimal powerspectrum
cov_min = calculate_minimalcov(C_lN["EE"].T[2:], C_lS[2:], C_lF[2:])
cov_min_k = calculate_minimalcov(C_lN["EE"].T[2:]/5., C_lS[2:], C_lF[2:])
cov_min_s = calculate_minimalcov(C_lS[2:], C_lF[2:], C_lN["EE"].T[2:]*0.000001)

# C_full = np.concatenate((
#     np.concatenate((C_p1, np.zeros_like(C_p1)), axis=1),
#     np.concatenate((np.zeros_like(C_p1), C_p2), axis=1)), axis=2)

C_full = np.concatenate((
    np.concatenate((C_p1, C_lS[2:]), axis=1),
    np.concatenate((C_lS[2:], C_p2), axis=1)), axis=2)

cov_min_patched = calculate_minimalcov(C_full)

# %%

cov_min_patched = calculate_minimalcov(C_full)
# plt.figure(figsize=(8,6))
plt.plot(cov_min, label='full noise')
plt.plot(cov_min_s, label= 'signal only', alpha=0.5)
plt.plot(cov_min_k, label='1/5 noise', alpha=0.5)
plt.plot(spectrum_trth, label= 'Planck EE-spectrum')
plt.plot(cov_min_patched, label = 'patched 4/5 + 1/5 noise', alpha=0.5)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('multipole')
plt.ylabel('powerspectrum')
plt.xlim((10,lmax))
plt.legend()
# plt.ylim((1e-1,1e3))
plt.show()


# plt.plot(((cov_min_s-cov_min)/(cov_min_s-cov_min_k))[50:])

# %%
raw = np.nan_to_num(C_lN["EE"].T[2:] + C_lS[2:] + C_lF[2:])
cosvar = 2*(raw * raw)[:,5,5]*[1/(2*l+1) for l in range(1, 3000)]
print(cosvar.shape)
plt.plot(cosvar)
plt.plot(raw[:,5,5])
plt.xscale('log')
plt.yscale('log')

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
noisevarmask = np.where(noise_level<1*1e-9,True, False)

# %%
print(
    np.mean(noise_level), "1" ,"\n",
    np.mean(noise_level* pmask["217"]), np.sum(pmask["217"]/len(pmask["217"])), "\n",
    np.mean(noise_level* pmask["217"] * noisevarmask), np.sum((pmask["217"]*noisevarmask)/len(pmask["217"])), "\n"
)


# %%
hp.mollview(noise_level* pmask["217"] * noisevarmask, norm='hist')


# %%
weights = calculate_weights(np.linalg.inv(C_lS[2:] + C_lF[2:] + C_lN["EE"].T[2:]))
for n in range(C_lN["EE"].shape[1]):
    plt.plot(weights[:,n], label='{} Channel'.format(PLANCKMAPFREQ[n]))
plt.xlabel("Multipole l")
plt.ylabel("weight")
# plt.xscale('log', base=2)
plt.ylim((-0.1,1.0))
plt.legend()
plt.savefig('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/weights.jpg')


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

# %% Weights
weights = calculate_weights(np.linalg.inv(C_lS + C_lF + C_lN))
plt.plot(weights[:,0], label='data0')
plt.plot(weights[:,1], label='data1')
plt.plot(weights[:,2], label='data2')
plt.ylim((-1,1))
plt.legend()



# old
# 
# # %% Noisemaps
# noiselevel = np.array([5,6,7])
# ll = np.arange(1,lmax+2,1)

# noise = np.array(
#     [gauss_beam(noiselevel[0]*0.000290888, lmax)[:,1]*ll*(ll+1),
#     gauss_beam(noiselevel[1]*0.000290888, lmax)[:,1]*ll*(ll+1),
#     gauss_beam(noiselevel[2]*0.000290888, lmax)[:,1]*ll*(ll+1)])
# noise = create_noisespectrum(arcmin=noiselevel)