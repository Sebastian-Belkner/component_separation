#!/usr/local/bin/python

# %%
"""
run.py: script for executing main functionality of component_separation

"""

__author__ = "S. Belkner"


import json
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

import component_separation.powspec as pw
import healpy as hp

import matplotlib.pyplot as plt
import numpy as np

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"
lmax = 3000
n_cha = 3
shape = (lmax, n_cha, n_cha)


def create_noisespectrum(arcmin, fwhm=np.array([2,10,27])):
    fwhm = 0.000290888 * fwhm
    rad = arcmin[np.newaxis].T * np.ones(lmax)  * 0.000290888
    ll = np.arange(0,lmax,1)
    def _spec(rad):
        return (2*0.000290888/(2.728*1e6))**2 * np.exp(ll*(ll+1)*rad**2/(8*np.log(2)))
    return _spec(rad)


def create_covmatrix(spectrum):
    # for starter, i assume its noise only

    C_lN = np.zeros(shape, float)
    for l in range(lmax):
        C_lN[l,row,col] = spectrum[:,l]
    # NSIDE = 64
    # n = hp.nside2npix(NSIDE)
    # data = np.random.normal(0, noiselevel, size=n)
    # hp.mollview(data)
    # plt.show()
    # alms = hp.map2alm(data)
    # cl = hp.alm2cl(alms)
    return C_lN


def calculate_weights(cov):
    elaw = np.ones(cov.shape[1])
    weights = (cov @ elaw) / (elaw @ cov @ elaw)[np.newaxis].T
    return weights
    

def calculate_minimalcov(C_lS: np.ndarray, C_lF: np.ndarray, C_lN: np.ndarray) -> np.array:
    """Returns the minimal covariance using inverse variance weighting, i.e. calculates
    :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

    Args:
        C_lS (np.ndarray): An array of auto- and cross-covariance matrices of the CMB for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lF (np.ndarray): An array of auto- and cross-covariance matrices of the superposition of Foregrounds for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lN (np.ndarray): An array of auto- and cross-covariance matrices of the Noise for all instruments, its dimension is [Nspec,Nspec,lmax]
    """

    elaw = np.ones(C_lS.shape[1])
    cov_minimal = (elaw @ np.linalg.inv(C_lS + C_lF + C_lN) @ elaw)
    return cov_minimal


# %% Noisemaps
noiselevel = np.array([5,6,7])
noise = create_noisespectrum(arcmin=noiselevel)
C_lN = create_covmatrix(noise)

# noise_cl = create_covmatrix(50)
plt.title("Noisespectra")
plt.plot(C_lN[:,0,0], label='{} arcmin Noiselevel'.format(noiselevel[0]))
plt.plot(C_lN[:,1,1], label='{} arcmin Noiselevel'.format(noiselevel[1]))
plt.plot(C_lN[:,2,2], label='{} arcmin Noiselevel'.format(noiselevel[2]))
plt.xlabel("Multipole l")
plt.ylabel("Power")
plt.legend()
plt.yscale('log')
plt.savefig('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/noisespectra.jpg')
plt.show()


# %%
C_lS = np.zeros_like(C_lN, float)
C_lF = np.zeros_like(C_lN, float)
cov_min = calculate_minimalcov(C_lS, C_lF, C_lN)
plt.plot(cov_min, label='minimal')



# %%
weights = calculate_weights(np.linalg.inv(C_lS + C_lF + C_lN))
plt.plot(weights[:,0], label='Channel with noiselevel: {} arcmin'.format(noiselevel[0]))
plt.plot(weights[:,1], label='Channel with noiselevel: {} arcmin'.format(noiselevel[1]))
plt.plot(weights[:,2], label='Channel with noiselevel: {} arcmin'.format(noiselevel[2]))
plt.xlabel("Multipole l")
plt.ylabel("weight")
# plt.xscale('log', base=2)
plt.ylim((0,1))
plt.legend()
plt.savefig('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/weights.jpg')







# %% Covmin
n_cha = 3
shape = (lmax+1, n_cha,n_cha)
a = np.zeros(shape, float)
C_lS = 1000*np.ones(shape, float)
C_lF = np.zeros(shape, float)
C_lN = np.zeros(shape, float)

row, col = np.diag_indices(a[0,:,:].shape[0])
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