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

import matplotlib.pyplot as plt
import numpy as np



uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"
lmax = 30


def calculate_weights(cov):
    # print(cov)
    elaw = np.ones(len([dum for dum in range(len(cov))]))
    weights = np.array([
        (cov[:,:,l] @ elaw) / (elaw.T @ cov[:,:,l] @ elaw)
            if cov[:,:,l] is not None else np.array([np.nan for n in range(len(elaw))])
            for l in range(lmax)])

    return weights
    

def calculate_analytic_minimalcov(C_lS: np.ndarray, C_lF: np.ndarray, C_lN: np.ndarray) -> np.array:
    """Returns the minimal covariance using inverse variance weighting, i.e. calculates
    :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

    Args:
        C_lS (np.ndarray): An array of auto- and cross-covariance matrices of the CMB for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lF (np.ndarray): An array of auto- and cross-covariance matrices of the superposition of Foregrounds for all instruments, its dimension is [Nspec,Nspec,lmax]
        C_lN (np.ndarray): An array of auto- and cross-covariance matrices of the Noise for all instruments, its dimension is [Nspec,Nspec,lmax]
    """
    def is_invertible(a, l):
        truth = a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
        if not truth:
            print('{} not invertible: {}'.format(l, a) )
        return truth

    elaw = np.ones(C_lS.shape[0])
    cov_minimal = 1/np.array([elaw.T @ np.linalg.inv(C_lS[:,:,l] + C_lF[:,:,l] + C_lN[:,:,l]) @ elaw
                    if is_invertible(C_lS[:,:,l] + C_lF[:,:,l] + C_lN[:,:,l], l) 
                    else None
                    for l in range(lmax)])
    return cov_minimal


if __name__ == '__main__':
    n_cha = 3
    a = np.zeros((n_cha,n_cha,lmax+1), float)
    C_lS = 1000*np.ones((n_cha,n_cha,lmax+1), float)
    C_lF = np.zeros((n_cha,n_cha,lmax+1), float)
    C_lN = np.zeros((n_cha,n_cha,lmax+1), float)

    row, col = np.diag_indices(a[:,:,0].shape[0])
    for l in range(lmax+1):
        # C_lS[row,col,l] = 400*np.array([1,1,1])*np.sin(l*np.pi*2/10)**2
        C_lF[row,col,l] = [
            1*np.random.randint(2000-60*(l+1), 2000-58*l),
            2*np.random.randint(2000-60*(l+1), 2000-58*l),
            3*np.random.randint(2000-60*(l+1), 2000-58*l)]

        a = 1*np.random.randint(2000-60*(l+1), 2000-59*l)
        C_lF[0,1,l] = a
        C_lF[1,0,l] = a

        a = 2*np.random.randint(2000-60*(l+1), 2000-59*l)
        C_lF[0,2,l] = a
        C_lF[2,0,l] = a

        C_lF[1,2,l] = 6*np.random.randint(2000-60*(l+1), 2000-59*l)
        C_lF[2,1,l] = 6*np.random.randint(2000-60*(l+1), 2000-59*l)

        C_lN[row,col,l] = [
            2*np.random.randint(10+l*(l+1+l)*1.2, 20+l*(l+1+l)*1.5),
            4*np.random.randint(10+l*(l+1+l)*1.2, 20+l*(l+1+l)*1.5),
            2*np.random.randint(10+l*(l+1+l)*1.2, 20+l*(l+1+l)*1.5)]
        # np.fill_diagonal(a[:,:,l], np.random.randint(1,10))
    cov_min = calculate_analytic_minimalcov(C_lS, C_lF, C_lN)

    
# %%
print(C_lS[:,:,0])
idx=0
plt.plot(C_lS[idx,idx,:] + C_lF[idx,idx,:] + C_lN[idx,idx,:], label="data0")
idx=1
plt.plot(C_lS[idx,idx,:] + C_lF[idx,idx,:] + C_lN[idx,idx,:], label="data1")
idx=2
plt.plot(C_lS[idx,idx,:] + C_lF[idx,idx,:] + C_lN[idx,idx,:], label="data2")
    
plt.plot(cov_min, label='minimal')
plt.plot(C_lS[0,0,:], label='signal0')
plt.plot(C_lF[0,0,:], label='foreground')
plt.plot(C_lN[0,0,:], label='noise')
plt.legend()
plt.show()

# %%
weights = calculate_weights(
    np.array([np.linalg.inv((C_lS + C_lF + C_lN)[:,:,l]) for l in range(lmax)]).reshape((3,3,lmax))
)
for w, lab in zip(weights.T, ["030", "044", "070"]):
    plt.plot(w, label=lab)
    plt.ylim((0,1))
plt.legend()


# %%
