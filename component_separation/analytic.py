#!/usr/local/bin/python

"""
run.py: 
    Script for calculating the gain of the optimal powerspectrum. 
    Section III takes no data.
"""


# %% ONLY DO ONCE!
import json
import platform

import component_separation.io as io

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"
with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/config.json', "r") as f:
    cf = json.load(f)
cf[mch]["indir"] = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/"
noisevar_map = io.load_plamap_new(cf, field=7)








# %% Load packages
__author__ = "S. Belkner"

import os
import sys
from typing import Dict, List, Optional, Tuple

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib.patches import Patch
from numpy import inf

import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks

lmax = 3000
detector = ["030", "044", "070", "143", "217"]
npatch = 50
shape = (lmax+1, len(detector), len(detector))

a = np.zeros(shape, float)
C_lF = np.zeros(shape, float)
C_lN = np.zeros(shape, float)

PLANCKSPECTRUM = [p.value for p in list(Plancks)]
PLANCKMAPFREQ = [p.value for p in list(Planckf)]


# %% Load functions
signal = pd.read_csv(
        "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)
spectrum_trth = signal["Planck-"+"EE"][:shape[0]].to_numpy()


def std_dev_binned(d, bins):
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


def create_noisespectrum(dp, fwhm, nside):
    """Analog to gauss_beam

    Args:
        dp ([type]): [description]
        fwhm ([type], optional): [description]. Defaults to ..

    Returns:
        [type]: [description]
    """   
    ll = np.arange(0,lmax+1,1)
    # TODO AMPLITUDE SHOULD be noise level in muK arcmin
    # multiply by area of pixel (in unit rad)
    # hp.nside2pix()
    return dp * 1e6 *  np.exp(ll*(ll+1)*(fwhm*0.000290888)**2/(8*np.log(2)))

def create_noisecl_from_beamf(beamf, dp, freqc):
    TEB_dict = {
        "T": 0,
        "E": 1,
        "B": 2
    }
    LFI_dict = {
        "030": 28,
        "044": 29,
        "070": 30
    }
    ret = 0
    freqs = freqc.split('-')
    hdul = beamf[freqc]
    if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
        ret = 1 / hdul["HFI"][1].data.field(TEB_dict["E"])[:lmax+1]**2
    elif int(freqs[0]) < 100 and int(freqs[1]) < 100:
        b = np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0))
        buff = np.concatenate((
            b[:min(lmax+1, len(b))],
            np.array([np.NaN for n in range(max(0, lmax+1-len(b)))])))
        ret = 1 / buff**2
    return ret * 1e6 * dp



def gauss_beam(fwhm):
    return 1/hp.gauss_beam(fwhm[0], lmax=lmax, pol=True)[:,1]**2


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


def calculate_minimalcov(C_l, frac=1) -> np.array:
    """Returns the minimal covariance using inverse variance weighting, i.e. calculates
    :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

    Args:
        C_l (np.ndarray): An array of auto- and cross-covariance matrices for all instruments, its dimension is [Nspec,Nspec,lmax]

    """
    def isDiag(M):
        i, j = M.shape
        assert i == j 
        test = M.reshape(-1)[:-1].reshape(i-1, j+1)
        return ~np.any(test[:, 1:])

    def invdiagmat(C):
        import copy
        ret = copy.deepcopy(C)
        row, col = np.diag_indices(ret.shape[0])
        ret[row, col] = 1/np.diag(ret)
        return ret

    elaw = np.ones(C_l.shape[-1])*frac
    if isDiag(C_l):
        inv = invdiagmat(C_l)
    else:
        inv = np.linalg.inv(C_l)

    cov_minimal = elaw @ inv @ elaw
    return 1/cov_minimal


def get_beamsize(det):
    beamsizes = {
        "030":32,
        "044":27,
        "070":13,
        "100":10,
        "143":7,
        "217":5,
        "353":5
    }
    return np.array([beamsizes[n] for n in det])


# %% Build toy-cl
C_lS = np.zeros_like(C_lN.T, float)
row, col = np.diag_indices(C_lS[0,:,:].shape[0])
ll = np.arange(0,lmax+1,1)
C_lS =  (np.ones((len(detector),len(detector),3001))* spectrum_trth/(ll*(ll+1))*2*np.pi).T
C_lS[:10,:,:] = np.zeros((10, len(detector), len(detector)))
C_lF = np.zeros_like(C_lN.T, float)


"""
                                    ##########################################
                                    ##########################################
                                                Load Noiselevel info
                                    ##########################################
                                    ##########################################
""" 

# %% Calculate Noiselevel for each patch
def get_noiselevel(data, npatch, freq):
    patch_bounds = np.array(list(range(npatch+1)))/npatch
    mean, binedges = np.histogram(data, bins=np.logspace(np.log10(data.min()),np.log10(data.max()),100000))
    patch_noiselevel = np.zeros((len(patch_bounds)-1))
    buff=0
    buff2=0
    patchidx=0
    # boundaries = np.zeros((npatch))
    noisebuff = 0
    # print('sum(mean): {}'.format(sum(mean)))
    for idx,n in enumerate(mean):
        buff += mean[idx]
        buff2 += mean[idx]
        if buff < patch_bounds[patchidx+1] * len(data):
            noisebuff +=  mean[idx] * (binedges[idx+1]+binedges[idx])/2
        else:
            # boundaries[patchidx] = (binedges[idx+1]+binedges[idx])/2
            patch_noiselevel[patchidx] = noisebuff/buff2
            buff2=0
            patchidx+=1
            noisebuff=0
    # boundaries[-1] = (binedges[-2]+binedges[-1])/2
    patch_noiselevel[-1] = noisebuff/buff2
    # plt.vlines(np.concatenate(([data.min()],boundaries)), ymin=0, ymax=1e6, color='red', alpha=0.5)
    print(freq, patch_noiselevel.shape, np.mean(patch_noiselevel))
    return patch_noiselevel
noise_level = np.array([get_noiselevel(noisevar_map[freq], npatch, freq) for freq in detector]) #["217"]["030", "044", "070", "100", "143", "217", "353"]])

# %% Calculate Noiselevel for Full Sky
def get_noiselevelFS(data):
    # print("mean: {}, min: {}, max: {}".format(data.mean(), data.min(), data.max()))
    return [data.mean()]
noise_level_FS = np.array([get_noiselevelFS(noisevar_map[freq]) for freq in detector]) #["217"]["030", "044", "070", "100", "143", "217", "353"]])


# %% Create Noise Powerspectrum for all patches
fwhm = np.array([get_beamsize(detector) for n in range(npatch)]).T
noise_spectrum = np.zeros((len(detector), npatch,lmax+1))
noise_spectrum_new = np.zeros((len(detector), npatch,lmax+1))
for n in range(noise_spectrum.shape[0]):
    freqc = str(detector[n])+"-"+str(detector[n])
    beamf = io.load_beamf([freqc], abs_path="/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/")
    for m in range(noise_spectrum.shape[1]):
        if int(detector[n]) < 100:
            nside = 1024
        else:
            nside = 2048
        noise_spectrum[n,m,:] = create_noisecl_from_beamf(beamf, noise_level[n,m], freqc)
        # noise_spectrum[n,m,:] = create_noisespectrum(noise_level[n,m], fwhm[n,m], nside)

C_lN = np.zeros((npatch, lmax+1,len(detector), len(detector)))
for n in range(C_lN.shape[0]):
    C_lN[n]  = create_covmatrix(noise_spectrum[:,n,:])


# %% Create Noise Powerspectrum for Full Sky
noise_spectrum_FS = np.zeros((len(detector), 1,lmax+1))
HUnoise_spectrum_FS = np.zeros((len(detector), 1,lmax+1))
for n in range(noise_spectrum_FS.shape[0]):
    for m in range(noise_spectrum_FS.shape[1]):
        if int(detector[n]) < 100:
            nside = 1024
        else:
            nside = 2048
        # noise_spectrum_FS[n,m,:] = create_noisespectrum(noise_level_FS[n,m], fwhm[n,m], nside)
        noise_spectrum_FS[n,m,:] = create_noisespectrum(noise_level_FS[n,m], fwhm[n,m], nside)
        HUnoise_spectrum_FS[n,m,:] = create_noisespectrum(0.000290888**2*40**2*np.sqrt(2)/(1e6*2.782), 7, nside)
C_lN_fullsky = np.zeros((1, lmax+1, len(detector), len(detector)))
HUC_lN_fullsky = np.zeros((1, lmax+1, len(detector), len(detector)))
for n in range(C_lN_fullsky.shape[0]):
    C_lN_fullsky[n]  = create_covmatrix(noise_spectrum_FS[:,n,:])
    HUC_lN_fullsky[n] = create_covmatrix(HUnoise_spectrum_FS[:,n,:])


# %%
"""
                                    ##########################################
                                    ##########################################
                                                    SECTION 1
                                    ##########################################
                                    ##########################################
""" 


# %% Create Noisespectrum from gaussbeam
fsky = np.zeros((npatch, npatch), float)
np.fill_diagonal(fsky, 1/npatch*np.ones((npatch)))


# %% Calculate minimal covariance matrices
C_p = C_lS + C_lN
C_pNS = 0 * C_lS + C_lN
C_pNN = C_lS + 1e-6 * C_lN

C_pFS = C_lS + C_lN_fullsky[0]
C_pNS_FS = 0 * C_lS + C_lN_fullsky[0]
C_pNN_FS = C_lS + 1e-6 * C_lN_fullsky[0]


cov_min = np.concatenate(
    (
        np.zeros(shape=(C_p.shape[0],1)),

        np.array([
            [calculate_minimalcov(C_p[n,l])
                for l in range(1,C_p.shape[1])]
            for n in range(C_p.shape[0])])),axis=1)

cov_min_NS = np.concatenate(
    (
        np.zeros(shape=(C_pNS.shape[0],1)),
                np.array([
            [calculate_minimalcov(C_pNS[n,l])
                for l in range(1,C_pNS.shape[1])]
            for n in range(C_pNS.shape[0])])),axis=1)
cov_min_NN = np.concatenate(
    (
        np.zeros(shape=(C_pNN.shape[0],1)),
                        np.array([
            [calculate_minimalcov(C_pNN[n,l])
                for l in range(1,C_pNN.shape[1])]
            for n in range(C_pNN.shape[0])])),axis=1)

cov_min_FS = np.concatenate((np.zeros(shape=(1)),np.nan_to_num([calculate_minimalcov(C_pFS[l])for l in range(1,C_pFS.shape[0])])),axis=0)
cov_min_NS_FS = np.concatenate((np.zeros(shape=(1)),np.nan_to_num([calculate_minimalcov(C_pNS_FS[l])for l in range(1,C_pNS_FS.shape[0])])),axis=0)
cov_min_NN_FS = np.concatenate((np.zeros(shape=(1)),np.nan_to_num([calculate_minimalcov(C_pNN_FS[l])for l in range(1,C_pNN_FS.shape[0])])),axis=0)


# %% Calculate Variance of Full sky and Skypatches
var_patches = np.zeros((lmax+1,C_p.shape[0],C_p.shape[0]), float)
var_patches_NS = np.zeros((lmax+1,C_pNS.shape[0],C_p.shape[0]), float)
var_patches_NN = np.zeros((lmax+1,C_pNN.shape[0],C_p.shape[0]), float)

for n in range(npatch):
    var_patches[:,n,n] = 2 * cov_min[n,:] * cov_min[n,:]/((2*ll+1)*fsky[n,n])
    var_patches_NS[:,n,n] = 2 * cov_min_NS[n,:] * cov_min_NS[n,:]/((2*ll+1)*fsky[n,n])
    var_patches_NN[:,n,n] = 2 * cov_min_NN[n,:] * cov_min_NN[n,:]/((2*ll+1)*fsky[n,n])

# no noise -> trivial result  
var_patches[var_patches == inf] = 0
var_patches_NS[var_patches_NS == inf] = 0
var_patches_NN[var_patches_NN == inf] = 0

var_FS = 2*cov_min_FS * cov_min_FS/(2*ll+1)
var_FSma = ma.masked_array(var_FS, mask=np.where(var_FS<=0, True, False))

var_FS_NS = 2*cov_min_NS_FS * cov_min_NS_FS/(2*ll+1)
var_FS_NSma = ma.masked_array(var_FS_NS, mask=np.where(var_FS_NS<=0, True, False))


# %%
print(var_FS_NS.shape)
plt.plot(var_FS_NS*(ll*(ll+1)/(2*np.pi))*0.000290888**2)
plt.xlim((1,3e3))
# plt.ylim((1e-15,1e-9))
plt.yscale('log')
plt.grid()
plt.xscale('log')
# %% Calculate minimal variance from Skypatches 
opt = np.zeros((var_patches.shape[0]))
for l in range(var_patches.shape[0]):
    try:
        opt[l] = np.nan_to_num(
            calculate_minimalcov(var_patches[l], frac=1))#np.sqrt(np.sqrt(fsky[0,0]))))
    except:
        
        pass
opt_ma = ma.masked_array(opt, mask=np.where(opt<=0, True, False))


# %% Calculate minimal variance from Skypatches without Signal
opt_NS = np.zeros((var_patches_NS.shape[0]))
for l in range(var_patches_NS.shape[0]):
    try:
        opt_NS[l] = np.nan_to_num(
            calculate_minimalcov(var_patches_NS[l], frac=1))#np.sqrt(np.sqrt(fsky[0,0]))))
    except:
        pass
opt_NSma = ma.masked_array(opt_NS, mask=np.where(opt_NS<=0, True, False))


# %% Calculate minimal variance from Skypatches without Noise
opt_NN = np.zeros((var_patches_NN.shape[0]))
for l in range(var_patches_NN.shape[0]):
    try:
        opt_NN[l] = np.nan_to_num(
            calculate_minimalcov(var_patches_NN[l], frac=1))
    except:
        pass
opt_NNma = ma.masked_array(opt_NN, mask=np.where(opt_NN<=0, True, False))


# %% Plot results including Signal and Noise
fig, ax = plt.subplots(figsize=(8,6))

plt.plot(opt_ma, label='Variance, combined patches', lw=3, ls="-", color="black")
plt.plot(var_FSma, label="Variance, full sky", lw=3, ls="--", color="black")

plt.plot(C_lS[:,0,0], label = 'Planck EE', lw=3, ls="-", color="red", alpha=0.5)
for n in range(npatch):
    plt.plot(cov_min[n,:], lw=1, ls="-", color="red", alpha=0.5)
plt.plot(0,0, label='Minimal powerspectrum from patches', color='red', alpha=0.5, lw=1)

plt.plot(2*C_lS[:,0,0]*C_lS[:,0,0]/((2*ll+1)), label="Variance, Planck EE, full sky", lw=3, ls="-", color="blue")
plt.plot(opt_NN, label = "Variance, Planck EE, combined patches", lw=3, ls="--", color="green")

leg1 = plt.legend(loc='upper left')
pa = [None for n in range(var_patches.shape[1])]
for n in range(var_patches.shape[1]):
    p = plt.plot(var_patches[:,n,n], lw=2, ls="-", alpha=0.5)#, color=CB_color_cycle[0])
#     p = plt.plot(var_patches_NS[:,n,n], lw=2, ls="-", alpha=0.5)#, color=CB_color_cycle[0])
    col = p[0].get_color()
    pa[n] = Patch(facecolor=col, edgecolor='grey', alpha=0.5)
leg2 = plt.legend(handles=pa,
          labels=["" for n in range(var_patches.shape[1]-1)] + ['Variance, {} skypatches'.format(str(npatch))],
          ncol=var_patches.shape[1], handletextpad=0.5, handlelength=0.5, columnspacing=-0.5,
          loc='lower left')
ax.add_artist(leg1)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Multipole")
plt.ylabel(r"Variance $\sigma^2$ [$\mu K^4$]")
plt.xlim((2e1,3e3))
plt.ylim((1e-10,1e-2))
# plt.tight_layout()
ax2 = ax.twinx()
ax2.tick_params(axis='y', labelcolor="red")
ax2.set_ylabel(r'Powerspectrum $C_l$ [$\mu K^2$]', color= 'red')
plt.ylim((1e-10,1e-2))
plt.yscale('log')
plt.title("Combining {} skypatches which have noise and signal".format(str(npatch)))
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/analyticpatching-NoiseSignal{}patches.jpg".format(str(npatch)))


# %%
plt.figure(figsize=(8,6))
# plt.plot(var_FSma/opt_ma-1, label='{} patches - Noise+Signal'.format(str(npatch)), lw=2, ls="-", color='black')

# plt.plot(patch50, label='50 patches ', lw=2, ls="-", color=CB_color_cycle[1])
# plt.plot(patch2, label='2 patches ', lw=2, ls="-", color=CB_color_cycle[6])
# plt.plot(patch4, label='4 patches ', lw=2, ls="-", color=CB_color_cycle[5])
# plt.plot(patch5, label='5 patches ', lw=2, ls="-", color=CB_color_cycle[1])
plt.plot(patch10, label='10 patches ', lw=2, ls="-", color=CB_color_cycle[2])
# plt.plot(patch25, label='25 patches ', lw=2, ls="-", color=CB_color_cycle[0])
# plt.plot(patch20, label='20 patches ', lw=2, ls="-", color=CB_color_cycle[3])
# plt.plot(patch40, label='40 patches ', lw=2, ls="-", color=CB_color_cycle[1])
plt.plot(patch50, label='50 patches ', lw=2, ls="-", color=CB_color_cycle[3])
# plt.plot(patch80, label='80 patches ', lw=2, ls="-", color=CB_color_cycle[5])
plt.plot(patch100, label='100 patches ', lw=2, ls="-", color=CB_color_cycle[4])


# plt.plot(patch8, label='8 patches ', lw=2, ls="-", color=CB_color_cycle[4])
# plt.plot(patch16, label='16 patches ', lw=2, ls="-", color=CB_color_cycle[5])
plt.hlines(0,1e0,3e3, color="black", ls="--")
plt.xscale('log')
plt.xlabel("Multipole")
plt.ylabel(r"$\frac{\sigma^{2}_{Full}-\sigma^{2}_{patched}}{\sigma^{2}_{patched}}= \Delta \sigma^{2}_i$", fontsize=20)
plt.title("Variances comparison between full sky and patched sky")
# plt.yscale('log')
plt.legend(loc='upper left')
plt.xlim((2e1,3e3))
plt.ylim((-.02,3))
# plt.tight_layout()
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/comppachfullNoiseSignal2-100patches.jpg".format(str(npatch)))


# %%
plt.title('Comparison between improvement for various patches')
# plt.plot(patch4/patch2, label='i=4, j=2 patches ', lw=2, ls="-", color=CB_color_cycle[0])
# plt.plot(patch10/patch5, label='i=10, j=5 patches ', lw=2, ls="-", color=CB_color_cycle[1])
# plt.plot(patch50/patch25, label='i=50, j=25 patches ', lw=2, ls="-", color=CB_color_cycle[2])
# plt.plot(patch100/patch40, label='80/40 patches ', lw=2, ls="-", color=CB_color_cycle[3])
plt.plot(patch100/patch50, label='i=100, j=50 patches ', lw=2, ls="-", color=CB_color_cycle[3])
plt.ylabel(r"$\frac{\Delta \sigma^{2}_i}{\Delta \sigma^{2}_j}$", fontsize=20)
# plt.xscale('log')
# plt.plot(patch16/patch8, label='16/8 patches ', lw=2, ls="-", color=CB_color_cycle[3])
# plt.plot(patch32/patch16, label='32/16 patches ', lw=2, ls="-", color=CB_color_cycle[4])
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/compareimprovement100-50patches.jpg".format(str(npatch)))
# plt.ylim((1.8,2.01))

# %% Plot results including Noise only
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(opt_NSma, label='Variance, combined patches', lw=3, ls="-", color="black")
plt.plot(var_FS_NSma, label="Variance, full sky", lw=3, ls="--", color="black")
leg1 = plt.legend(loc='upper right')

pa = [None for n in range(var_patches_NS.shape[1])]
for n in range(var_patches_NS.shape[1]):
        p = plt.plot(var_patches_NS[:,n,n], lw=2, ls="-")
        col = p[0].get_color()
        pa[n] = Patch(facecolor=col, edgecolor='grey')
leg2 = plt.legend(handles=pa,
          labels=["" for n in range(var_patches_NS.shape[1]-1)] + ['Variance, {} skypatches'.format(str(npatch))],
          ncol=var_patches.shape[1], handletextpad=0.5, handlelength=0.5, columnspacing=-0.5,
          loc='lower left')
ax.add_artist(leg1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Multipole")
plt.ylabel("Powerspectrum / C_l")
plt.xlim((2e1,3e3))
# plt.tight_layout()
plt.ylim((1e-2,1e2))

plt.title("Combining {} skypatches which have noise only".format(str(npatch)))
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/analyticpatching-Noise{}patches.jpg".format(str(npatch)))

plt.figure(figsize=(8,6))
plt.plot(var_FS_NSma/opt_NSma-1, label='100 patches - Noise only', lw=2, ls="-", color='black')
# plt.plot(patch20, label='20 patches ', lw=2, ls="-", color=CB_color_cycle[1])
# plt.plot(patch4, label='4 patches ', lw=2, ls="-", color=CB_color_cycle[2])
plt.hlines(0,1e0,3e3, color="black", ls="--")
plt.xscale('log')
plt.xlabel("Multipole")
plt.ylabel(r"$\frac{\sigma^{Full}-\sigma^{patched}}{\sigma^{patched}}$", fontsize=20)
plt.title("Variances comparison between full sky and patched sky")
# plt.yscale('log')
plt.legend(loc='upper left')
plt.xlim((2e1,3e3))
# plt.ylim((-.2,6))
plt.tight_layout()
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/comppachfullNoiseOnly{}patches.jpg".format(str(npatch)))


# %%
patch1_NS = var_FS_NSma/opt_NSma-1
patch1 = var_FSma/opt_ma-1
var_patches1 = var_patches
var_patches_NS1 = var_patches_NS

# %%
patch2_NS = var_FS_NSma/opt_NSma-1
patch2 = var_FSma/opt_ma-1
var_patches2 = var_patches
var_patches_NS2 = var_patches_NS


# %%
patch4_NS = var_FS_NSma/opt_NSma-1
patch4 = var_FSma/opt_ma-1
var_patches4 = var_patches
var_patches_NS4 = var_patches_NS


# %%
patch5_NS = var_FS_NSma/opt_NSma-1
patch5 = var_FSma/opt_ma-1
var_patches5 = var_patches
var_patches_NS5 = var_patches_NS


# %%
patch8_NS = var_FS_NSma/opt_NSma-1
patch8 = var_FSma/opt_ma-1
var_patches8 = var_patches
var_patches_NS8 = var_patches_NS


# %%
patch10_NS = var_FS_NSma/opt_NSma-1
patch10 = var_FSma/opt_ma-1
var_patches10 = var_patches
var_patches_NS10 = var_patches_NS


# %%
patch16_NS = var_FS_NSma/opt_NSma-1
patch16 = var_FSma/opt_ma-1
var_patches16 = var_patches
var_patches_NS16 = var_patches_NS


# %%
patch20_NS = var_FS_NSma/opt_NSma-1
patch20 = var_FSma/opt_ma-1
var_patches20 = var_patches
var_patches_NS20 = var_patches_NS

# %%
patch25_NS = var_FS_NSma/opt_NSma-1
patch25 = var_FSma/opt_ma-1
var_patches25 = var_patches
var_patches_NS25 = var_patches_NS


# %%
patch32_NS = var_FS_NSma/opt_NSma-1
patch32 = var_FSma/opt_ma-1
var_patches32 = var_patches
var_patches_NS32 = var_patches_NS


# %%
patch40_NS = var_FS_NSma/opt_NSma-1
patch40 = var_FSma/opt_ma-1
var_patches40 = var_patches
var_patches_NS40 = var_patches_NS


# %%
patch50_NS = var_FS_NSma/opt_NSma-1
patch50 = var_FSma/opt_ma-1
var_patches50 = var_patches
var_patches_NS50 = var_patches_NS


# %%
patch80_NS = var_FS_NSma/opt_NSma-1
patch80 = var_FSma/opt_ma-1
var_patches80 = var_patches
var_patches_NS80 = var_patches_NS


# %%
patch100_NS = var_FS_NSma/opt_NSma-1
patch100 = var_FSma/opt_ma-1
var_patches100 = var_patches
var_patches_NS100 = var_patches_NS



# %%
patch200_NS = var_FS_NSma/opt_NSma-1
patch200 = var_FSma/opt_ma-1
var_patches200 = var_patches
var_patches_NS200 = var_patches_NS


# %%
fig, ax = plt.subplots(figsize=(8,6))
plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma), label="Full sky", alpha=0.5)
# plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch2+1)), label="2 patches", alpha=0.5)
# plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch4+1)), label="4 patches", alpha=0.5)
plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch10+1)), label="10 patches", alpha=0.5)
# plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch25+1)), label="25 patches", alpha=0.5)
# plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch50+1)), label="50 patches", alpha=0.5)
plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch100+1)), label="100 patches", alpha=0.5)
plt.legend()
plt.yscale('log')
# plt.tight_layout()
plt.title('Optimal variances for different number of skypatches')

plt.ylabel(r"Powerspectrum $C_l$ [$\mu K^2$]")
plt.xlabel("Multipole")
plt.xscale('log')
plt.ylim((1e-5,1e-2))
plt.xlim((1e1,3e3))
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/comperrorbarsfs-100patches.jpg".format(str(npatch)))



# %%

# %%
print(noise_spectrum_FS.shape)
for n in range(len(detector)):
    plt.plot(noise_spectrum[n,5,:], label = n)
    plt.plot(noise_spectrum_FS[n,0,:], label = n, color='black')
plt.legend()
plt.ylim((1e-3,1e1))
plt.xlim((1e0,3e3))
plt.yscale("log")
plt.xscale("log")

# %%
plt.plot(np.mean(noise_level, axis=0))
plt.yscale('log')
# print(noise_level[0][499])

# %%
plt.plot([noise_level_FS[3,0] for n in range(npatch)], label='noise level full sky')
plt.plot(noise_level[3,:], label='noise level per patch')
plt.plot([np.mean(noise_level[3,:]) for n in range(npatch)], label='noise level mean over all patches')
plt.legend()

# %%
print(noise_level_FS[:,0], fwhm[:,0])
print(C_lN_fullsky.shape)
plt.plot(HUC_lN_fullsky[0,:,0,0]*(ll*(ll+1)/(2*np.pi))*1e-12)
plt.plot(C_lN_fullsky[0,:,0,0]*(ll*(ll+1)/(2*np.pi))*1e-12)
plt.plot(noise_level_FS[0]*(ll*(ll+1)/(2*np.pi))[:2049]*1e-6*pows)
plt.xlim((1,3e3))
plt.ylim((1e-15,1e-6))
plt.yscale('log')
plt.grid()
plt.xscale('log')


# %%
freqc = ['030-030']
freqs = freqc.split('-')
beamf = io.load_beamf(['030-030'], abs_path="/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/")
hdul = beamf[freqc]
LFI_dict = {
        "030": 28,
        "044": 29,
        "070": 30
    }
b = np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0))
pows = 1/b**2
plt.plot(noise_level_FS[0]*pows)


# %%
C_p = C_lS + C_lN
plt.plot(C_lS[:,0,0])
for n in range(C_lN.shape[2]):
    plt.plot(C_lN[int(npatch/2),:,n,n], label = "cln, {} patch".format(detector[n]), lw=2, ls="--")#, color=CB_color_cycle[0])
    plt.plot(C_lN_fullsky[0,:,n,n], color='black')
    plt.plot(C_p[int(npatch/2),:,n,n], color='grey')
    plt.plot(C_p[0,:,n,n], color='red')
plt.legend()
plt.yscale("log")
plt.xlim((1e1,4e3))
plt.grid()

plt.ylim((1e-5,1e-2))
plt.xscale("log")
# %%
"""
                                    ##########################################
                                    ##########################################
                                                    Backup
                                    ##########################################
                                    ##########################################
"""



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

# %% no-noise combination, cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_NN = np.zeros((var_patches_NN.shape[0]))
for l in range(var_patches_NN.shape[0]):
    try:
        opt_NN[l] = np.nan_to_num(
            calculate_minimalcov(np.array([
                [cov_min_NN[0][l], 0],
                [0, cov_min_NN[l]]])))
    except:
        pass
opt_NNma = ma.masked_array(opt_NN, mask=np.where(opt_NN<=0, True, False))


# %% Idea 3.3: no signal all sky, cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_NS = np.zeros((var_patches_NS.shape[0]))
print(opt_NS.shape)
for l in range(var_patches_NS.shape[0]):
    try:
        opt_NS[l] = np.nan_to_num(
            calculate_minimalcov(np.array([
                [cov_min_NS[0,l], 0,0,0,0,0,0,0,0,0],
                [0,cov_min_NS[1,l],0,0,0,0,0,0,0,0],
                [0,0,cov_min_NS[2,l], 0,0,0,0,0,0,0],
                [0,0,0,cov_min_NS[3,l], 0,0,0,0,0,0],
                [0,0,0,0,cov_min_NS[4,l], 0,0,0,0,0],
                [0,0,0,0,0, cov_min_NS[5,l] ,0,0,0,0],
                [0,0,0,0,0,0,cov_min_NS[6,l], 0,0,0],
                [0,0,0,0,0,0,0,cov_min_NS[7,l], 0,0],
                [0,0,0,0,0,0,0,0,cov_min_NS[8,l], 0],
                [0,0,0,0,0,0,0,0,0,cov_min_NS[9,l]]])))
    except:
        pass
opt_NSma = ma.masked_array(opt_NS, mask=np.where(opt_NS<=0, True, False))

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
