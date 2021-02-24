"""
interactive.py: script for calling draw.component_separation using jupyter

"""

__author__ = "S. Belkner"
# %% run header
import json
from matplotlib import lines
import logging
import logging.handlers
from matplotlib.patches import Patch
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import component_separation.spherelib.python.spherelib.astro as slhpastro

import numpy as np
import component_separation.io as io
import seaborn as sns
import component_separation.masking as msk

import healpy as hp
from astropy.io import fits
from component_separation.cs_util import Planckf, Plancks

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/config.json', "r") as f:
    cf = json.load(f)



uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]


mskset = cf['pa']['mskset'] # smica or lens
freqdset = cf['pa']['freqdset'] # DX12 or NERSC

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir']
indir_path = cf[mch]['indir']
specfilter = cf['pa']["specfilter"]


# %%

freqdset = cf['pa']['freqdset'] # DX12 or NERSC
hitsvar = io.load_hitsvar(cf["pa"], 3, "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/")
0

# %%
print(np.min(hitsvar["143"]))
#%%
# build
tresh_low, tresh_up = 0.05, 1.0
masks = msk.hitsvar2mask(hitsvar, tresh_low, tresh_up)

# %%
# hitsmask = np.load('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/tmp/mask/hitsvar/DX12-freq_143-0.0to0.5-split_Full.hitshist.npy')
mean = np.mean(masks["143"])
std = np.std(masks["143"])
print(len(masks["143"][masks["143"]==True]))
# hp.mollview(hitsmask, norm= 'hist')
# hp.mollview(np.where(np.abs(hitsmask)>mean+20*std, 0.0, hitsmask))
hp.mollview(masks["143"])


# %%
print("K_CMB to K_RJ conversion factors")
print()
print("Instrument  Factor")
print(30*"-")
for channel in Planckf:
    print("{}         {}".format(channel.value, slhpastro.convfact(freq=int(channel.value)*1e9, fr=r'K_CMB',to=r'K_RJ')))


# %%
mean = np.mean(hitsvar["143"])
std = np.std(hitsvar["143"])
print(mean, std)
plt.hist(np.where(np.abs(hitsvar["143"])>mean+20*std, 0.0, hitsvar["143"]), bins = 100, alpha=0.5)
# plt.plot(mean+40*std, 1e8, lw=5, color="red")
plt.yscale('log')
plt.show()


# %%
plt.plot(np.sqrt(hdul[28].data.field(0)))
plt.plot(np.sqrt(hdul[29].data.field(0)))
plt.plot(np.sqrt(hdul[30].data.field(0)))
plt.plot(np.sqrt(np.sqrt(hdul[28].data.field(0))*np.sqrt(hdul[29].data.field(0))))
plt.grid()


hdul = fits.open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/map/frequency/HFI_SkyMap_100-field_2048_R3.01_full.fits")
hitsmap = hp.read_map("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/map/frequency/HFI_SkyMap_143-field_2048_R3.01_full.fits", field=4)

# %%
hdul[1].header
# %%
hdul = fits.open(
    "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/map/frequency/LFI_SkyMap_030-field_1024_R3.00_full.fits")

# %% Pixelisation
hdul[1].header
# %%
from pylab import *
import numpy
import re

Jansky2SI=1.0e-26
SI2Jansky=1.0e+26
speedOfLight=2.99792458e8
kBoltzmann=1.380658e-23
sigmaStefanBoltzmann=5.67051e-8
hPlanck=6.6260755e-34
astronomicalUnit=1.49597893e11
solarConstant=1368.0
tropicalYear=3.15569259747e7
tcmb = 2.726
prefixes = {'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'M':1e6,'G':1e9}

def K_CMB(freq):
    """ Return conversion factor from Kelvin CMB to SI at a given frequency.
    
    Parameters
    ----------
    freq : scalar. Frequency value. 

    Returns
    -------
    scalar, conversion factor value. 
    """
    x = (hPlanck *freq)/ (kBoltzmann * tcmb)
    ex = exp(x)
    den = ex-1
    den *= den
    den = 1/den
    fc = freq /speedOfLight 
    return 2*kBoltzmann * fc *fc * x * x * ex * den

def K_RJ(freq):
    """ Return conversion factor from Kelvin Raleigh Jeans to SI at a given frequency.

    Parameters
    ----------
    freq : scalar. Frequency value. 

    Returns
    -------
    scalar, conversion factor value. 
    """
    return 2*kBoltzmann*freq*freq/(speedOfLight*speedOfLight)


def convfact(freq=1.0e10, fr=r'mK_CMB',to=r'mK_CMB'):
    """ Compute the conversion factor between two units at a given frequency in GHz.

    Parameters
    ----------
    freq : scalar. Frequency value. 
    fr : string as prefixunit. unit can be either 'Jy/sr', 'K_CMB', 'K_RJ', 'K/KCMB', 'y_sz', 'si', 'taubeta'
    with optionnal prefix 'n', 'u', 'm', 'k', 'M', 'G' 
    to : string as prefixunit. 

    Returns
    -------
    scalar, conversion factor value. 
    """

    frpre, fru =  re.match(r'(n|u|m|k|M|G)?(Jy/sr|K_CMB|K_RJ|K/KCMB|y_sz|si|taubeta2)', fr).groups()
    topre, tou = re.match(r'(n|u|m|k|M|G)?(Jy/sr|K_CMB|K_RJ|K/KCMB|y_sz|si|taubeta2)', to).groups()

    if fru == 'Jy/sr':
        fru = 'Jy'
    if tou == 'Jy/sr':
        tou = 'Jy'
    if fru == tou:
        fac = 1.0
    else:
        fac = eval(fru+'(freq)/'+tou+'(freq)')
    
    if not frpre is None:
        fac *= prefixes[frpre]
    if not topre is None:
        fac /= prefixes[topre]

    return fac
for freq in ["030", "044", "070", "100", "143", "217","353"]:
    print(freq, ": ",  convfact(int(freq)*1e9, fr="K_CMB", to="K_RJ"))
# %%
