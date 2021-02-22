"""
interactive.py: script for calling draw.component_separation using jupyter

"""

__author__ = "S. Belkner"

# %% interactive header
import matplotlib.pyplot as plt

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
import numpy as np
import seaborn as sns

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
import healpy as hp
from astropy.io import fits
hdul = fits.open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/beamf/BeamWF_LFI/LFI_RIMO_R3.31.fits")

# %%
hdul[31].header


# %%
hp.read_map("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/beamf/BeamWF_LFI/LFI_RIMO_R3.31.fits", field=0)


# %%
plt.plot(np.sqrt(hdul[28].data.field(0)))
plt.plot(np.sqrt(hdul[29].data.field(0)))
plt.plot(np.sqrt(hdul[30].data.field(0)))
plt.plot(np.sqrt(np.sqrt(hdul[28].data.field(0))*np.sqrt(hdul[29].data.field(0))))
plt.grid()

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
