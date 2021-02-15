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
hdul = hp.read_map("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/map/frequency/LFI_SkyMap_030-field_1024_R3.00_full.fits")

# %% Pixelisation
