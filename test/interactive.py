
# %%
import os, sys
import matplotlib.pyplot as plt
from MSC import pospace as ps
import plancklens
import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
path = 'data/simulations/'


# %%
filename = "cmb/febecop_ffp10_lensed_scl_cmb_143_mc_0000.fits"
f = '{}{}'.format(path,filename)
hp_map = hp.read_map(f)
hdul = fits.open(f)


# %%
hp.mollview(hp_map, norm='hist')


# %%
hdul[1].header

# hdul.info()
# %%
