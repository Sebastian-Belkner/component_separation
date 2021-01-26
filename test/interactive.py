
# %%
import os, sys
import matplotlib.pyplot as plt
from MSC import pospace as ps
import plancklens
import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
path = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/"


# %%
filename = "beamf/BeamWF_LFI/beams_044_3.fits"
f = '{}{}'.format(path,filename)
# hp_map = hp.read_map(f)
hdul = fits.open(f)


# %%
hp.mollview(hp_map, norm='hist')


# %%
print(hdul[1].header)
hdul.info()
# %%
elaw = np.array([1,1,1])

cov = np.matrix("1 0 1;"+\
                "0 5 2;"+\
                "1 2 2")
print(np.array((cov @ elaw) / (elaw.T @ cov @ elaw)))
# %%
