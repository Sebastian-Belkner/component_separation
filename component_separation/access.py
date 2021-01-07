import os, sys
import matplotlib.pyplot as plt
from MSC import pospace as ps
import plancklens
import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits

#%%
### FITS file related
#%%
hdul = fits.open("data/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_217x353.fits")
hdul[1].header
#%%
hdul.info()

#%%
hdul["WINDOW FUNCTION"].header
# hdul[1].header[12]

# %%
hdul = fits.open("data/LFI_RIMO_R3.31.fits")
# %%
hdul[1].header

##### OLD #######
##### OLD #######
##### OLD #######

#%%
int_mask = hp.read_map("data/mask/HFI_Mask_GalPlane-apo0_2048_R2.00.fits")
hp.mollview(int_mask)
#%%
# tmap[FREQ]['mask']
# hp.mollview(qmap, norm='hist')
hdul_mask = fits.open('data/mask/psmaskP_2048.fits.gz')
hdul_mask[1].header
hp_psmask = hp.read_map('data/mask/psmaskP_2048.fits.gz')
hp.mollview(hp_psmask)
hp_gmask = hp.read_map('data/mask/gmaskP_apodized_0_2048.fits.gz')
hp.mollview(hp_gmask)

hp.mollview(hp_psmask*hp_gmask)
#%%

for l in range(lmax):
    for comp in PLANCKSPECTRUM:
        if comp not in specfilter:
            print(l, comp)
            if is_invertible(cov[comp][:,:,l])
                np.linalg.inv(cov[comp][:,:,l])

cov_inv = {comp: np.linalg.inv(np.cov(df[comp], rowvar=False))
            for comp in PLANCKSPECTRUM if comp not in specfilter}
print(cov_inv)


#%%
# hp_psmask = hp.read_map('data/mask/psmaskP_2048.fits.gz')
# hp_gmask = hp.read_map('data/mask/gmaskP_apodized_0_2048.fits.gz')
hdul = fits.open('data/map/frequency/HFI_SkyMap_143-field_2048_R3.00_full.fits')
print(hdul[1].header["ORDERING"])

# %%
msk = hp.read_map('data/mask/gmaskP_apodized_0_2048.fits.gz').astype(np.bool)
map_P = hp.reorder(hp.read_map("data/map/frequency/HFI_SkyMap_143-field_2048_R3.00_full.fits", field=1), n2r=True)
map_P_masked = hp.ma(map_P)
map_P_masked.mask = np.logical_not(msk)
#%%
nest=True
hp.mollview(map_P_masked, nest=True)
hp.mollview(map_P_masked)
hp.mollview(msk, nest=False)
# %%
pmsk, pmskheader = hp.read_map('data/mask/gmaskP_apodized_0_2048.fits.gz', h=True)
print(pmskheader)
hp.mollview(pmsk)
# %%
hd=fits.open('data/mask/gmaskP_apodized_0_2048.fits.gz', h=True)
hd[1].header
# %%
