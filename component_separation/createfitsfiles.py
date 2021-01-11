#%%
import os, sys
import matplotlib.pyplot as plt
from MSC import pospace as ps
import plancklens
import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
os.chdir("..")

#%%
size=128
mappath = "data/map/frequency/HFI_SkyMap_143-field_2048_R3.00_full.fits"
testmappath = 'test/data/map/newtable.fits'

#%%
hdul = fits.open(mappath)
hdul.writeto(testmappath)

#%%
hdul_maps = hp.read_map(mappath, field=(0,1,2))
reduced_maps = hp.pixelfunc.ud_grade(hdul_maps, nside_out=size)

#%%
hdul[1].header['comment'][-2]

#%%
hdul = fits.open(testmappath)
hp_map = hp.read_map(
    testmappath,
    field=(0,1,2),
    nest=True)
reduced_maps = hp.pixelfunc.ud_grade(hp_map, nside_out=256)
print(hdul[1].data)
print(hp_map)
print(reduced_maps)

#%%
a = fits.Column(
    name='I_STOKES',
    array=reduced_maps[0],
    format='E',
    unit='Kcmb'
    )
b = fits.Column(
    name='Q_STOKES',
    array=reduced_maps[1],
    format='E',
    unit='Kcmb'
    )
c = fits.Column(
    name='U_STOKES',
    array=reduced_maps[2],
    format='E',
    unit='Kcmb'
    )

table_hdu = fits.BinTableHDU.from_columns([a, b, c])

#%%
hdul_keepo = fits.open('test/data/map/newtable.fits')
print(table_hdu.data)
print(hdul_keepo[1].data)

#%%
with fits.open('test/data/map/newtable.fits', mode='update') as hdul:
    hdul[1]=table_hdu

#%%

with fits.open('test/data/map/newtable.fits', mode='update') as hdtest:
    hdtest[1].header['comment'] = "It's sole purpose is for testing the component_separation pipeline from 'https://github.com/Sebastian-Belkner/component_separation'"

#%%
hdtest = fits.open('test/data/map/newtable.fits')
hdtest[1].header

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
            if is_invertible(cov[comp][:,:,l]):
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
