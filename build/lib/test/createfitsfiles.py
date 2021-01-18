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
freqpath = "data/map/frequency/HFI_SkyMap_143-field_2048_R3.00_full.fits"
testfreqpath = 'test/data/map/HFI_SkyMap_143-field_128_R3.00_full.fits'

maskpath = "data/mask/HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
testmaskpath = "test/data/mask/HFI_Mask_GalPlane-apo0_{}_R2.00.fits".format(size)


beampath = "data/beamf/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_{}x{}.fits".format(143, 143)
testbeampath = "test/data/beamf/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_{}x{}.fits".format(143, 143)
#%%
hdul = fits.open(maskpath)
hdul.writeto(testmaskpath)

#%%
hdul_maps = fits.open(testmaskpath)
print(hdul_maps[1].header)
# reduced_maps = hp.pixelfunc.ud_grade(hdul_maps, nside_out=size)

#%%
hdul = fits.open(testmaskpath)
hp_map = hp.read_map(
    testmaskpath,
    field=(0,1,2),
    nest=True)
reduced_maps = hp.pixelfunc.ud_grade(hp_map, nside_out=size)

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
hdul_keepo = fits.open(testmaskpath)
print(table_hdu.data)
print(hdul_keepo[1].data)

#%%
with fits.open(testmaskpath, mode='update') as hdul:
    hdul[1]=table_hdu

#%%
with fits.open(testmaskpath, mode='update') as hdtest:
    hdtest[1].header['comment'] = "This is a modified file for testpurposes only It's sole purpose is for testing the component_separation pipeline from 'https://github.com/Sebastian-Belkner/component_separation' NSIDE is reduced to 128, using hp.ud_grade(). USE WITH CARE"

#%%
hdtest = fits.open(testmaskpath)
hdtest[1].header

#%%
int_mask = hp.read_map("test/data/mask/HFI_Mask_GalPlane-apo0_128_R2.00.fits")
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
def count_bad(m):
    i = 0
    nbad = 0
    size = m.size
    for i in range(m.size):
        if np.abs(m[i] - UNSEEN) < UNSEEN_tol:
            nbad += 1
    return nbad

def mkmask(m):
    nbad = 0
    size = m.size
    i = 0
    # first, count number of bad pixels, to see if allocating a mask is needed
    nbad = count_bad(m)
    mask = np.ndarray(shape=(1,), dtype=np.int8)
    #cdef np.ndarray[double, ndim=1] m_
    if nbad == 0:
        return False
    else:
        mask = np.zeros(size, dtype = np.int8)
        #m_ = m
        for i in range(size):
            if np.abs(m[i] - UNSEEN) < UNSEEN_tol:
                mask[i] = 1
    mask.dtype = bool
    return mask
# %%
hpdata_0 = hp.read_map("data/map/frequency/HFI_SkyMap_100-field_2048_R3.00_full.fits", field=0)
hpdata_1 = hp.read_map("data/map/frequency/HFI_SkyMap_100-field_2048_R3.00_full.fits", field=1)
hpdata_2 = hp.read_map("data/map/frequency/HFI_SkyMap_100-field_2048_R3.00_full.fits", field=2)

# %%
norm='hist'
hp.mollview(hpdata_0, norm=norm)
hp.mollview(hpdata_1, norm=norm)
hp.mollview(hpdata_2, norm=norm)

# %%
# 'mask/HFI_Mask_GalPlane-apo0_{}_R2.00.fits'
pmsk = hp.read_map('data/mask/gmaskP_apodized_0_2048.fits.gz', dtype=None).astype(np.bool)
data = hp.read_map("data/map/frequency/HFI_SkyMap_143-field_2048_R3.00_full.fits", field=0, dtype=None)

map_P_masked = hp.ma(data)
# map_P_masked.mask = np.logical_not(pmsk)
hp.mollview(map_P_masked, norm='hist')
plt.show()

# %%
hp_psmask = hp.read_map('data/mask/psmaskP_2048.fits.gz', dtype=np.bool)
hp_gmask = hp.read_map('data/mask/gmaskP_apodized_0_2048.fits.gz', dtype=np.bool)
gen_mask = hp.read_map('data/mask/HFI_Mask_GalPlane-apo0_2048_R2.00.fits', dtype=np.bool)
pmask = hp_psmask*hp_gmask
hp.mollview(gen_mask)

# %%
elaw = np.ones(len([10]))
None @ elaw

# %%
import component_separation.MSC.MSC.pospace as ps
from component_separation.cs_util import Planckf, Plancks
import logging
import logging.handlers
from logging import DEBUG, ERROR, INFO, CRITICAL
import os, sys
import component_separation.powspec as pw
import component_separation.io as io
from typing import Dict, List, Optional, Tuple


PLANCKMAPFREQ = [p.value for p in list(Planckf)]
LOGFILE = 'messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
freqfilter = [
        Planckf.LFI_1.value,
        Planckf.LFI_2.value,
        Planckf.HFI_1.value,
        # Planckf.HFI_2.value,
        # Planckf.HFI_3.value,
        # Planckf.HFI_4.value,
        Planckf.HFI_5.value,
        Planckf.HFI_6.value
        ]
specfilter = [
    Plancks.TE.value,
    Plancks.TB.value,
    Plancks.ET.value,
    Plancks.BT.value,
    Plancks.TT.value,
    Plancks.BE.value,
    Plancks.EB.value
    ]
lmax = 200
lmax_mask = 400
llp1=True

path = 'data/'
spec_path = 'data/tmp/spectrum/'

tmask_filename = 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
# HFI_Mask_GalPlane-apo0_128_R2.00.fits

pmask_filename = ['psmaskP_2048.fits.gz', 'gmaskP_apodized_0_2048.fits.gz']
# ['HFI_Mask_GalPlane-apo0_2048_R2.00.fits']
# ['psmaskP_2048.fits.gz', 'gmaskP_apodized_0_2048.fits.gz']

spec_filename = 'lmax_{lmax}-lmax_mask_{lmax_mask}-tmask_{tmask}-pmask_{pmask}-freqs_{freqs}_.npy'.format(
    lmax = lmax,
    lmax_mask = lmax_mask,
    tmask = tmask_filename[::5],
    pmask = ','.join([pmsk[::5] for pmsk in pmask_filename]),
    freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter])
)

tqumap = io.get_data(
    path=path,
    freqfilter=freqfilter,
    tmask_filename=tmask_filename,
    pmask_filename=pmask_filename,
    nside=[1024,2048])

# %%
spectrum = io.load_spectrum(spec_path, spec_filename)
# %%
print(spectrum['143-143'])
# print(spectrum.item()['143-143']['EE'])
# print(spectrum.item())
# %%
beamf = fits.open(
    "data/beamf/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_100x100.fits")
beamf[1].data
# %%
