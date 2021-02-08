
# Beam windowfunctions for CMB @ all Ghz
# Wiener filter on map level or subtract noise from powerspectrum?


# %%
import os, sys
import matplotlib.pyplot as plt
from MSC import pospace as ps
import plancklens
import numpy as np

import pandas as pd
import healpy as hp
import functools
import json
from astropy.io import fits

import component_separation.io as io
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks

path_cmb = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/test/data/simulations/cmb/"
path_noise = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/test/data/simulations/noise/"
path_smica = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/test/data/map/"

mch = "XPS"
with open('./../config.json', "r") as f:
    cf = json.load(f)

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
mskset = cf['pa']['mskset'] # smica or lens
freqdset = cf['pa']['freqdset'] # DX12 or NERSC
lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]
spec_path = cf[mch]['outdir']
indir_path = cf[mch]['indir']
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]
spec_filename = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
    freqdset = freqdset,
    lmax = lmax,
    lmax_mask = lmax_mask,
    mskset = mskset,
    spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
    freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
    split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
spectrum = io.load_spectrum("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+spec_path, spec_filename)
freqcomb =  ["{}-{}".format(FREQ,FREQ2)
                for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
beamf = io.get_beamf(cf=cf, mch=mch, freqcomb=freqcomb, abs_path="/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/") 
fkey = "143-143"
hdul = beamf[fkey]
df = pw.create_df(spectrum, freqfilter, specfilter)
df_sc = pw.apply_scale(df, specfilter, llp1=llp1)
df_scbf = pw.apply_beamfunction(df_sc, beamf, lmax, specfilter)

mskset = cf['pa']['mskset'] # smica or lens
indir_path = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/"
pmask_filename = cf[mch][mskset]['pmask']['filename']
pmask_path = cf[mch][mskset]['pmask']["path"]
def multi(a,b):
    return a*b
pmasks = [hp.read_map(
    '{path}{pmask_path}{pmask_filename}'
    .format(
        path = indir_path,
        pmask_path = pmask_path,
        pmask_filename = a), dtype=np.bool) for a in pmask_filename]
pmask = functools.reduce(multi, pmasks)
pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=1024)


# %%
synalm = hp.synfast(spectrum["143-143"]["EE"], nside=2048)
hp.mollview(synalm)


# %%
cmb_filename = "febecop_ffp10_lensed_scl_cmb_143_mc_0000.fits"
f = '{}{}'.format(path_cmb,cmb_filename)
q_cmb = hp.read_map(f, field=1)
u_cmb = hp.read_map(f, field=2)
spec_cmb = ps.map2cl_spin(
    qumap=[q_cmb,u_cmb],
    mask=pmask,
    spin=2,
    lmax=4000,
    lmax_mask=8000)
spec_cmb_sc = spec_cmb[0] * 1e12
spec_cmb_scaled = [s*l*(l+1)/(4*np.pi) for l,s in enumerate(spec_cmb_sc)]
spec_cmb_scbf = spec_cmb_scaled / (hdul[1].data.field(1)[:lmax+1] * hdul[1].data.field(1)[:lmax+1])


# %%
noise_filename = "ffp10_noise_143_full_map_mc_00000.fits"
f = '{}{}'.format(path_noise,noise_filename)
q_noise = hp.read_map(f, field=1)
u_noise = hp.read_map(f, field=2)
spec_noise = ps.map2cl_spin(
    qumap=[q_noise,u_noise],
    mask=pmask,
    spin=2,
    lmax=4000,
    lmax_mask=8000)
spec_noise_sc = spec_noise[0] * 1e12
spec_noise_scaled = [s*l*(l+1)/(4*np.pi) for l,s in enumerate(spec_noise_sc)]
spec_noise_scbf = spec_noise_scaled / (hdul[1].data.field(1)[:lmax+1] * hdul[1].data.field(1)[:lmax+1])


# %%
smica_hm1_filename = "COM_CMB_IQU-smica_2048_R3.00_oe1.fits"
f1 = '{}{}'.format(path_smica,smica_hm1_filename)
smica_hm2_filename = "COM_CMB_IQU-smica_2048_R3.00_oe2.fits"
f2 = '{}{}'.format(path_smica,smica_hm2_filename)
q_cnoise = hp.read_map(f1, field=1)-hp.read_map(f2, field=1)
u_cnoise = hp.read_map(f1, field=2)-hp.read_map(f2, field=2)
spec_cnoise = ps.map2cl_spin(
    qumap=[q_cnoise,u_cnoise],
    mask=pmask,
    spin=2,
    lmax=4000,
    lmax_mask=8000)
spec_cnoise_sc = spec_noise[0] * 1e12
spec_cnoise_scaled = [s*l*(l+1)/(4*np.pi) for l,s in enumerate(spec_cnoise_sc)]
spec_cnoise_scbf = spec_cnoise_scaled / (hdul[1].data.field(1)[:lmax+1] * hdul[1].data.field(1)[:lmax+1])


# %%
path_fg = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/test/data/simulations/foregrounds/"
fg_filename = "COM_SimMap_thermaldust-ffp10-skyinbands-143_2048_R3.00_full.fits"
f = '{}{}'.format(path_fg,fg_filename)
q_fg = hp.read_map(f, field=1)
u_fg = hp.read_map(f, field=2)
spec_fg = ps.map2cl_spin(
    qumap=[q_fg,u_fg],
    mask=pmask,
    spin=2,
    lmax=4000,
    lmax_mask=8000)
spec_fg_sc = spec_fg[0] * 1e12
spec_fg_scaled = [s*l*(l+1)/(4*np.pi) for l,s in enumerate(spec_fg_sc)]
spec_fg_scbf = spec_fg_scaled / (hdul[1].data.field(1)[:lmax+1] * hdul[1].data.field(1)[:lmax+1])


# %%
ccmb_filename = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
f = '{}{}'.format(path_cmb,ccmb_filename)
q_ccmb = hp.read_map(f, field=1)
u_ccmb = hp.read_map(f, field=2)
spec_ccmb = ps.map2cl_spin(
    qumap=[q_ccmb,u_ccmb],
    mask=pmask,
    spin=2,
    lmax=4000,
    lmax_mask=8000)
spec_ccmb_sc = spec_ccmb[0] * 1e12
spec_ccmb_scaled = [s*l*(l+1)/(4*np.pi) for l,s in enumerate(spec_ccmb_sc)]
spec_ccmb_scbf = spec_ccmb_scaled / (hdul[1].data.field(1)[:lmax+1] * hdul[1].data.field(1)[:lmax+1])


# %%
hp.mollview(pmask, norm='hist')
hp.mollview(q_cmb, norm='hist')
hp.mollview(q_ccmb, norm='hist')
hp.mollview(q_noise, norm='hist')
hp.mollview(q_fg, norm='hist')


# %%
import camb
from camb import model, initialpower
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(3800, lens_potential_accuracy=0)
#calculate results for these parameters
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
print(totCL.shape)
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ls = np.arange(totCL.shape[0])
fig, ax = plt.subplots(2,2, figsize = (12,12))
ax[0,0].plot(ls,totCL[:,0], color='k')
ax[0,0].plot(ls,unlensedCL[:,0], color='r')
ax[0,0].set_title('TT')
ax[0,1].plot(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0])
ax[0,1].set_title(r'$\Delta TT$')
ax[1,0].plot(ls,totCL[:,1], color='k')
ax[1,0].plot(ls,unlensedCL[:,1], color='r')
ax[1,0].set_title(r'$EE$')
ax[1,1].plot(ls,totCL[:,3], color='k')
ax[1,1].plot(ls,unlensedCL[:,3], color='r')
ax[1,1].set_title(r'$TE$');
for ax in ax.reshape(-1): ax.set_xlim([2,3800])


# %%
def ee_noise(l):
    dp=np.sqrt(2)*27*1e-6
    T=2.728
    sig=1/(60*7)
    return (dp/T)**2*np.exp(l*(l+1)*sig**2/(8*np.log(2)))

param_noise = [ee_noise(l) for l in range(4000)]
# %%

plt.figure()
plt.xlim((10,4000))
# plt.ylim((3e-16,2e-15))
plt.title("EE spectra @ 143Ghz")
plt.xlabel("Multipole l")
plt.ylabel("Scaled powerspectrum")
plt.loglog(spec_cmb_scaled, label='CMB from PL ')
plt.loglog(df_scbf["EE"]['143-143'], label='Estimated CMB')
# plt.loglog(spec_noise_scbf, label='Noise', alpha=0.5)
plt.loglog(df_scbf["EE"]['143-143']-np.array(spec_noise_scbf)-np.array(spec_fg_scbf), label='Estimated_clean CMB (minus noise and dust)')
# plt.loglog(spec_fg_scbf, label='Dust', alpha=0.5)
plt.legend()
plt.grid()
plt.savefig('powerspectrum-calculator-comparison143ghz.jpg')
plt.show()


plt.figure()
plt.xlim((10,4000))
plt.title("EE spectra @ all Ghz")
plt.xlabel("Multipole l")
plt.ylabel("Scaled powerspectrum")
plt.loglog(totCL[:,1], label="CAMB lensed CMB")
# plt.loglog(unlensedCL[:,1], label="CAMB unlensed CMB")
plt.loglog(spec_ccmb_scbf-np.array(spec_cnoise_scbf), label="CMB from SMICA minus Noise")
plt.loglog(param_noise, label = 'Parametric Noise (from Wayne & Hu paper)')
plt.loglog(spec_cnoise_scbf, label='Noise (from odd-even ring)', alpha=0.5)
# plt.loglog(np.array(spec_ccmb_scaled)-np.array(spec_cnoise_scaled), label="Noise-subtracted-CMB from SMICA")
# plt.loglog(spec_fg_scbf, label='Dust', alpha=0.5)

plt.legend()
plt.grid()
plt.savefig('powerspectrum-calculator-comparison-allGhz.jpg')
plt.show()
# %%

FREQ = "as"
FREQ2 = "dsaf"

"-".join([FREQ,FREQ2])
# %%
import numpy as np
data = {
    "a": [10, 10, 10, 10, 10],
    "b": [10, 10, 10, 10, 10]
}
ll = lambda x: x*(x+1)*1e12/(2*np.pi)
lmax = 4
sc = np.array([ll(idx) for idx in range(lmax+1)])

for specID, specval in data.items():
    data[specID] *= sc

print(len(list(data.values())[0]))
# %%
