#!/usr/local/bin/python

"""
app_smica.py: script for executing main functionality of smica.
Depends on all (noise, signal, full) spectra being generated to begin with. Use ``app_powerspectrum.py``, if e.g. noisespectra are missing.

To let SMICA run sucessfully, we have adapted the following:
1. For B-fit, set all C_ltot bins < 0 to zero.
2. smica.mocel.quasi_newton()'s local diag(CRB): I assume all elements to be positive
"""

__author__ = "S. Belkner"

import os, sys

import healpy as hp
import numpy as np

import component_separation.covariance as cv
import component_separation.smica_interface as smint
import component_separation.map as mp
import component_separation.MSC.MSC.pospace as ps #TODO remove dependency
import component_separation.transformer as trsf

from component_separation.cs_util import Helperfunctions as hpf

from component_separation.cs_util import Config
from component_separation.cs_util import Smica_bins
from component_separation.cs_util import Filename_gen_SMICA as fns_gen
from component_separation.cs_util import Filename_gen as fn_gen
from component_separation.io import IO

os.environ["OMP_NUM_THREADS"] = "32"
csu = Config()
fn = fn_gen(csu)
fns = fns_gen(csu)
io = IO(csu)

tmask_fn = fn.get_mask('T')
pmask_fn = fn.get_mask('P')
tmask_sg = io.load_mask(tmask_fn, stack=True)
pmask_sg = io.load_mask(pmask_fn, stack=True)

if csu.spectrum_type == 'pseudo':
    tmask_sg = mp.apodize_mask(tmask_sg)
    pmask_sg = mp.apodize_mask(pmask_sg)

tmask, pmask = dict(), dict()
for FREQ in csu.PLANCKMAPFREQ:
    if FREQ not in csu.freqfilter:
        tmask[FREQ] = tmask_sg
        pmask[FREQ] = pmask_sg

smica_params = dict({
    'cov': dict(), 
    "cov4D": dict(), 
    "CMB": dict(),
    "gal": dict(),
    "gal_mm": dict(), 
    "w": dict()})


def bin_data(bins, cov_ltot_s, cov_lN_s, cov_lS_s):

    cov_ltot_bnd = hpf.bin_it(cov_ltot_s, bins=bins)

    cov_lN_bnd = hpf.bin_it(cov_lN_s, bins=bins)
    cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T

    C_lS_bnd =  hpf.bin_it(cov_lS_s, bins=bins)

    cov_ltot_bnd[cov_ltot_bnd<0] = 0.0 #Reminder: this would not work for TE!

    return cov_ltot_bnd.copy(), cov_lN_bnd.copy(), C_lS_bnd.copy()


def smooth_data(covltot, covlN, covlS):
    cutoff = 1200
    covlT_smoothed = cv.cov2cov_smooth(covltot, cutoff=cutoff)
    covlN_smoothed = cv.cov2cov_smooth(covlN, cutoff=cutoff)
    covlS_smoothed = cv.cov2cov_smooth(covlS, cutoff=cutoff)

    # print(covlT_smoothed, covlN_smoothed, covlS_smoothed)
    return np.nan_to_num(covlT_smoothed), np.nan_to_num(covlN_smoothed), np.nan_to_num(covlS_smoothed)


def fitp(cov_ltot, cov_lN, cov_lS, nmodes, gal_mixmat, B_fit):

    smica_model = smint.build_smica_model(len(nmodes), np.nan_to_num(cov_lN), np.nan_to_num(cov_lS), gal_mixmat, B_fit)
    smica_model, hist = smint.fit_model_to_cov(
        smica_model,
        cov_ltot,
        nmodes,
        maxiter=50,
        noise_template=np.where(cov_lN<4e-3, 0.0, 1.0),
        afix=None)

    return smica_model


def fitp_old(covltot, covlN, covlS, nmodes, gal_mixmat, B_fit):

    smica_model = smint.build_smica_model_old(len(nmodes), np.nan_to_num(covlN), np.nan_to_num(covlS), gal_mixmat, B_fit)
    smica_model, hist = smint.fit_model_to_cov_old(
        smica_model,
        covltot,
        nmodes,
        maxiter=50,
        noise_template=np.where(False, 0.0, 1.0),#covlN<4e-3
        afix=None)

    return smica_model


def extract_model_parameters(smica_model):

    smica_cmb = smica_model.get_comp_by_name('cmb').powspec()[0]
    smica_gal = np.array([smica_model.get_comp_by_name('gal').powspec()])
    smica_cov = np.array([smica_model.covariance()])
    smica_cov4D = np.array([smica_model.covariance4D()])

    smica_weights_tot = cv.cov2weight(smica_cov, np.array(csu.PLANCKMAPFREQ_f))

    return smica_cov, smica_cov4D, smica_cmb, smica_gal, smica_weights_tot


def run_fit(fit):

    _Tscale = "K_CMB"
    Cltot = io.load_data(fn.get_spectrum("T", "non-separated"))
    covltot = cv.build_covmatrices(Cltot, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)

    ClN = io.load_data(fn.get_spectrum("N", "non-separated"))
    covlN = cv.build_covmatrices(ClN, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)

    ClS = io.load_data(fns.get_spectrum("S", "non-separated"))
    covlS = cv.build_covmatrices(ClS, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)

    nmodes_lowell = smint.calc_nmodes(Smica_bins.SMICA_lowell_bins, pmask['100'])
    nmodes_highell = smint.calc_nmodes(Smica_bins.SMICA_highell_bins, pmask['100'])

    covltot_smoothed, covlN_smoothed, ClS_smoothed = covltot, covlN, covlS#smooth_data(covltot, covlN, covlS)

    ## EE fit
    covltot_bnd, covlN_bnd, ClS_bnd = bin_data(Smica_bins.SMICA_lowell_bins, covltot_smoothed[1], covlN_smoothed[1], ClS_smoothed[1])
    smica_model = fit(covltot_bnd, covlN_bnd, ClS_bnd, nmodes_lowell, None, B_fit=False)
    smica_params["gal_mm"]["E"] = np.array([smica_model.get_comp_by_name('gal').mixmat()])

    covltot_bnd, covlN_bnd, ClS_bnd = bin_data(Smica_bins.SMICA_highell_bins, covltot_smoothed[1], covlN_smoothed[1], ClS_smoothed[1])
    smica_model = fit(covltot_bnd, covlN_bnd, ClS_bnd, nmodes_highell, smica_params["gal_mm"]["E"][0], B_fit=False)
    smica_params["cov"]["E"], smica_params["cov4D"]["E"], smica_params["CMB"]["E"], smica_params["gal"]["E"], smica_params["w"]["E"]  = extract_model_parameters(smica_model)

    ## BB fit
    covltot_bnd, covlN_bnd, ClS_bnd = bin_data(Smica_bins.SMICA_lowell_bins, covltot_smoothed[2], covlN_smoothed[2], ClS_smoothed[2])
    smica_model = fit(covltot_bnd, covlN_bnd, ClS_bnd, nmodes_lowell, None, B_fit=True)
    smica_params["gal_mm"]["B"] = np.array([smica_model.get_comp_by_name('gal').mixmat()])

    covltot_bnd, covlN_bnd, ClS_bnd = bin_data(Smica_bins.SMICA_highell_bins, covltot_smoothed[2], covlN_smoothed[2], ClS_smoothed[2])
    smica_model = fit(covltot_bnd, covlN_bnd, ClS_bnd, nmodes_highell, smica_params["gal_mm"]["B"][0], B_fit=True)
    smica_params["cov"]["B"], smica_params["cov4D"]["B"], smica_params["CMB"]["B"], smica_params["gal"]["B"], smica_params["w"]["B"]  = extract_model_parameters(smica_model)

    for k, v in smica_params.items():
        io.save_data(
            np.concatenate(
                (v["E"],v["B"])),
                fns.get_misc(k)
            )


def run_propag():

    bins = csu.bins
    W_smica = io.load_data(fns.get_misc('w'))
    W_mv = io.load_data(fn.get_misc('w'))
    W_total = hpf.interp_smica_mv_weights(W_smica, W_mv, bins, 4001)
    W_total[:,:,0:2] = 0.0

    nalm = int((csu.lmax)*(csu.lmax-1+2)/2) 
    alm = np.zeros(shape=(len(csu.PLANCKMAPFREQ_f),3,nalm))
    
    maps = dict()
    for FREQ in csu.PLANCKMAPFREQ:
        if FREQ not in csu.freqfilter:
            inpath_map_pla_name = fn.get_pla(FREQ, "T")
            print("inpath_map_pla_name: {}".format(inpath_map_pla_name))
            maps[FREQ] = io.load_pla(inpath_map_pla_name, field=(0,1,2), ud_grade=(True, FREQ))

    maps = mp.process_all(maps)
    beamf_dict = fn.get_beamf()
    beamf = io.load_beamf(beamf_dict, csu.freqcomb)

    for itf, freq in enumerate(csu.PLANCKMAPFREQ_f):
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        alm[itf][1:] = trsf.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, csu.lmax-1) # full sky QU->EB        #TODO no TT at the moment
 
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)

    for itf, det in enumerate(csu.PLANCKMAPFREQ): #weights do not depend on freqfilter, but almE/B do
        if det in csu.PLANCKMAPFREQ_f:
            print('freq: ', det)
            ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
            # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
            combalmE += hp.almxfl(hp.almxfl(alm[itf][1],np.nan_to_num(1/beamf[1,itf,itf,:csu.lmax])), np.squeeze(W_total[1,itf,:]))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:csu.lmax])
            combalmB += hp.almxfl(hp.almxfl(alm[itf][2],np.nan_to_num(1/beamf[1,itf,itf,:csu.lmax])), np.squeeze(W_total[2,itf,:]))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:csu.lmax])

    mapT_combined = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(mapT_combined, fns.get_map('T', 'combined'))
    ClT_combined = trsf.map2cls({'combined':mapT_combined}, {'combined':tmask['030']}, {'combined':pmask['030']}, csu.spectrum_type, csu.lmax-1, freqcomb=['combined-combined'], lmax_mask=csu.lmax_mask)
    io.save_data(ClT_combined, fns.get_spectrum('T', 'combined'))

    maq_lpDXS = hp.smoothing(hp.ma(mapT_combined[1]), np.radians(1))
    mau_lpDXS = hp.smoothing(hp.ma(mapT_combined[2]), np.radians(1))

    mapT_combined_fn = fns.get_spectrum('T', 'combined')
    mapT_combined_smoothed_fn = mapT_combined_fn.replace('.', 'smoothed.')

    io.save_data(np.array([np.zeros_like(maq_lpDXS),mau_lpDXS, mau_lpDXS]), mapT_combined_smoothed_fn)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_fit = True
    bool_propag = False
    store_data = True

    if bool_fit:
        # run_fit()
        run_fit(fitp_old)

    if bool_propag:
        run_propag()
        

def run_propag_ext():
    from typing import Dict
    """
    Follows the SMICA propagation code to combine maps with set of weights.
    Only runs for the chosen bins and up to the max value of bins
    """
    CMB = dict()
    CMB["TQU"] = dict()
    almT, almE, almB = dict(), dict(), dict()

    def load_mask(path_to_mask: str) -> Dict[np.array]:
        """Return mask per frequency as a Dict. Mask is expected to be a numpy array."
        Args:
            path_to_mask: example, '/global/xx/mask030.fits'
        Returns
            dict[np.array] : keys must be ['030', '044', ..] and values are of len(3), i.e. one np.array for each map
        """
        return dict(np.array([]))

    def load_maps(path_to_map: str) -> Dict[np.array]:
        """Return mask per frequency as a Dict. Mask is expected to be a numpy array."
        Args:
            path_to_map: example, '/global/xx/map.fits'
        Returns
            dict[np.array]
        """
        return dict(np.array([]))

    def load_bins():
        pass

    def load_weights():
        pass

    def load_beamf():
        pass

    lmax = 4000

    tmask, pmask, pmask = load_mask()
    bins = load_bins()
    W_total = load_weights()
    maps = load_maps()
    beamf = load_beamf()

    maps = mp.process_all(maps)

    for freq in csu.PLANCKMAPFREQ_f:
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        alms = trsf.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmax-1) # full sky QU->EB
        # almT[det] = alms[0]
        almE[freq] = alms[0]
        almB[freq] = alms[1]

    nalm = int((lmax)*(lmax-1+2)/2)  
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)

    for it, det in enumerate(csu.PLANCKMAPFREQ): #weights do not depend on freqfilter, but almE/B do
        if det in csu.PLANCKMAPFREQ_f:
            print('freq: ', det)
            ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
            # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
            combalmE += hp.almxfl(hp.almxfl(almE[det],np.nan_to_num(1/beamf[1,it,it,:lmax])), np.squeeze(W_total[1,it,:]))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmax])
            combalmB += hp.almxfl(hp.almxfl(almB[det],np.nan_to_num(1/beamf[1,it,it,:lmax])), np.squeeze(W_total[2,it,:]))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmax])

    CMB["TQU"]['out'] = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(CMB["TQU"]['out'], io.fh.cmbmap_smica_path_name) #TODO rename to MVmap or similar
    smica_C_lmin_unsc = np.array(ps.map2cl_spin(qumap=CMB["TQU"]['out'][1:3], spin=2, mask=pmask['100'], lmax=lmax-1,
        lmax_mask=lmax*2))*1e12 #maps are different scale than processed powerspectra from this' package pipeline, thus *1e12
    io.save_data(smica_C_lmin_unsc, io.fh.clmin_smica_path_name)