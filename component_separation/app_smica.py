#!/usr/local/bin/python

"""
app_smica.py: script for executing main functionality of smica.
Depends on all (noise, signal, full) spectra being generated to begin with. Use ``app_powerspectrum.py``, if e.g. noisespectra are missing.

To let SMICA run sucessfully, we have adapted the following:
1. For B-fit, set all C_ltot bins < 0 to zero.
2. smica.mocel.quasi_newton()'s local diag(CRB): assume all elements to be positive
"""

__author__ = "S. Belkner"

import os, sys

import healpy as hp
import numpy as np

import component_separation.covariance as cv
import component_separation.smica_interface as smint
import component_separation.map as mp
import component_separation.transformer as trsf

from component_separation.cs_util import Helperfunctions as hpf

from component_separation.cs_util import Config
from component_separation.cs_util import Smica_bins
from component_separation.cs_util import Filename_gen_SMICA as fns_gen
from component_separation.cs_util import Filename_gen as fn_gen
from component_separation.io import IO

experiment='Pico'
simid=0
cutoff=900
os.environ["OMP_NUM_THREADS"] = "32"
csu = Config(experiment=experiment)
fn = fn_gen(csu)
fns = fns_gen(csu)
io = IO(csu)


apo = csu.spectrum_type == 'pseudo'
tmask_fn = fn.get_mask('T', apodized=apo)
pmask_fn = fn.get_mask('P', apodized=apo)
tmask_sg = io.load_mask(tmask_fn)
pmask_sg = io.load_mask(pmask_fn)
tmask, pmask = dict(), dict()
for FREQ in csu.FREQ:
    if FREQ not in csu.FREQFILTER:
        tmask[FREQ] = tmask_sg
        pmask[FREQ] = pmask_sg
smica_params = dict({
    'cov': dict(), 
    "cov4D": dict(), 
    "CMB": dict(),
    "gal": dict(),
    "gal_mm": dict(), 
    "w": dict()})


def planck_cutoff(fr):
    # KEEP. Cuts LFI channels for ell=700 as they cause numerical problems
    return {
        30: cutoff,
        44: cutoff,
        70: cutoff,
        100: lmaxp1,
        143: lmaxp1,
        217: lmaxp1,
        353: lmaxp1
    }[fr]


def pico_cutoff(fr):
    return {
        21: cutoff,
        25: cutoff,
        30: cutoff,
        36: cutoff,
        43: cutoff,
        52: cutoff,
        62: cutoff,
        75: cutoff,
        90: cutoff,
        108: cutoff,
        129: cutoff,
        155: cutoff,
        186: cutoff,
        223: cutoff,
        268: cutoff,
        321: cutoff,
        385: cutoff,
        462: cutoff,
        555: cutoff,
        666: cutoff,
        799: cutoff
    }[fr]


def bin_data(bins, cov_ltot_s, cov_lN_s, cov_lS_s):

    cov_ltot_bnd = hpf.bin_it(cov_ltot_s, bins=bins)
    cov_lN_bnd = hpf.bin_it(cov_lN_s, bins=bins)
    cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T
    C_lS_bnd =  hpf.bin_it(cov_lS_s, bins=bins)
    cov_ltot_bnd[cov_ltot_bnd<0] = 0.0 #Reminder: this would not work for TE!

    return cov_ltot_bnd.copy(), cov_lN_bnd.copy(), C_lS_bnd.copy()


def smooth_data(covltot, covlN, covlS):

    cutoff = 900
    covlT_smoothed = cv.cov2cov_smooth(covltot, cutoff=cutoff)
    covlN_smoothed = cv.cov2cov_smooth(covlN, cutoff=cutoff)
    covlS_smoothed = cv.cov2cov_smooth(covlS, cutoff=cutoff)

    return np.nan_to_num(covlT_smoothed), np.nan_to_num(covlN_smoothed), np.nan_to_num(covlS_smoothed)


def fitp(cov_ltot, cov_lN, cov_lS, nmodes, gal_mixmat, B_fit):

    smica_model = smint.build_smica_model(len(nmodes), np.nan_to_num(cov_lN), np.nan_to_num(cov_lS), csu.mskset, gal_mixmat, B_fit)
    smica_model, hist = smint.fit_model_to_cov_old(
        smica_model,
        cov_ltot,
        nmodes,
        maxiter=50,
        noise_template=np.where(cov_lN<2e-15, 0.0, 1.0),#np.where(cov_lN, 0.0, 1.0),
        afix=None)

    return smica_model


def extract_model_parameters(smica_model):

    smica_cmb = smica_model.get_comp_by_name('cmb').powspec()[0]
    smica_gal = np.array([smica_model.get_comp_by_name('gal').powspec()])
    smica_cov = np.array([smica_model.covariance()])
    smica_cov4D = np.array([smica_model.covariance4D()])

    smica_weights_tot = cv.cov2weight(smica_cov, np.array(csu.FREQ_f))

    return smica_cov, smica_cov4D, smica_cmb, smica_gal, smica_weights_tot


def run_fit(fit):

    _Tscale = "K_CMB"
    Cltot = io.load_data(fn.get_spectrum("T", "non-separated",simid=simid))
    covltot = cv.build_covmatrices(Cltot, _Tscale, csu.freqcomb, csu.FREQ_f, pico_cutoff, cutoff)

    ClN = io.load_data(fn.get_spectrum("N", "non-separated",simid=simid))
    covlN = cv.build_covmatrices(ClN, _Tscale, csu.freqcomb, csu.FREQ_f, pico_cutoff, cutoff)

    ClS = io.load_data(fns.get_spectrum("S", "non-separated",simid=simid))
    covlS = cv.build_covmatrices(ClS, _Tscale, csu.freqcomb, csu.FREQ_f, pico_cutoff, cutoff)

    nmodes_lowell = smint.calc_nmodes(Smica_bins.SMICA_lowell_bins, pmask['100'])
    nmodes_highell = smint.calc_nmodes(Smica_bins.SMICA_highell_bins, pmask['100'])

    covltot_smoothed, covlN_smoothed, ClS_smoothed = smooth_data(covltot, covlN, covlS) # covltot, covlN, covlS

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
    lmax_loc = 2500
    bins = csu.bins
    W_smica = io.load_data(fns.get_misc('w', simid=simid))
    W_mv = io.load_data(fn.get_misc('w', simid=simid))
    W_total = hpf.interp_smica_mv_weights(W_smica, W_mv, bins, 4001)
    W_total[:,:,0:2] = 0.0

    nalm = int((lmax_loc+1)*(lmax_loc+2)/2) 
    alm = np.zeros(shape=(len(csu.FREQ_f),3,nalm))
    
    maps = dict()
    for FREQ in csu.FREQ:
        if FREQ not in csu.FREQFILTER:
            inpath_map_pla_name = fn.get_d(FREQ, "T", simid=simid)
            print("inpath_map_pla_name: {}".format(inpath_map_pla_name))
            nside_out = csu.nside_out[0] if int(FREQ)<100 else csu.nside_out[1]
            maps[FREQ] = io.load_d(inpath_map_pla_name, field=(0,1,2), nside_out=nside_out)

    maps = mp.process_all(maps)
    beamf = io.load_beamf(csu.freqcomb, csu.lmax, csu.freqdatsplit)

    for itf, freq in enumerate(csu.FREQ_f):
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        if apo:
            alm[itf][1:] = hp.map2alm(np.array([n*hp.ud_grade(pmask[freq], nside_out=ns) for n in maps[freq]]), lmax_loc)[1:] # full sky QU->EB        #TODO no TT at the moment
        else:
            alm[itf][1:] = trsf.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmax_loc) # full sky QU->EB        #TODO no TT at the moment
 
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    beam_e = hp.gauss_beam(np.radians(5/60), 4100, pol = True)[:,1]
    beam_b = hp.gauss_beam(np.radians(5/60), 4100, pol = True)[:,2]

    for itf, det in enumerate(csu.FREQ): #weights do not depend on FREQFILTER, but almE/B do
        if det in csu.FREQ_f:
            print('freq: ', det)
            ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
            # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
            combalmE += hp.almxfl(
                hp.almxfl(
                    hp.almxfl(
                        alm[itf][1], np.nan_to_num(1/beamf[1,itf,itf,:lmax_loc])),
                    beam_e[:lmax_loc]),
                np.squeeze(W_total[1,itf,:lmax_loc]))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmax_loc])
            combalmB += hp.almxfl(
                hp.almxfl(
                    hp.almxfl(
                        alm[itf][2], np.nan_to_num(1/beamf[2,itf,itf,:lmax_loc])),
                        beam_b[:lmax_loc]),
                np.squeeze(W_total[2,itf,:lmax_loc]))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmax_loc])

    mapT_combined = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(mapT_combined, fns.get_map('T', 'combined', simid=simid))
    ClT_combined = trsf.map2cls({'combined':mapT_combined}, {'combined':tmask['030']}, {'combined':pmask['030']}, csu.spectrum_type, lmax_loc, freqcomb=['combined-combined'], lmax_mask=csu.lmax_mask)
    io.save_data(ClT_combined, fns.get_spectrum('T', 'combined', simid=simid))

    maq_lpDXS = hp.smoothing(hp.ma(mapT_combined[1]), np.radians(1))
    mau_lpDXS = hp.smoothing(hp.ma(mapT_combined[2]), np.radians(1))

    mapT_combined_fn = fns.get_map('T', 'combined', simid=simid)
    mapT_combined_smoothed_fn = mapT_combined_fn.replace('.', 'smoothed.')

    io.save_data(np.array([np.zeros_like(maq_lpDXS),maq_lpDXS, mau_lpDXS]), mapT_combined_smoothed_fn)


def run_propag_mv():
    lmax_loc = 2000
    W_mv = io.load_data(fn.get_misc('w', simid=simid))
    W_total = W_mv

    nalm = int((lmax_loc+1)*(lmax_loc+2)/2) 
    alm = np.zeros(shape=(len(csu.FREQ_f),3,nalm))

    beamf = io.load_beamf(csu.freqcomb, csu.lmax, csu.freqdatsplit)
    
    maps = dict()
    for FREQ in csu.FREQ:
        if FREQ not in csu.FREQFILTER:
            inpath_map_pla_name = fn.get_d(FREQ, "T", simid=simid)
            print("inpath_map_pla_name: {}".format(inpath_map_pla_name))
            nside_out = csu.nside_out[0] if int(FREQ)<100 else csu.nside_out[1]
            maps[FREQ] = io.load_d(inpath_map_pla_name, field=(0,1,2), nside_out=nside_out)

    maps = mp.process_all(maps)

    for itf, freq in enumerate(csu.FREQ_f):
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        if apo:
            alm[itf][1:] = hp.map2alm(np.array([n*hp.ud_grade(pmask[freq], nside_out=ns) for n in maps[freq]]), lmax_loc)[1:] # full sky QU->EB        #TODO no TT at the moment
        else:
            alm[itf][1:] = trsf.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmax_loc) # full sky QU->EB        #TODO no TT at the moment
 
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    beam_e = hp.gauss_beam(np.radians(0.005/60), 4100, pol = True)[:,1]
    beam_b = hp.gauss_beam(np.radians(0.005/60), 4100, pol = True)[:,2]

    for itf, det in enumerate(csu.FREQ_f):
        print('freq: ', det)
        ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
        # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
        combalmE += hp.almxfl(
            hp.almxfl(
                hp.almxfl(
                    alm[itf][1], np.nan_to_num(1/beamf[1,itf,itf,:lmax_loc])),
                beam_e[:lmax_loc]),
            np.squeeze(W_total[1,itf,:lmax_loc]))
        combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmax_loc])
        combalmB += hp.almxfl(
            hp.almxfl(
                hp.almxfl(
                    alm[itf][2], np.nan_to_num(1/beamf[2,itf,itf,:lmax_loc])),
                    beam_b[:lmax_loc]),
            np.squeeze(W_total[2,itf,:lmax_loc]))
        combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmax_loc])

    mapT_combined = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(mapT_combined, fns.get_map('T', 'combined', simid=simid))
    ClT_combined = trsf.map2cls({'combined':mapT_combined}, {'combined':tmask[csu.FREQ_f[0]]}, {'combined':pmask[csu.FREQ_f[0]]}, csu.spectrum_type, lmax_loc, freqcomb=['combined-combined'], lmax_mask=csu.lmax_mask)
    io.save_data(ClT_combined, fns.get_spectrum('T', 'combined', simid=simid))

    maq_lpDXS = hp.smoothing(hp.ma(mapT_combined[1]), np.radians(1))
    mau_lpDXS = hp.smoothing(hp.ma(mapT_combined[2]), np.radians(1))

    mapT_combined_fn = fns.get_map('T', 'combined', simid=simid)
    mapT_combined_smoothed_fn = mapT_combined_fn.replace('.', 'smoothed.')

    io.save_data(np.array([np.zeros_like(maq_lpDXS),maq_lpDXS, mau_lpDXS]), mapT_combined_smoothed_fn)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_fit = False
    bool_propag = True
    store_data = True

    if bool_fit:
        # run_fit()
        run_fit(fitp)

    if bool_propag:
        run_propag_mv()
        

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

    for freq in csu.FREQ_f:
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

    for it, det in enumerate(csu.FREQ): #weights do not depend on FREQFILTER, but almE/B do
        if det in csu.FREQ_f:
            print('freq: ', det)
            ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
            # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
            combalmE += hp.almxfl(hp.almxfl(almE[det],np.nan_to_num(1/beamf[1,it,it,:lmax])), np.squeeze(W_total[1,it,:]))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmax])
            combalmB += hp.almxfl(hp.almxfl(almB[det],np.nan_to_num(1/beamf[1,it,it,:lmax])), np.squeeze(W_total[2,it,:]))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmax])

    CMB["TQU"]['out'] = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(CMB["TQU"]['out'], io.fh.cmbmap_smica_path_name) #TODO rename to MVmap or similar
    smica_C_lmin_unsc = np.array(trsf.map2cl_ss(qumap=CMB["TQU"]['out'][1:3], spin=2, mask=pmask['100'], lmax=lmax-1,
        lmax_mask=lmax*2))*1e12 #maps are different scale than processed powerspectra from this' package pipeline, thus *1e12
    io.save_data(smica_C_lmin_unsc, io.fh.clmin_smica_path_name)