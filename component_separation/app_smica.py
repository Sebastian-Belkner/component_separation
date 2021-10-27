#!/usr/local/bin/python

"""
app_smica.py: script for executing main functionality of smica.
Depends on all (noise, signal, full) spectra being generated to begin with. Use ``app_powerspectrum.py``, if e.g. noisespectra are missing.

To let SMICA run sucessfully, we have adapted the following:
1. For B-fit, set all C_ltot bins < 0 to zero.
2. smica.mocel.quasi_newton()'s local diag(CRB): I assume all elements to be positive
"""

__author__ = "S. Belkner"

import os
from typing import Dict, List, Optional, Tuple

import healpy as hp
import numpy as np

import component_separation.covariance as cv
import component_separation.interface as cslib
import component_separation.map as mp
import component_separation.MSC.MSC.pospace as ps
import component_separation.transformer as trsf
from component_separation.cs_util import Config
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf
from component_separation.io import IO

os.environ["OMP_NUM_THREADS"] = "32"
csu = Config()
io = IO(csu)
tmask, pmask, pmask =  io.load_one_mask_forallfreq() # io.load_mask_per_freq(csu.nside_out[0])


def run_fit(path_name, overw):
    """
    Runs SMICA using,
        * signal estimator: C_lS_in.py 
        * noise estimator: io.noise_unsc_path_name
        * empiric data to be fitted: freqdset attribute from config

    Result is,
        * signal estimator: C_lS_out.py
        * channel weights: io.weight_path + "SMICAWEIG_" + cf['pa']["Tscale"] + "_" + io.total_filename
    """
    def bin_data(binname, cov_ltot_s, cov_lN_s, cov_lS_s):
        bins = getattr(const, binname)
        cov_ltot_bnd = hpf.bin_it(cov_ltot_s, bins=bins)

        cov_lN_bnd = hpf.bin_it(cov_lN_s, bins=bins)
        cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T

        C_lS_bnd =  hpf.bin_it(cov_lS_s, bins=bins)
        return cov_ltot_bnd.copy(), cov_lN_bnd.copy(), C_lS_bnd.copy()

    def calc_nmodes(bins, mask):
        nmode = np.ones((bins.shape[0]))
        for idx, q in enumerate(bins):
            rg = np.arange(q[0],q[1]+1)
            nmode[idx] = np.sum(2*rg+1, axis=0)
            fsky = np.mean(mask**2)
            nmode *= fsky
        # print('nmodes: {}, fsky: {}'.format(nmode, fsky))
        return nmode

    _Tscale = "K_CMB"#csu.Tscale
    C_ltot = io.load_powerspectra('full')
    cov_ltot = cv.build_covmatrices(C_ltot, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_ltot = cv.cov2cov_smooth(cov_ltot, cutoff=800)

    C_lN = io.load_powerspectra('noise')
    cov_lN = cv.build_covmatrices(C_lN, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_lN = cv.cov2cov_smooth(cov_lN, cutoff=800)

    C_lS = io.load_powerspectra('signal')
    # Fakes the same shape so pw.build_covmatrices() may be used
    C_lS_shaped = np.zeros_like(C_lN)
    for freqcom in range(C_lS_shaped.shape[0]):
        for specidx in range(C_lS.shape[1]):
            C_lS_shaped[freqcom,specidx,:] = C_lS[0,specidx]
    cov_lS = cv.build_covmatrices(C_lS_shaped, "K_CMB", csu.freqcomb, csu.PLANCKMAPFREQ_f)

    if True:
        ## EE fitting
        ##
        ##
        cov_ltotEE = cov_ltot[1]
        cov_lNEE = cov_lN[1]
        cov_lSEE = cov_lS[1]

        #Fitting galactic emissivity only
        binname = "SMICA_lowell_bins"
        cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotEE, cov_lNEE, cov_lSEE)
        nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
        smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), None)
        smica_model, hist = cslib.fit_model_to_cov(
            smica_model,
            cov_ltot_bnd,
            nmodes,
            maxiter=50,
            noise_template=np.where(cov_lN_bnd<4e-3, 0.0, 1.0),#
            afix=None)
        EEgal_mixmat = smica_model.get_comp_by_name('gal').mixmat()


        #Fitting everything with fixed gal emis
        binname = "SMICA_highell_bins"
        cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotEE, cov_lNEE, cov_lSEE)
        nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
        smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), EEgal_mixmat)#)
        smica_model, hist = cslib.fit_model_to_cov(
            smica_model,
            cov_ltot_bnd,
            nmodes,
            maxiter=50,
            noise_template=np.where(cov_lN_bnd<4e-3, 0.0, 1.0),#,#None,#
            afix=None)

        EEsmica_cmb = smica_model.get_comp_by_name('cmb').powspec()[0]
        EEsmica_gal = np.array([smica_model.get_comp_by_name('gal').powspec()])
        EEsmica_theta = smica_model.get_theta()
        EEsmica_cov4D = np.array([smica_model.covariance4D()])
        EEsmica_cov = smica_model.covariance()
        EEsmica_hist = hist
        if store_data:
            io.save_data(EEsmica_cmb, io.fh.cmb_specsmica_sc_path_name)
            io.save_data(EEsmica_gal, io.fh.gal_specsmica_sc_path_name)
            io.save_data(EEsmica_theta, io.fh.out_specsmica_path+"theta_{}".format(binname) + "_" + io.fh.total_filename)
            io.save_data(EEsmica_cov4D, io.fh.out_specsmica_path+"cov4D_{}".format(binname) + "_" + io.fh.total_filename)
            io.save_data(EEsmica_cov, io.fh.out_specsmica_path+"cov_{}".format(binname) + "_" + io.fh.total_filename)
            io.save_data(EEsmica_hist, io.fh.out_specsmica_path+"hist_{}".format(binname) + "_" + io.fh.total_filename)
            io.save_data(np.array([EEgal_mixmat]), io.fh.out_specsmica_path+"gal_mixmat_{}".format(binname) + "_" + io.fh.total_filename)
        
        bins = getattr(const, binname)
        smica_weights_tot = np.zeros(shape=(2,7,len(bins)))
        smica_weights_tot[0] = cv.cov2weight(np.array([smica_model.covariance()])) #EE
        # io.save_data(smica_weights_tot, path_name)

        print(20*"===")
        print(20*"===")
        print("EE fitting done")
        print(20*"===")
        print(20*"===")

    if True:
        ## BB fitting
        ##
        ##
        ##
        cov_ltotBB = cov_ltot[2]
        cov_lNBB = cov_lN[2]
        cov_lSBB = cov_lS[2]


        binname = "SMICA_lowell_bins"
        cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotBB, cov_lNBB, cov_lSBB)
        nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
        smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), None, B_fit=True)
        cov_ltot_bnd[cov_ltot_bnd<0] = 0.0 #Reminder: this would not work for TE!
        print("SMICA model built for SMICA_lowell_bins")
        smica_model, hist = cslib.fit_model_to_cov(
            smica_model,
            cov_ltot_bnd,
            nmodes,
            maxiter=50,
            noise_template=np.where(cov_lN_bnd<4e-3, 0.0, 1.0),#None,#
            afix=None)
        BBgal_mixmat = smica_model.get_comp_by_name('gal').mixmat()
        print("SMICA model fitted for SMICA_lowell_bins")


        #Fitting everything with fixed gal emis
        binname = "SMICA_highell_bins"
        cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotBB, cov_lNBB, cov_lSBB)
        cov_ltot_bnd[cov_ltot_bnd<0] = 0.0 #Reminder: this would not work for TE!
        nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
        smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), BBgal_mixmat, B_fit=True)
        print("SMICA model built for SMICA_highell_bins")
        smica_model, hist = cslib.fit_model_to_cov(
            smica_model,
            cov_ltot_bnd,
            nmodes,
            maxiter=50,
            noise_template=np.where(cov_lN_bnd<4e-3, 0.0, 1.0),#None,
            afix=None)
        print("SMICA model fitted for SMICA_highell_bins")

        EEBBsmica_cmb = np.concatenate((EEsmica_cmb, smica_model.get_comp_by_name('cmb').powspec()[0]))
        EEBBsmica_gal = np.concatenate((EEsmica_gal, [smica_model.get_comp_by_name('gal').powspec()]))
        EEBBsmica_theta = np.concatenate((EEsmica_theta, smica_model.get_theta()))
        EEBBsmica_cov4D = np.concatenate((EEsmica_cov4D, [smica_model.covariance4D()]))
        EEBBsmica_cov = np.concatenate((EEsmica_cov, smica_model.covariance()))
        # EEBBsmica_hist = np.concatenate(([EEsmica_hist], [hist]))
        EEBBgal_mixmat = np.concatenate(([EEgal_mixmat],[BBgal_mixmat]))
        if store_data:
            io.save_data(EEBBsmica_cmb, io.fh.cmb_specsmica_sc_path_name)
            io.save_data(EEBBsmica_gal, io.fh.gal_specsmica_sc_path_name)
            io.save_data(EEBBsmica_theta, io.fh.out_specsmica_path+"theta_{}".format(binname) + "_" + io.fh.total_filename)
            io.save_data(EEBBsmica_cov4D, io.fh.out_specsmica_path+"cov4D_{}".format(binname) + "_" + io.fh.total_filename)
            print(EEBBsmica_cov4D.shape)
            io.save_data(EEBBsmica_cov, io.fh.out_specsmica_path+"cov_{}".format(binname) + "_" + io.fh.total_filename)
            # io.save_data(EEBBsmica_hist, io.fh.out_specsmica_path+"hist_{}".format(binname) + "_" + io.fh.total_filename)
            io.save_data(EEBBgal_mixmat, io.fh.out_specsmica_path+"gal_mixmat_{}".format(binname) + "_" + io.fh.total_filename)

            smica_weights_tot[1] = cv.cov2weight(np.array([smica_model.covariance()])) #BB
            io.save_data(smica_weights_tot, path_name)


def run_propag():
    """
    Follows the SMICA propagation code to combine maps with set of weights.
    Only runs for the chosen bins and up to the max value of bins
    """
    import copy
    cfDXS = copy.deepcopy(csu.cf)
    cfDXS['pa']['freqdset'] = 'NPIPE'
    cfDXS['pa']['mskset'] = 'lens'
    csuDXS = Config(cfDXS)
    ioDXS = IO(csuDXS)
    tmask, pmask, pmask = ioDXS.load_one_mask_forallfreq()
    bins = getattr(const, "SMICA_highell_bins")

    CMB = dict()
    CMB["TQU"] = dict()
    almT, almE, almB = dict(), dict(), dict()
    lmax = csu.lmax

    W_smica = io.load_data(io.fh.weight_smica_path_name)
    W_mv = io.load_data(io.fh.weight_path_name)
    W_total = hpf.interp_smica_mv_weights(W_smica, W_mv, bins, 4001)
    W_total[:,:,0:2] = 0.0

    # full maps
    maps = io.load_plamap(csu.cf, field=(0,1,2), nside_out=csu.nside_out)
    maps = mp.process_all(maps)
    beamf = io.load_beamf(freqcomb=csu.freqcomb)

    for freq in csu.PLANCKMAPFREQ_f:
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        alms = cv.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmax-1) # full sky QU->EB
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

def run_propag_ext():
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

if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_fit = True
    bool_propag = False
    store_data = True

    if bool_fit:
        run_fit(io.fh.weight_smica_path_name, csu.overwrite_cache)
        if not store_data:
            print("Nothing has been stored as per user-request.")

    if bool_propag:
        run_propag()
