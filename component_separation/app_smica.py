#!/usr/local/bin/python
"""
app_smica.py: script for executing main functionality of smica.
Depends on all (noise, signal, full) spectra being generated to begin with. Use ``app_powerspectrum.py``, if e.g. noisespectra are missing.
"""

#TODO
# check if fixing LFI covariances to high values solves the issue? 

__author__ = "S. Belkner"

import os
os.environ["OMP_NUM_THREADS"] = "32"
from astropy.io import fits
from scipy import interpolate

import numpy as np

import healpy as hp
import sys
import component_separation.MSC.MSC.pospace as ps
import smica

from component_separation.io import IO
import component_separation.powspec as pw
import component_separation.interface as cslib
import component_separation.transform_map as trsf_m
import component_separation.transform_spec as trsf_s
from component_separation.cs_util import Config
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf

csu = Config()
io = IO(csu)
bins = csu.bins
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
        print(cov_ltot_bnd.shape)

        cov_lN_bnd = hpf.bin_it(cov_lN_s, bins=bins)
        cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T
        print(cov_lN_bnd.shape)

        C_lS_bnd =  hpf.bin_it(cov_lS_s, bins=bins)
        print(C_lS_bnd.shape)
        return cov_ltot_bnd.copy(), cov_lN_bnd.copy(), C_lS_bnd.copy()

    def calc_nmodes(bins, mask):
        nmode = np.ones((bins.shape[0]))
        for idx,q in enumerate(bins):
            rg = np.arange(q[0],q[1]+1) #rg = np.arange(bins[q,0],bins[q,1]+1) correct interpretation?
            nmode[idx] = np.sum(2*rg+1, axis=0)
            fsky = np.mean(mask**2)
            nmode *= fsky
        # print('nmodes: {}, fsky: {}'.format(nmode, fsky))
        return nmode

    _Tscale = "K_CMB"#csu.Tscale
    C_ltot = io.load_powerspectra('full')
    cov_ltot = pw.build_covmatrices(C_ltot, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_ltotEE = cov_ltot[1]
    print(cov_ltotEE.shape)

    C_lN = io.load_powerspectra('noise')
    cov_lN = pw.build_covmatrices(C_lN, _Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_lNEE = cov_lN[1]
    for n in range(cov_lNEE.shape[0]):
        for m in range(cov_lNEE.shape[1]):
            if n != m:
                cov_lNEE[n,m] = np.zeros(shape=cov_lNEE.shape[2])
    print(cov_lNEE.shape)

    C_lS = io.load_powerspectra('signal')
    print(C_lS.shape)
    # Fakes the same shape so pw.build_covmatrices() may be used
    C_lS_shaped = np.zeros_like(C_lN)
    for freqcom in range(C_lS_shaped.shape[0]):
        for specidx in range(C_lS.shape[1]):
            C_lS_shaped[freqcom,specidx,:] = C_lS[0,specidx]
    cov_lS = pw.build_covmatrices(C_lS_shaped, "K_CMB", csu.freqcomb, csu.PLANCKMAPFREQ_f)
    print(cov_lS.shape)
    cov_lSEE = cov_lS[1]
    print(cov_lSEE.shape)

    #Fitting galactic emissivity only
    binname = "SMICA_lowell_bins"
    cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotEE, cov_lNEE, cov_lSEE)
    nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
    smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), None)
    smica_model, hist = cslib.fit_model_to_cov(
        smica_model,
        np.nan_to_num(cov_ltot_bnd),
        nmodes,
        maxiter=100,
        noise_fix=True,
        noise_template=np.nan_to_num(cov_lN_bnd),
        afix=None, qmin=0,
        asyn=None,
        logger=None,
        qmax=len(nmodes),
        no_starting_point=False)
    gal_mixmat = smica_model.get_comp_by_name('gal').mixmat()
    io.save_data(gal_mixmat, io.fh.out_specsmica_path+"gal_mixmat_{}".format(binname) + "_" + io.fh.total_filename)

    #Fitting everything with fixed gal emis
    binname = "SMICA_highell_bins"
    cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotEE, cov_lNEE, cov_lSEE)
    nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
    smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), gal_mixmat)
    smica_model, hist = cslib.fit_model_to_cov(
        smica_model,
        np.nan_to_num(cov_ltot_bnd),
        nmodes,
        maxiter=100,
        noise_fix=True,
        noise_template=np.nan_to_num(cov_lN_bnd),
        afix=None, qmin=0,
        asyn=None,
        logger=None,
        qmax=len(nmodes),
        no_starting_point=False)

    io.save_data(smica_model.get_comp_by_name('cmb').powspec(), io.fh.cmb_specsmica_sc_path_name)
    io.save_data(smica_model.get_comp_by_name('gal').powspec(), io.fh.gal_specsmica_sc_path_name)
    io.save_data(smica_model.get_theta(), io.fh.out_specsmica_path+"theta_{}".format(binname) + "_" + io.fh.total_filename)
    io.save_data(smica_model.covariance4D(), io.fh.out_specsmica_path+"cov4D_{}".format(binname) + "_" + io.fh.total_filename)
    io.save_data(smica_model.covariance(), io.fh.out_specsmica_path+"cov_{}".format(binname) + "_" + io.fh.total_filename)
    io.save_data(hist, io.fh.out_specsmica_path+"hist_{}".format(binname) + "_" + io.fh.total_filename)

    smica_weights_tot = np.zeros(shape=(2,7,len(bins)))
    print(smica_weights_tot.shape)
    smica_weights_tot[0] = pw.cov2weight(np.array([smica_model.covariance()])) #EE
    # io.save_data(smica_weights_tot, path_name)


    ## BB fitting
    ##
    ##
    ##
    cov_ltotBB = cov_ltot[2]
    print(cov_ltotBB.shape)

    cov_lNBB = cov_lN[2]
    for n in range(cov_lNBB.shape[0]):
        for m in range(cov_lNBB.shape[1]):
            if n != m:
                cov_lNBB[n,m] = np.zeros(shape=cov_lNBB.shape[2])
    print(cov_lNBB.shape)

    cov_lSBB = cov_lS[1]
    print(cov_lNBB.shape)

    binname = "SMICA_lowell_bins"
    cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotBB, cov_lNBB, cov_lSBB)
    nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
    smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), None)
    smica_model, hist = cslib.fit_model_to_cov(
        smica_model,
        np.nan_to_num(cov_ltot_bnd),
        nmodes,
        maxiter=100,
        noise_fix=True,
        noise_template=np.nan_to_num(cov_lN_bnd),
        afix=None, qmin=0,
        asyn=None,
        logger=None,
        qmax=len(nmodes),
        no_starting_point=False)
    gal_mixmat = smica_model.get_comp_by_name('gal').mixmat()
    io.save_data(gal_mixmat, io.fh.out_specsmica_path+"gal_mixmat_{}".format(binname) + "_" + io.fh.total_filename)

    #Fitting everything with fixed gal emis
    binname = "SMICA_highell_bins"
    cov_ltot_bnd, cov_lN_bnd, C_lS_bnd = bin_data(binname, cov_ltotBB, cov_lNBB, cov_lSBB)
    for n in range(cov_ltot_bnd.shape[0]):
        cov_lN_bnd[n,:] = trsf_s.apply_smoothing(cov_lN_bnd[n,:])
        for m in range(cov_ltot_bnd.shape[1]):
            # if n != m:
                cov_ltot_bnd[n,m,:] = trsf_s.apply_smoothing(cov_ltot_bnd[n,m,:])
    nmodes = calc_nmodes(getattr(const, binname), pmask['100']) #any mask will do, only fsky needed
    smica_model = cslib.build_smica_model(len(nmodes), np.nan_to_num(cov_lN_bnd), np.nan_to_num(C_lS_bnd), gal_mixmat)
    smica_model, hist = cslib.fit_model_to_cov(
        smica_model,
        np.nan_to_num(cov_ltot_bnd),
        nmodes,
        maxiter=100,
        noise_fix=True,
        noise_template=np.nan_to_num(cov_lN_bnd),
        afix=None, qmin=0,
        asyn=None,
        logger=None,
        qmax=len(nmodes),
        no_starting_point=False)

    io.save_data(smica_model.get_comp_by_name('cmb').powspec(), io.fh.cmb_specsmica_sc_path_name)
    # io.save_data(smica_model.get_comp_by_name('gal').powspec(), io.fh.gal_specsmica_sc_path_name)
    # io.save_data(smica_model.get_theta(), io.fh.out_specsmica_path+"theta_{}".format(binname) + "_" + io.fh.total_filename)
    # io.save_data(smica_model.covariance4D(), io.fh.out_specsmica_path+"cov4D_{}".format(binname) + "_" + io.fh.total_filename)
    # io.save_data(smica_model.covariance(), io.fh.out_specsmica_path+"cov_{}".format(binname) + "_" + io.fh.total_filename)
    # io.save_data(hist, io.fh.out_specsmica_path+"hist_{}".format(binname) + "_" + io.fh.total_filename)

    smica_weights_tot[1] = pw.cov2weight(np.array([smica_model.covariance()])) #BB
    print(smica_weights_tot.shape)
    io.save_data(smica_weights_tot, path_name)


def run_propag():
    """
    Follows the SMICA propagation code to combine maps with set of weights.
    Only runs for the chosen bins and up to the max value of bins
    """
    CMB = dict()
    CMB["TQU"] = dict()
    almT, almE, almB = dict(), dict(), dict()
    lmaxbin = int(bins[-1][1]+1)

    W = io.load_data(io.fh.weight_smica_path_name)

    # full maps
    maps = io.load_plamap(csu.cf, field=(0,1,2), nside_out=csu.nside_out)
    maps = trsf_m.process_all(maps)

    for freq in csu.PLANCKMAPFREQ_f:
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        alms = pw.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmaxbin-1) # full sky QU->EB
        # almT[det] = alms[0]
        almE[freq] = alms[0]
        almB[freq] = alms[1]

    nalm = int((lmaxbin)*(lmaxbin-1+2)/2)  
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    beamf = io.load_beamf(freqcomb=csu.freqcomb)

    xnew = np.arange(0,lmaxbin,1)
    for it, det in enumerate(csu.PLANCKMAPFREQ): #weights do not depend on freqfilter, but almE/B do
        if det in csu.PLANCKMAPFREQ_f:
            ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
            W_Einterp = interpolate.interp1d(np.mean(bins, axis=1), W[0,it,:], bounds_error = False, fill_value='extrapolate')
            #TODO switch to W[2,:] once BB-weights are correctly calculated
            W_Binterp = interpolate.interp1d(np.mean(bins, axis=1), W[1,it,:], bounds_error = False, fill_value='extrapolate')

            # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
            LHFI = "LFI" if int(det)<100 else "HFI"
            if csu.cf['pa']['freqdset'].startswith('NPIPE'):
                LHFI = "HFI"
            
            combalmE += hp.almxfl(hp.almxfl(almE[det],1/beamf[str(det)+'-'+str(det)][LHFI][1].data.field(1)[:lmaxbin]), np.squeeze(W_Einterp(xnew)))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmaxbin])
            combalmB += hp.almxfl(hp.almxfl(almB[det],1/beamf[str(det)+'-'+str(det)][LHFI][1].data.field(2)[:lmaxbin]), np.squeeze(W_Binterp(xnew)))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmaxbin])

    CMB["TQU"]['out'] = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(CMB["TQU"]['out'], io.fh.cmbmap_smica_path_name)
    smica_C_lmin_unsc = np.array(ps.map2cl_spin(qumap=CMB["TQU"]['out'][1:3], spin=2, mask=pmask['100'], lmax=lmaxbin-1,
        lmax_mask=lmaxbin*2))*1e12 #maps are different scale than processed powerspectra from this' package pipeline, thus *1e12
    io.save_data(smica_C_lmin_unsc, io.fh.clmin_smica_path_name)


def run_propag_complete():
    """
    Follows the SMICA propagation code to combine maps with set of weights.
    Only runs for the chosen bins and up to the max value of bins
    """
    CMB = dict()
    CMB["TQU"] = dict()
    almT, almE, almB = dict(), dict(), dict()
    lmaxbin = int(bins[-1][1]+1)
    lmax = csu.lmax

    W_smica = io.load_data(io.fh.weight_smica_path_name)
    W_mv = io.load_data(io.fh.weight_path_name)
    W_total = np.zeros(shape=(*W_mv.shape[:-1], csu.lmax+1))

    xnew = np.arange(0,lmaxbin+1,1)
    for it, det in enumerate(csu.PLANCKMAPFREQ): #weights do not depend on freqfilter, but almE/B do
        if det in csu.PLANCKMAPFREQ_f:
            ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
            W_Einterp = interpolate.interp1d(np.mean(bins, axis=1), W_smica[0,it,:], bounds_error = False, fill_value='extrapolate')
            W_total[1,it] = np.concatenate((W_Einterp(xnew),W_mv[1,it,xnew.shape[0]:]))
            W_Binterp = interpolate.interp1d(np.mean(bins, axis=1), W_smica[1,it,:], bounds_error = False, fill_value='extrapolate')
            W_total[2,it] = np.concatenate((W_Binterp(xnew),W_mv[2,it,xnew.shape[0]:]))


    # full maps
    maps = io.load_plamap(csu.cf, field=(0,1,2), nside_out=csu.nside_out)
    maps = trsf_m.process_all(maps)
    beamf = io.load_beamf(freqcomb=csu.freqcomb)

    for freq in csu.PLANCKMAPFREQ_f:
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        alms = pw.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmax-1) # full sky QU->EB
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
            LHFI = "LFI" if int(det)<100 else "HFI"
            if csu.cf['pa']['freqdset'].startswith('NPIPE'):
                LHFI = "HFI"
            combalmE += hp.almxfl(hp.almxfl(almE[det],1/beamf[str(det)+'-'+str(det)][LHFI][1].data.field(1)[:lmax]), np.squeeze(W_total[1,it,:]))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmax])
#             combalmE = hp.smoothalm(combalmE, fwhm = np.radians(5/60))
            combalmB += hp.almxfl(hp.almxfl(almB[det],1/beamf[str(det)+'-'+str(det)][LHFI][1].data.field(2)[:lmax]), np.squeeze(W_total[2,it,:]))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmax])
#             combalmB = hp.smoothalm(combalmB, fwhm = np.radians(5/60))

    CMB["TQU"]['out'] = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(CMB["TQU"]['out'], io.fh.cmbmap_smica_path_name)
    smica_C_lmin_unsc = np.array(ps.map2cl_spin(qumap=CMB["TQU"]['out'][1:3], spin=2, mask=pmask['100'], lmax=lmax-1,
        lmax_mask=lmax*2))*1e12 #maps are different scale than processed powerspectra from this' package pipeline, thus *1e12
    io.save_data(smica_C_lmin_unsc, io.fh.clmin_smica_path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_fit = True
    bool_propag = False
    bool_propag_complete = True

    if bool_fit:
        run_fit(io.fh.weight_smica_path_name, csu.overwrite_cache)

    if bool_propag:
        run_propag()

    if bool_propag_complete:
        run_propag_complete()