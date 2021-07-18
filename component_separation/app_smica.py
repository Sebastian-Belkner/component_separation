#!/usr/local/bin/python
"""
app_smica.py: script for executing main functionality of smica.
Depends on all (noise, signal, full) spectra being generated to begin with. Use ``app_powerspectrum.py``, if e.g. noisespectra are missing.
"""



#TODO ISSUE: 
#   possibly, K_CMB and K_RJ are not taken into account when storing smica fit data which leads to erronoeus results
#   SMICA fit input should ignore Tscale?   Only use Tscale for deriving weights?


# next steps: for npipemask: switch bins to highellbins. switch to K_RJ
#             for dx12smica: switch to highell bins. switch to K_RJ


__author__ = "S. Belkner"

import os
os.environ["OMP_NUM_THREADS"] = "8"
from astropy.io import fits
from scipy import interpolate

import numpy as np

import healpy as hp
import sys
from typing import Dict, List, Optional, Tuple
import component_separation.MSC.MSC.pospace as ps
import smica
import component_separation.io as csio

import component_separation
from component_separation.io import IO
import component_separation.powspec as pw
import component_separation.interface as cslib
import component_separation.transform_map as trsf_m
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
    def calc_nmodes(bins, mask):
        nmode = np.ones((bins.shape[0]))
        for idx,q in enumerate(bins):
            rg = np.arange(q[0],q[1]+1) #rg = np.arange(bins[q,0],bins[q,1]+1) correct interpretation?
            nmode[idx] = np.sum(2*rg+1, axis=0)
            fsky = np.mean(mask**2)
            nmode *= fsky
        # print('nmodes: {}, fsky: {}'.format(nmode, fsky))
        return nmode

    C_ltot = io.load_powerspectra('full')
    cov_ltot = pw.build_covmatrices(C_ltot, "K_CMB", csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_ltotEE = cov_ltot[1]
    print(cov_ltotEE.shape)

    C_lN = io.load_powerspectra('noise')
    cov_lN = pw.build_covmatrices(C_lN, "K_CMB", csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_lNEE = cov_lN[1]
    print(cov_lNEE.shape)

    C_lS = io.load_powerspectra('signal')
    # Fakes the same shape so pw.build_covmatrices() may be used
    C_lS_shaped = np.zeros_like(C_lN)
    for freqcom in range(C_lS_shaped.shape[0]):
        C_lS_shaped[freqcom,1,:] = C_lS[0,1]
    cov_lS = pw.build_covmatrices(C_lS_shaped, "K_CMB", csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov_lSEE = cov_lS[1]
    print(cov_lNEE.shape)
    

    cov_ltot_bnd = hpf.bin_it(cov_ltotEE, bins=bins)
    print(cov_ltot_bnd.shape)

    cov_lN_bnd = hpf.bin_it(cov_lNEE, bins=bins)
    cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T
    print(cov_lN_bnd.shape)

    C_lS_bnd =  hpf.bin_it(cov_lSEE, bins=bins)
    print(C_lS_bnd.shape)


    nmodes = calc_nmodes(bins, pmask['100']) #any mask will do, only fsky needed
    smica_model = cslib.build_smica_model(len(nmodes), cov_lN_bnd, C_lS_bnd)

    smica_model, hist = cslib.fit_model_to_cov(
        smica_model,
        cov_ltot_bnd,
        nmodes,
        maxiter=50,
        noise_fix=True,
        noise_template=cov_lN_bnd,
        afix=None, qmin=0,
        asyn=None,
        logger=None,
        qmax=len(nmodes),
        no_starting_point=False)

    io.save_data(smica_model.get_comp_by_name('cmb').powspec(), io.fh.cmb_specsmica_sc_path_name)
    io.save_data(smica_model.get_theta(), io.fh.out_specsmica_path+"theta.npy")
    io.save_data(smica_model.covariance4D(), io.fh.out_specsmica_path+"cov4D.npy")
    io.save_data(smica_model.covariance(), io.fh.out_specsmica_path+"cov.npy")
    io.save_data(hist, io.fh.out_specsmica_path+"hist.npy")

    #TODO smica needs to run for both BB and EE, as BB-weights are needed for later map generation
    zer = np.zeros_like(smica_model.covariance())
    smica_cov_full = np.zeros(shape=(len(csu.PLANCKSPECTRUM), *zer.shape))
    
    smica_cov_full[1] = smica_model.covariance()
    smica_cov_full_inv_ltot = pw.invert_covmatrices(smica_cov_full)
    smica_weights_tot = pw.calculate_weights(smica_cov_full_inv_ltot, csu.PLANCKMAPFREQ[:-2], csu.Tscale)
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
            W_Einterp = interpolate.interp1d(np.mean(bins, axis=1), W[1,it,:], bounds_error = False, fill_value='extrapolate')
            #TODO switch to W[2,:] once BB-weights are correctly calculated
            W_Binterp = interpolate.interp1d(np.mean(bins, axis=1), W[1,it,:], bounds_error = False, fill_value='extrapolate')

            # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
            LHFI = "LFI" if int(det)<100 else "HFI"
            combalmE += hp.almxfl(hp.almxfl(almE[det],1/beamf[str(det)+'-'+str(det)][LHFI][1].data.field(1)[:lmaxbin]), np.squeeze(W_Einterp(xnew)))
            combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmaxbin])
            combalmB += hp.almxfl(hp.almxfl(almB[det],1/beamf[str(det)+'-'+str(det)][LHFI][1].data.field(2)[:lmaxbin]), np.squeeze(W_Binterp(xnew)))
            combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmaxbin])

    CMB["TQU"]['out'] = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(CMB["TQU"]['out'], io.fh.cmbmap_smica_path_name)
    smica_C_lmin_unsc = np.array(ps.map2cl_spin(qumap=CMB["TQU"]['out'][1:3], spin=2, mask=pmask['100'], lmax=lmaxbin-1,
        lmax_mask=lmaxbin*2))*1e12 #maps are different scale than processed powerspectra from this' package pipeline, thus *1e12
    io.save_data(smica_C_lmin_unsc, io.fh.clmin_smica_path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_fit = True
    bool_propag = False

    if bool_fit:
        run_fit(io.fh.weight_smica_path_name, csu.overwrite_cache)

    if bool_propag:
        run_propag()