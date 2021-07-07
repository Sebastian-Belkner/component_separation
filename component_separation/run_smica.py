#!/usr/local/bin/python
"""
run_powerspectrum.py: script for executing main functionality of component_separation
"""


#TODO check single frequency transferfunctions

__author__ = "S. Belkner"

import os
os.environ["OMP_NUM_THREADS"] = "8"
import json
import logging
from astropy.io import fits
from scipy import interpolate
import logging.handlers

import numpy as np

import platform
import healpy as hp
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple
import component_separation.MSC.MSC.pospace as ps
import pandas as pd
import smica

import component_separation
import matplotlib.pyplot as plt
import component_separation.io as io
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

# LOGFILE = 'data/tmp/logging/messages.log'
# logger = logging.getLogger("")
# handler = logging.handlers.RotatingFileHandler(
#         LOGFILE, maxBytes=(1048576*5), backupCount=0
# )
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

nside_out = cf['pa']['nside_out'] if cf['pa']['nside_out'] is not None else cf['pa']['nside_desc_map']
num_sim = cf['pa']["num_sim"]

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
detectors = csu.PLANCKMAPFREQ_f
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]
tmask, pmask, pmask = io.load_mask_per_freq(nside_out[0]) #io.load_one_mask_forallfreq(nside_out=nside_out)


def set_logger(loglevel=logging.INFO):
    logger.setLevel(loglevel)
    logging.StreamHandler(sys.stdout)


def map2spec(data, tmask, pmask):
    spectrum = pw.tqupowerspec(data, tmask, pmask, lmax, lmax_mask)
    return spectrum


def specsc2weights(spectrum, Tscale):
    cov = pw.build_covmatrices(spectrum, lmax, freqfilter, specfilter, Tscale)
    cov_inv_l = pw.invert_covmatrices(cov, lmax)
    weights = pw.calculate_weights(cov_inv_l, lmax, Tscale)
    return weights


def spec_weight2weighted_spec(spectrum, weights):
    alms = pw.spec2alms(spectrum)
    alms_w = pw.alms2almsxweight(alms, weights)
    spec = pw.alms2cls(alms_w)
    return spec


def postprocess_spectrum(data, freqcomb, smoothing_window, max_polynom):
    if smoothing_window > 0 or max_polynom > 0:
        spec_sc = pw.smoothC_l(data, smoothing_window=smoothing_window, max_polynom=max_polynom)
    spec_sc = pw.apply_scale(data, scale=cf['pa']["Spectrum_scale"])
    beamf = io.load_beamf(freqcomb=freqcomb)
    spec_scbf = pw.apply_beamfunction(spec_sc, beamf, lmax, specfilter)
    return spec_scbf


def calc_nmodes(bins, mask):
    nmode = np.ones((bins.shape[0]))
    for idx,q in enumerate(bins):
        rg = np.arange(q[0],q[1]+1) #rg = np.arange(bins[q,0],bins[q,1]+1) correct interpretation?
        nmode[idx] = np.sum(2*rg+1, axis=0)
        fsky = np.mean(mask**2)
        nmode *= fsky
    # print('nmodes: {}, fsky: {}'.format(nmode, fsky))
    return nmode


def build_smica_model(Q, N_cov_bn, C_lS_bnd):
    # Noise part
    nmap = N_cov_bn.shape[0]
    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_ampl(np.ones((nmap,1)), fixed="all")
    noise.set_powspec(np.nan_to_num(N_cov_bn), fixed="all") # where N is a (nmap, Q) array with noise spectra
    # print("noise cov: {}".format(N_cov_bn))

    # CMB part
    cmb = smica.Source1D (nmap, Q, name='cmb')
    acmb = np.ones((nmap,1)) # if cov in cmb unit
    cmb.set_mixmat(acmb, fixed='null')
    
    cmbcq = C_lS_bnd[0,0,:]
    cmb.set_powspec(cmbcq) # where cmbcq is a starting point for cmbcq like binned lcdm

    # Galactic foreground part
    # cmb.set_powspec(cmbcq*0, fixed='all') # B modes fit
    dim = 3
    gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6
    gal.fix_mixmat("null")
    # galmixmat = np.ones((nmap,dim))*0.1
    # gal.set_mixmat(galmixmat, fixed='null')

    model = smica.Model(complist=[cmb, gal, noise])
    return model


def fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_fix=False, noise_template=None, afix=None, qmin=0, asyn=None, logger=None, qmax=None, no_starting_point=False):
    """ Fit the model to empirical covariance.
    
    """
    cg_maxiter = 1
    cg_eps = 1e-20
    nmap = stats.shape[0]
    cmb,gal,noise = model._complist
    print('dim is {}'.format(model.dim))

    # find a starting point
    acmb = model.get_comp_by_name("cmb").mixmat()
    fixed = True
    if fixed:
        afix = 1-cmb._mixmat.get_mask() #0:free 1:fixed
        cfix = 1-cmb._powspec.get_mask() #0:free 1:fixed
    else:
        afix = 0-cmb._mixmat.get_mask() #0:free 1:fixed
        cfix = 0-cmb._powspec.get_mask() #0:free 1:fixed

    cmbcq = np.squeeze(model.get_comp_by_name("cmb").powspec())
    polar = True if acmb.shape[1]==2 else False

    print("starting point chosen.")

    if fixed: # TODO; did i interpret this right?  if not is_mixmat_fixed(model)
        if not no_starting_point:
            print(acmb.shape)
            model.ortho_subspace(stats, nmodes, acmb, qmin=qmin, qmax=qmax)
            cmb.set_mixmat(acmb, fixed=afix)
            if asyn is not None:
                agfix = 1-gal._mixmat.get_mask()
                ag = gal.mixmat()
                agfix[:,-2:] = 1
                ag[0:int(nmap/2),-2] = asyn
                ag[int(nmap/2):,-1]  = asyn
                gal.set_mixmat(ag, fixed=agfix)

    print('starting quasi newton')
    model.quasi_newton(np.abs(stats), nmodes)
    print('starting set_powspec 1')
    cmb.set_powspec (cmbcq, fixed=cfix)
    print('starting close_form')
    model.close_form(stats)
    print('starting set_powspec 2')
    cmb.set_powspec (cmbcq, fixed=cfix)
    cmb.set_powspec (cmbcq, fixed=cfix)
    mm = model.mismatch(stats, nmodes, exact=True)
    mmG = model.mismatch(stats, nmodes, exact=True)

    # start CG/close form
    for i in range(maxiter):
        # fit mixing matrix
        gal.fix_powspec("null")
        cmbfix = "null" if polar else cfix 
        cmb.fix_powspec(cmbfix)
        model.conjugate_gradient (stats, nmodes,maxiter=cg_maxiter, avextol=cg_eps)
        if 0:#logger is not None:
            np.set_printoptions(precision=5)
            logger.info(str(cmb.mixmat()/acmb))
        # fit power spectra
        gal.fix_powspec("null")
        if mmG!=model.mismatch(stats, nmodes, exact=True):
            cmbfix = cfix if polar else "all" 
            cmb.fix_powspec(cmbfix)
        if i==int(maxiter/2) and not noise_fix: # fit also noise at some point
            Nt = noise.powspec()
            noise = smica.NoiseDiag(nmap, len(nmodes), name='noise')
            model = smica.Model([cmb, gal, noise])
            if noise_template is not None:
                noise.set_powspec(Nt, fixed=noise_template)
            else:
                noise.set_powspec(Nt)
            cmb.set_powspec (cmbcq)
        model.close_form(stats)

        # compute new mismatch 
        mm2 = model.mismatch(stats, nmodes, exact=True)
        mm2G = model.mismatch(stats, nmodes)
        gain = np.real(mmG-mm2G)
        if gain==0 and i>maxiter/2.0:
            break
        strtoprint = "iter= % 4i mismatch = %10.5f  gain= %7.5f " % (i, np.real(mm2), gain)
        if logger is not None:
            logger.info(strtoprint)
        else:
            print(strtoprint)
        mm = mm2
        mmG = mm2G

    cmb.fix_powspec(cfix)
    gal.fix_powspec("null")
    return model


def cmbmaps2C_lS():
    C_lS = dict()
    cmb_map = dict()
    for det in detectors:
        if int(det)<100:
            nside_desc = cf['pa']['nside_desc_map'][0]
        else:
            nside_desc = cf['pa']['nside_desc_map'][1]
        hdul = fits.open("/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20_sim/0200/input/ffp10_cmb_{det}_alm_mc_0200_nside{nside}_quickpol.fits".format(det=det, nside=nside_desc))
        cmb_map[det] = np.array([hp.ud_grade(hdul[1].data.field(spec).reshape(-1), nside_out = nside_out, order_in = 'NESTED', order_out='RING')
                            for spec in [0,1,2]])
    # C_lS = pw.tqupowerspec(cmb_map, tmask, pmask, lmax, lmax_mask)

    for det in detectors:
        alms = pw.map2alm_spin(cmb_map[det], pmask[det], 2, lmax) # full sky QU->EB
        # almT[det] = alms[0]
        almE[det] = alms[0]
        almB[det] = alms[1]

    CMB_in = dict()
    nalm = int((lmax+1)*(lmax+2)/2)
    signalW = np.ones(shape=(750,7))*1/len(detectors)
    beamf = io.load_beamf(csu.freqcomb)
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    for m,det in zip(range(len(detectors)),detectors):
        # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
        combalmE += hp.almxfl(hp.almxfl(almE[det],1/beamf[str(det)+'-'+str(det)]["HFI"][1].data.field(1)[0:lmax]), np.squeeze(signalW[:,m]))
        combalmB += hp.almxfl(hp.almxfl(almB[det],1/beamf[str(det)+'-'+str(det)]["HFI"][1].data.field(2)[0:lmax]), np.squeeze(signalW[:,m]))

    CMB_in["EE"] = hp.almxfl(combalmE, 1/hp.pixwin(nside_out, pol=True)[0][0:lmax])
    CMB_in["BB"] = hp.almxfl(combalmB, 1/hp.pixwin(nside_out, pol=True)[1][0:lmax])

    CMB_in["TQU"] = dict()
    CMB_in["TQU"] = hp.alm2map([np.zeros_like(CMB_in["EE"]), CMB_in["EE"], CMB_in["BB"]], nside_out)
    return CMB_in["TQU"]


def cov_lS2cov_lSmin(icov_lS, C_lS):
    reC_lS = hpf.reorder_spectrum_dict(C_lS)
    C_lSmin = dict()
    for specc, data in reC_lS.items():
        icovsum = np.array([np.sum(icov_lS[specc][l]) for l in range(lmax)])
        C_lSmin[specc] = np.array([1/icovsum_l if icovsum_l is not None else 0 for icovsum_l in icovsum])
    return C_lSmin  


if __name__ == '__main__':
    # set_logger(DEBUG)
    run_weight = True
    run_cmbmap = False
    run_smica = False
    run_propag = False
    run_tf = False

    CMB = dict()
    CMB["TQU"] = dict()
    almT, almE, almB = dict(), dict(), dict()

    filename_raw = io.total_filename_raw
    filename = io.total_filename
    ndet = len(detectors)
    bins =  const.SMICA_lowell_bins    #const.linear_equisized_bins_10 #const.linear_equisized_bins_1
    maxiter = 50
    freqdset = cf['pa']['freqdset']
    sim_id = cf[mch][freqdset]['sim_id']

    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated filename(s) for this session: {}".format(filename_raw))
    print(filename)
    print(40*"$")


    if run_fit:
        """
        Runs SMICA using,
            * signal estimator: C_lS_in.py 
            * noise estimator: io.noise_unsc_path_name
            * empiric data to be fitted: freqdset attribute from config

        Result is,
            * signal estimator: C_lS_out.py
            * channel weights: io.weight_path + "SMICAWEIG_" + cf['pa']["Tscale"] + "_" + io.total_filename
        """
        C_ltot = load_powerspectra('full')
        cov_ltot = pw.build_covmatrices(C_ltot, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
        cov_ltotEE = cov_ltot["EE"]
        print(cov_ltotEE.shape)

        C_lN = load_powerspectra('noise')
        cov_lN = pw.build_covmatrices(C_lN, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
        cov_lNEE = cov_lN["EE"]
        print(cov_lNEE.shape)

        ##### new
        C_lS = load_powerspectra('signal')
        cov_lS = pw.build_covmatrices(C_lS, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
        cov_lSEE = cov_lS["EE"]
        print(cov_lSEE.shape)
        #####

        cov_ltot_bnd = hpf.bin_it(cov_ltotEE, bins=bins)
        print(cov_ltot_bnd.shape)

        cov_lN_bnd = hpf.bin_it(cov_lNEE, bins=bins)
        cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T
        print(cov_lN_bnd.shape)

        ##### old
        C_lS_EE = io.load_data("/global/cscratch1/sd/sebibel/misc/C_lS_in.npy")[1]
        cov_lS_EE = np.ones((ndet,ndet,lmax+1)) * C_lS_EE
        
        C_lS_bnd =  hpf.bin_it(cov_lS_EE, bins=bins)
        print(C_lS_bnd.shape)
        #####

        nmodes = calc_nmodes(bins, pmask['100']) #any mask will do, only fsky needed
        smica_model = build_smica_model(len(nmodes), cov_lN_bnd, C_lS_bnd)

        fit_model_to_cov(
            smica_model,
            np.abs(cov_ltot_bnd),
            nmodes,
            maxiter=maxiter,
            noise_fix=True,
            noise_template=cov_lN_bnd,
            afix=None, qmin=0,
            asyn=None,
            logger=None,
            qmax=len(nmodes),
            no_starting_point=False)

        cmb_specsmica_sc_path_name = io.out_specsmica_path + "CMB_" + io.specsmica_sc_filename
        io.save_data(smica_model.get_comp_by_name('cmb').powspec(), cmb_specsmica_sc_path_name)

        io.save_data(smica_model.get_comp_by_name('cmb').powspec(), "/global/cscratch1/sd/sebibel/smica/C_lS_out.npy")
        io.save_data(smica_model.get_theta(), "/global/cscratch1/sd/sebibel/smica/theta.npy")
        io.save_data(smica_model.covariance4D(), "/global/cscratch1/sd/sebibel/smica/cov4D.npy")
        io.save_data(smica_model.covariance(), "/global/cscratch1/sd/sebibel/smica/cov.npy")

        #TODO smica needs to run for both BB and EE, as BB-weights are needed for later map generation
        smica_cov_full = dict()
        zer = np.zeros_like(smica_model.covariance())
        for spec in csu.PLANCKSPECTRUM_f:
            smica_cov_full[spec] = zer
        smica_cov_full["EE"] = smica_model.covariance()
        smica_cov_full_inv_ltot = pw.invert_covmatrices(smica_cov_full, len(bins))
        smica_weights_tot = pw.calculate_weights(smica_cov_full_inv_ltot, len(bins), "K_CMB")
        print(smica_weights_tot.shape)
        io.save_data(smica_weights_tot, io.weight_path + "SMICAWEIG_" + cf['pa']["Tscale"] + "_" + io.total_filename)


    if run_propag:
        """
        Follows the SMICA propagation code to combine maps with set of weights.
        """
        W = io.load_data(io.weight_path + "SMICAWEIG_" + cf['pa']["Tscale"] + "_" + io.total_filename)

        # full maps
        maps = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
        maps = prep.preprocess_all(maps)

        for det in detectors:
            print(det)
            alms = pw.map2alm_spin(maps[det], pmask[det], 2, lmax) # full sky QU->EB
            # almT[det] = alms[0]
            almE[det] = alms[0]
            almB[det] = alms[1]

        nalm = int((lmax+1)*(lmax+2)/2)  
        # combalmT = np.zeros((nalm), dtype=np.complex128)
        combalmE = np.zeros((nalm), dtype=np.complex128)
        combalmB = np.zeros((nalm), dtype=np.complex128)
        beamf = io.load_beamf(freqcomb=csu.freqcomb)

        xnew = np.arange(0,lmax,1)
        for it, det in enumerate(csu.PLANCKMAPFREQ): #weights do not dependent on freqfilter, but almE/B do
            if det in csu.PLANCKMAPFREQ_f:
                ns = nside_out[0] if int(det) < 100 else nside_out[1]
                W_Einterp = interpolate.interp1d(np.mean(bins, axis=1), W[1,:,it], bounds_error = False, fill_value='extrapolate')
                #TODO switch to W[2,:] once BB-weights are correctly calculated
                W_Binterp = interpolate.interp1d(np.mean(bins, axis=1), W[1,:,it], bounds_error = False, fill_value='extrapolate')

                # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
                combalmE += hp.almxfl(hp.almxfl(almE[det],1/beamf[str(det)+'-'+str(det)]["HFI"][1].data.field(1)[0:lmax]), np.squeeze(W_Einterp(xnew)))
                combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][0:lmax])
                combalmB += hp.almxfl(hp.almxfl(almB[det],1/beamf[str(det)+'-'+str(det)]["HFI"][1].data.field(2)[0:lmax]), np.squeeze(W_Binterp(xnew)))
                combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][0:lmax])

        CMB["TQU"]['out'] = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], nside_out[1])
        io.save_data(CMB["TQU"]['out'], "/global/cscratch1/sd/sebibel/misc/smicaminvarmap.npy")
        smica_C_lmin_unsc = ps.map2cl_spin(qumap=CMB["TQU"]['out'][1:3], spin=2, mask=pmask['100'], lmax=lmax,
            lmax_mask=lmax_mask)
        io.save_data(smica_C_lmin_unsc, "/global/cscratch1/sd/sebibel/misc/smicaclmin.npy")