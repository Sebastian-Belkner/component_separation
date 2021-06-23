#!/usr/local/bin/python
"""
run_powerspectrum.py: script for executing main functionality of component_separation


"""

__author__ = "S. Belkner"

import json
import logging
from astropy.io import fits
import logging.handlers
import os
import platform
import healpy as hp
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple
import component_separation.MSC.MSC.pospace as ps
import numpy as np
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

num_sim = cf['pa']["num_sim"]

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


# def set_logger(loglevel=logging.INFO):
#     logger.setLevel(loglevel)
#     logging.StreamHandler(sys.stdout)


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


def calculate_powerspectra(maps, tmask, pmask):
    ### Calculate powerspectra

    C_ltot_unsc = map2spec(maps, tmask, pmask)
    io.save_data(C_ltot_unsc, io.spec_unsc_path_name)
    C_ltot = postprocess_spectrum(C_ltot_unsc, csu.freqcomb, cf['pa']['smoothing_window'], cf['pa']['max_polynom'])
    return C_ltot


def load_powerspectra():
    C_ltot = io.load_data(path_name=io.spec_sc_path_name)
    if C_ltot is None:
        print("couldn't find scaled spectrum with given specifications at {}. Trying unscaled..".format(io.spec_sc_path_name))
        C_ltot_unsc = io.load_data(path_name=io.spec_unsc_path_name)
        if C_ltot_unsc is None:
            print("couldn't find unscaled spectrum with given specifications at {}. Exit..".format(io.spec_unsc_path_name))
            sys.exit()
        C_ltot = postprocess_spectrum(C_ltot_unsc, csu.freqcomb, cf['pa']['smoothing_window'], cf['pa']['max_polynom'])
        io.save_data(C_ltot, io.spec_sc_path_name)
    return C_ltot


def cmbmap2C_lS():
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
    C_lS = pw.tqupowerspec(cmb_map, tmask, pmask, lmax, lmax_mask)
    return C_lS


def cov_lS2voc_lSmin(icov_lS, C_lS):
    reC_lS = hpf.reorder_spectrum_dict(C_lS)
    C_lSmin = dict()
    for specc, data in reC_lS.items():
        icovsum = np.array([np.sum(icov_lS[specc][l]) for l in range(lmax)])
        C_lSmin[specc] = np.array([1/icovsum_l if icovsum_l is not None else 0 for icovsum_l in icovsum])
        

if __name__ == '__main__':
    CMB = dict()
    almT, almE, almB = dict(), dict(), dict()

    detectors = csu.PLANCKMAPFREQ_f
    filename_raw = io.total_filename_raw
    filename = io.total_filename
    ndet = len(detectors)
    bins = const.SMICA_lowell_bins    #const.linear_equisized_bins_10 #
    maxiter = 50

    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated filename(s) for this session: {}".format(filename_raw))
    print(filename)
    print(40*"$")

    # set_logger(DEBUG)

    nside_out = cf['pa']['nside_out']
    tmask, pmask, pmask = io.load_one_mask_forallfreq(nside_out=nside_out)
    if cf['pa']['new_spectrum']:
        maps = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
        maps = prep.preprocess_all(maps)
        C_ltot = calculate_powerspectra(maps, tmask, pmask)
    else:
        C_ltot = load_powerspectra()
    cov_ltot = pw.build_covmatrices(C_ltot, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    
    C_lN_unsc = io.load_data(io.noise_unsc_path_name)
    C_lN = postprocess_spectrum(C_lN_unsc, csu.freqcomb, cf['pa']['smoothing_window'], cf['pa']['max_polynom'])
    cov_lN = pw.build_covmatrices(C_lN, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)

    C_lS = cmbmap2C_lS()
    io.save_data(C_lS, io.cmb_unsc_path_name)
    cov_lS = pw.build_covmatrices(C_lS, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    icov_lS = pw.invert_covmatrices(cov_lS, lmax=lmax)
    cov_lS_min = cov_lS2voc_lSmin(icov_lS, C_lS)
    # D_lS = pd.read_csv(
    #     cf[mch]['powspec_truthfile'],
    #     header=0,
    #     sep='    ',
    #     index_col=0)
    # D_lS_EE = D_lS["Planck-"+"EE"].to_numpy()
    # C_lS_EE = D_lS_EE[:lmax+1]/hpf.llp1e12(np.array([range(lmax+1)]))

    # cov_lS_EE = np.ones((ndet,ndet,lmax+1)) * C_lS_EE * 1e12

    cov_ltotEE = cov_ltot["EE"][0:8,0:8,:]
    cov_lNEE = cov_lN["EE"][0:8,0:8,:]

    cov_ltot_bnd = hpf.bin_it(cov_ltotEE, bins=bins)
    print(cov_ltot_bnd.shape)

    cov_lN_bnd = hpf.bin_it(cov_lNEE, bins=bins)
    # cov_lN_bnd[cov_lN_bnd==0.0] = 0.01
    cov_lN_bnd = np.diagonal(cov_lN_bnd, axis1=0, axis2=1).T
    print(cov_lN_bnd.shape)

    cov_lS_EE = cov_lS['EE']
    C_lS_bnd =  hpf.bin_it(cov_lS_EE, bins=bins)
    print(C_lS_bnd.shape)


    cov_inv_ltot = pw.invert_covmatrices(cov_ltot, lmax)
    weights_tot = pw.calculate_weights(cov_inv_ltot, len(bins), "K_CMB")
    print(weights_tot.shape)
    io.save_data(weights_tot, io.weight_path_name)


    nmodes = calc_nmodes(bins, pmask)
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
    
    io.save_data(smica_model.get_theta(), "/global/cscratch1/sd/sebibel/smica/theta.npy")
    # covariance4D
    io.save_data(smica_model.covariance4D(), "/global/cscratch1/sd/sebibel/smica/cov4D.npy")
    # covariance
    io.save_data(smica_model.covariance(), "/global/cscratch1/sd/sebibel/smica/cov.npy")


    # TODO calculate weights from smica cov output
    # cov_inv_ltot = pw.invert_covmatrices(cov_ltot, lmax)
    # weights_tot = pw.calculate_weights(cov_inv_ltot, len(bins), "K_CMB")
    # print(weights_tot.shape)
    # io.save_data(weights_tot, io.weight_path_name)

    """
    The next lines follow tightly the algorithm of SMICA propagation code
    """



    # smica_W_pol = np.loadtxt("/global/homes/s/sebibel/data/weights/weights_EB_smica_R3.00.txt").reshape(2,7,4001)
    # smica_W_t = np.loadtxt("/global/homes/s/sebibel/data/weights/weights_T_smica_R3.00_Xfull.txt")

    # TODO load weights from smica cov output
    W = io.load_data(io.weight_path_name)
    maps = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
    maps = prep.preprocess_all(maps)
    for det in detectors[:-2]:
        lmax = min(3 * nside_out - 1, lmax + lmax_mask)

        alms = pw.map2alm_spin(hp.ud_grade(maps[det], nside_out=nside_out), lmax) # full sky QU->EB
        almT[det] = alms[0]
        almE[det] = alms[1]
        almB[det] = alms[2]

    nalm = int((lmax+1)*(lmax+2)/2)  
    combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    for m,name in zip(range(len(names[:-2])),names[:-2]):
        combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
        combalmE += hp.almxfl(almE[name], np.squeeze(W[1,m,:]))
        combalmB += hp.almxfl(almB[name], np.squeeze(W[2,m,:]))

    CMB["TT"] = hp.almxfl(combalmT, hp.pixwin(nside_out)[0:lmax])
    CMB["EE"] = hp.almxfl(combalmE, hp.pixwin(nside_out, pol=True)[0][0:lmax])
    CMB["BB"] = hp.almxfl(combalmB, hp.pixwin(nside_out, pol=True)[1][0:lmax])

    CMB["TQU"]['out'] = hp.alm2map([CMB["TT"], CMB["EE"], CMB["BB"]], nside_out)
    smica_C_lS_unsc = pw.map2spec(CMB["TQU"], tmask, pmask)
    # SMICA result right above

    CMB["TQU"]['in'] = hp.synfast(cov_lS_min, nside=nside_out)
    # cmb input as given to SMICA

    # crosscovariance between cmb input and what smica gives
    pw.tqupowerspec(CMB["TQU"]['in'], CMB["TQU"]['out'])
    crosscov = ps.map2cls(
                tqumap=CMB["TQU"]['in'],
                tmask=tmask,
                pmask=pmask,
                lmax=lmax,
                lmax_mask=lmax_mask,
                tqumap2=CMB["TQU"]['out'],
                tmask2=tmask,
                pmask2=pmask
                )


    # TODO this should give the transferfunction
    transferfunction = crosscov/cov_lS_min


    """
    Now, follow the procedure described in https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Astrophysical_component_separation
    1. fit all model parameters over clean fraction of sky at  100 <= ell <= 680, keep emission spectrum a
    """