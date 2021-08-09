#!/usr/local/bin/python
"""
interface.py: in the future, this could be somewhat serve as connecting application level with backend

"""

__author__ = "S. Belkner"


import json
import os
import os.path
import platform
from logging import CRITICAL, DEBUG, ERROR, INFO
from os import path
from typing import Dict, List, Optional, Tuple
import sys

import healpy as hp
import numpy as np

import component_separation
from component_separation.io import IO
import smica
import component_separation.transform_map as trsf
from component_separation.cs_util import Config
csu = Config()
io = IO(csu)

def build_smica_model(Q, N_cov_bn, C_lS_bnd, gal_mixmat=None, B_fit=False):
    ### Noise part
    nmap = N_cov_bn.shape[0]
    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_ampl(np.ones((nmap,1)), fixed="all")
    print(Q, N_cov_bn.shape)
    noise.set_powspec(np.nan_to_num(N_cov_bn), fixed="all") # where N is a (nmap, Q) array with noise spectra

    ### CMB part
    cmb = smica.Source1D(nmap, Q, name='cmb')
    acmb = np.ones((nmap,1)) # if cov in cmb unit
    cmb.set_mixmat(acmb, fixed='null')
    cmbcq = C_lS_bnd[0,0,:]
    if B_fit:
        cmb.set_powspec(cmbcq*0, fixed='all') # B modes fit
    else:
        cmb.set_powspec(cmbcq) # where cmbcq is a starting point for cmbcq like binned lcdm

    ### Galactic foreground part
    dim = 6
    if gal_mixmat is None:
        gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6
        gal.fix_mixmat("null")
    else:
        print(gal_mixmat.shape)
        gal = smica.SourceND(nmap, Q, dim, name='gal') # dim=6
        gal.set_mixmat(gal_mixmat, fixed='all')

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
    hist = np.array([[np.nan, np.nan] for n in range(maxiter)])
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
        hist[i,0] = np.real(mm2)
        hist[i,1] = gain
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
    return model, hist


def load_alms(component, id):
    if component == 'cmb':
        cmb_tlm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(id), hdu=1)
        cmb_elm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(id), hdu=2)
        cmb_blm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(id), hdu=3)
        return cmb_tlm, cmb_elm, cmb_blm
    else:
        print('unclear request of loading alms. Exiting..')
        sys.exit()