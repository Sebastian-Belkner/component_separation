#!/usr/local/bin/python
"""
interface.py: in the future, this could be somewhat serve as connecting application level with backend
"""

__author__ = "S. Belkner"


import os
import os.path

from logging import CRITICAL, DEBUG, ERROR, INFO
from os import path

import sys

import numpy as np

from component_separation.io import IO
import smica
from component_separation.cs_util import Config

csu = Config()
io = IO(csu)

def build_smica_model(Q, N_cov_bn, C_lS_bnd, gal_mixmat=None, B_fit=False):
    ### Noise part
    nmap = N_cov_bn.shape[0]
    noise = smica.NoiseAmpl(nmap, Q, name='noise')
    noise.set_ampl(np.ones((nmap,1)), fixed='all') #For DX12 this may be fixed
    noise.set_powspec(np.nan_to_num(N_cov_bn), fixed="all")

    ### CMB part
    cmb = smica.Source1D(nmap, Q, name='cmb')
    acmb = np.ones((nmap,1)) # if cov in cmb unit
    cmb.set_mixmat(acmb, fixed='all')
    cmbcq = C_lS_bnd[0,0,:]
    if B_fit:
        cmb.set_powspec(cmbcq, fixed='all')
    else:
        cmb.set_powspec(cmbcq) # where cmbcq is a starting point for cmbcq like binned lcdm

    ### Galactic foreground part
    dim = 3 #6 is highest possible
    gal = smica.SourceND(nmap, Q, dim, name='gal')
    if gal_mixmat is None:
        gal.fix_mixmat('null')
    else:
        gal.set_mixmat(gal_mixmat, fixed='all')
        gal.fix_powspec("null")

    model = smica.Model(complist=[cmb, gal, noise])
    return model


def fit_model_to_cov(model, stats, nmodes, maxiter=50, noise_template=None, afix=None, no_starting_point=False):
    """ Fit the model to empirical covariance.
    """
    def is_mixmat_fixed(model):
        return np.sum(model.get_comp_by_name("gal")._mixmat.get_mask())==0
    qmin=0
    cg_maxiter = 1
    cg_eps = 1e-20
    nmap = stats.shape[0]
    cmb,gal,noise = model._complist

    # find a starting point
    acmb = model.get_comp_by_name("cmb").mixmat()
    afix = 1-cmb._mixmat.get_mask() #0:free 1:fixed
    cfix = 1-cmb._powspec.get_mask() #0:free 1:fixed

    cmbcq = np.squeeze(model.get_comp_by_name("cmb").powspec())
    polar = True if acmb.shape[1]==2 else False
    print("starting point chosen.")

    if not is_mixmat_fixed(model) and not no_starting_point:
        model.ortho_subspace(stats, nmodes, acmb, qmin=qmin, qmax=len(nmodes))
    print("ortho_subspace done.")

    model.quasi_newton(stats, nmodes)
    print('quasi_newton done.')

    # cmb.set_powspec(cmbcq, fixed=cfix)
    # print('CMB powspec set 1/2')

    model.close_form(stats)
    print('close_form done.')

    # cmb.set_powspec(cmbcq, fixed=cfix)
    # print('CMB powspec set 2/2')

    mm_i = model.mismatch(stats, nmodes, exact=True)
    print('model.mismatch calculated: {}'.format(mm_i))

    # start CG/close form
    cmbfix = "null" if polar else cfix 
    hist = np.array([[np.nan, np.nan] for n in range(maxiter)])
    for i in range(maxiter):
        gal.fix_powspec("null")
        cmb.fix_powspec(cmbfix)

        model.conjugate_gradient(stats, nmodes,maxiter=cg_maxiter, avextol=cg_eps)

        # gal.fix_powspec("null")
        if mm_i!=model.mismatch(stats, nmodes, exact=True):
            cmbfix = cfix if polar else "all" 
            cmb.fix_powspec(cmbfix)
        if i==int(maxiter/2): # fit also noise at some point
            Nt = noise.powspec()
            # noise = smica.NoiseDiag(nmap, len(nmodes), name='noise')
            # noise = smica.NoiseAmpl(nmap, len(nmodes), name='noise')
            # noise.set_ampl(np.ones((nmap,1)), fixed='all') #For DX12 this may be fixed
            # model = smica.Model([cmb, gal, noise])
            noise.set_powspec(Nt, fixed=noise_template)
            # cmb.set_powspec(cmbcq)

        model.close_form(stats)

        # compute new mismatch 
        mm_f = model.mismatch(stats, nmodes, exact=True)
        gain = mm_i-mm_f
        if gain==0 and i>maxiter/2.0:
            break
        print("iter= %4i mismatch = %10.5f  gain= %7.5f " % (i, mm_f, gain))
        hist[i,0] = mm_f
        hist[i,1] = gain
        mm_i = mm_f

    return model, hist


# Only way to get B-fit to run is when
    # noise.set_powspec(np.nan_to_num(N_cov_bn), fixed="all")  fixed =all
    # cmb.set_mixmat(acmb, fixed='all') fixed=all
    #         # fit mixing matrix
        # gal.fix_powspec("null")

# model.quasi_newton():
#  STDEV = sqrt(diag(CRB) / nq) results in nans (CRB negative)
# model.conjugate_gradient()
#   initial mismatch often nan for B-fit


# For B-fit, if maxiter too high -> singular matrix after a while