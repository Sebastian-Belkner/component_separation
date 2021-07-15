#!/usr/local/bin/python
"""
run_powerspectrum.py: script for executing main functionality of component_separation


"""

__author__ = "S. Belkner"

import os

os.environ["OMP_NUM_THREADS"] = "8"

import healpy as hp
import numpy as np
import smica

import component_separation
import component_separation.interface as cslib
from component_separation.io import IO
import component_separation.io as csio
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Config

csu = Config()
io = IO(csu)

filename = io.fh.total_filename

@csio.alert_cached
def run_weight(path_name, overw):
    """
    Calculates weights derived from data defined by freqdset attribute of config.
    No SMICA, straightforward weight derivation.
    Needed for combining maps without SMICA.
    """
    def specsc2weights(spectrum):
        print(spectrum.shape)
        cov = pw.build_covmatrices(spectrum, csu.Tscale, csu.freqcomb, csu.PLANCKMAPFREQ_f)
        print(cov.shape)
        cov_inv_l = pw.invert_covmatrices(cov)
        print(cov_inv_l.shape)
        weights = pw.calculate_weights(cov_inv_l, csu.PLANCKMAPFREQ[:-2], csu.Tscale)
        return weights
    C_ltot = io.load_powerspectra('full')
    weights_tot = specsc2weights(C_ltot)
    print(weights_tot.shape)
    io.save_data(weights_tot, path_name)


def run_mappsmica2crosscov():
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    lmaxbins = int(csu.bins[-1][1])
    lmaxbins_mask = 2*lmaxbins
    CMB = io.load_data(io.fh.map_cmb_sc_path_name)
    MV = io.load_data(io.fh.cmbmap_smica_path_name)

    # any mask will do here
    crosscov = np.array(ps.map2cl_spin(
        qumap=CMB[1:3],
        spin=2,
        mask=pmask['100'],
        lmax=lmaxbins,
        lmax_mask=lmaxbins_mask,
        qumap2=MV[1:3],
        mask2=pmask['100']
    ))*1e12
    io.save_data(crosscov, io.fh.out_misc_path+"crosscov_{}".format(csu.binname) + "_" + filename)


@csio.alert_cached
def run_tf(path_name, overwr):
    lmaxbins = int(csu.bins[-1][1])
    C_lS_EE = io.load_data(io.fh.signal_sc_path_name)[0,1]
    crosscov = io.load_data(io.fh.out_misc_path+"crosscov_{}".format(csu.binname) + "_" + filename)
    transferfunction = np.sqrt(crosscov[0][:lmaxbins]/C_lS_EE[:lmaxbins])
    io.save_data(transferfunction, io.fh.out_misc_path+"tf_{}".format(csu.binname) + "_" + filename)


if __name__ == '__main__':
    # set_logger(DEBUG)
    bool_weight = True
    bool_crosscov = False
    bool_tf = False

    if bool_weight:
        run_weight(io.fh.weight_path_name, csu.overwrite_cache)

    if bool_crosscov:
        run_mappsmica2crosscov()

    if bool_tf:
        run_tf(io.fh.out_misc_path+"tf_{}".format(csu.binname) + "_" + filename, csu.overwrite_cache)
