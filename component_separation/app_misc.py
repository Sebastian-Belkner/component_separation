#!/usr/local/bin/python
"""
run_powerspectrum.py: script for executing main functionality of component_separation


"""

#TODO check single frequency transferfunctions

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

@csio.alert_cached(io.fh.weight_path_name)
def run_weight():
    """
    Calculates weights derived from data defined by freqdset attribute of config.
    No SMICA, straightforward weight derivation.
    Needed for combining maps without SMICA.
    """
    def specsc2weights(spectrum):
        print(spectrum.shape)
        cov = pw.build_covmatrices(spectrum, csu.cf['pa']['Tscale'])
        print(cov.shape)
        cov_inv_l = pw.invert_covmatrices(cov)
        print(cov_inv_l.shape)
        weights = pw.calculate_weights(cov_inv_l, csu.cf['pa']['Tscale'])
        return weights
    C_ltot = cslib.load_powerspectra('full')
    weights_tot = specsc2weights(C_ltot)
    print(weights_tot.shape)
    io.save_data(weights_tot, io.weight_path_name)


@csio.alert_cached(io.fh.out_misc_path+"tf_{}".format(csu.cf['pa']['binname']) + "_" + filename)
@csio.alert_cached(io.fh.out_misc_path+"crosscov_{}".format(csu.cf['pa']['binname']) + "_" + filename)
def run_tf():
    lmax = csu.cf['pa']["lmax"]
    lmax_mask = csu.cf['pa']["lmax_mask"]
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    CMB = dict()
    CMB["TQU"] = dict()
    C_lS_EE = io.load_data(io.fh.signal_sc_path_name)[0,1]
    CMB["TQU"]["in"] = io.load_data(io.fh.map_cmb_sc_path_name)
    MV = io.load_data(io.fh.cmbmap_smica_path_name)

    # any mask will do here
    crosscov = ps.map2cl_spin(
        qumap=CMB["TQU"]["in"][1:3],
        spin=2,
        mask=pmask['100'],
        lmax=lmax,
        lmax_mask=lmax_mask,
        qumap2=MV[1:3],
        mask2=pmask['100']
    )
    transferfunction = np.sqrt(crosscov[0][:lmax]/C_lS_EE[:lmax])
    io.save_data(transferfunction, io.fh.out_misc_path+"tf_{}".format(cf['pa']['binname']) + "_" + filename)
    io.save_data(crosscov, io.fh.out_misc_path+"crosscov_{}".format(cf['pa']['binname']) + "_" + filename)


if __name__ == '__main__':
    # set_logger(DEBUG)
    bool_weight = True
    bool_tf = False

    if bool_weight:
        run_weight()

    if bool_tf:
        run_tf()
