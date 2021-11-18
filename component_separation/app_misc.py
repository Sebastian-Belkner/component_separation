#!/usr/local/bin/python
"""
app_misc.py: script for various functions
"""

__author__ = "S. Belkner"

import os, sys

os.environ["OMP_NUM_THREADS"] = "16"

import healpy as hp
import numpy as np

from component_separation.cs_util import Config
from component_separation.io import IO
from component_separation.cs_util import Filename_gen as fn_gen

import component_separation.MSC.MSC.pospace as ps #remove dependency
import component_separation.powerspectrum as pospec
import component_separation.covariance as cv

csu = Config(experiment='Pico')
fn = fn_gen(csu)
io = IO(csu)

cutoff = 2000
lmaxp1 = csu.lmax+1


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

#TODO perhaps exclude LFIs for ell>1000, i.e. remove from covmatrix, instead of smoothing them..
#TODO cov2weight seems bugged, only works when np.nan_to_num(data) is passed. but nan's is what cov2weight wants for filtering..
def run_weight(path_name):
    """
    Calculates weights derived from data defined by freqdset attribute of config.
    No SMICA, straightforward weight derivation.
    Needed for combining maps without SMICA.
    """
    Cl_tot = io.load_data(fn.get_spectrum("T", "non-separated", simid=csu.simid))

    covl_tot = cv.build_covmatrices(Cl_tot, "K_CMB", csu.freqcomb, csu.FREQ_f, pico_cutoff, cutoff=1400)
    # covl_tot = pospec.cov2cov_smooth(covl_tot, cutoff=1000)
    weights_tot = cv.cov2weight(covl_tot, freqs=csu.FREQ_f, Tscale="K_CMB")
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


def run_tf(path_name):

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
        run_weight(fn.get_misc('w', simid=csu.simid))

    if bool_crosscov:
        run_mappsmica2crosscov()

    if bool_tf:
        run_tf(io.fh.out_misc_path+"tf_{}".format(csu.binname) + "_" + filename)
