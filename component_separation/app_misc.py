#!/usr/local/bin/python
"""
app_misc.py: script for various functions
"""
#TODO why is NPIPE_sim CMB map uncorrelated with NPIPE MV map? Both contain signal

__author__ = "S. Belkner"

import os

os.environ["OMP_NUM_THREADS"] = "8"

import healpy as hp
import numpy as np
import smica

from component_separation.io import IO
import component_separation.io as csio
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Config
import component_separation.transform_spec as trsf_s

csu = Config()
io = IO(csu)

filename = io.fh.total_filename

#TODO perhaps exclude LFIs for ell>1000, i.e. remove from covmatrix, instead of smoothing them..
@csio.alert_cached
def run_weight(path_name, overw):
    """
    Calculates weights derived from data defined by freqdset attribute of config.
    No SMICA, straightforward weight derivation.
    Needed for combining maps without SMICA.
    """
    C_ltot = io.load_powerspectra('full')
    cov = pw.build_covmatrices(C_ltot, "K_CMB", csu.freqcomb, csu.PLANCKMAPFREQ_f)
    cov = pw.cov2cov_smooth(cov, cutoff=800)
    weights_tot = np.zeros(shape=(2,7,csu.lmax))
    weights_tot = pw.cov2weight(cov, Tscale=csu.Tscale)
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
