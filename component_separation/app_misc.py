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
from component_separation.cs_util import Filename_gen_SMICA as fns_gen

import component_separation.MSC.MSC.pospace as ps #remove dependency
import component_separation.powerspectrum as pospec
import component_separation.covariance as cv
import component_separation.map as mp
import component_separation.transformer as trsf

experiment = 'Pico'
simid=0
# csu = Config(experiment=experiment, freqdset='NPIPE', mskset='lens', spectrum_type='JC')
csu = Config(experiment=experiment)
fn = fn_gen(csu)
fns = fns_gen(csu)
io = IO(csu)

cutoff = 4000
lmaxp1 = csu.lmax+1
apo = csu.spectrum_type == 'pseudo'

tmask_fn = fn.get_mask('T', apodized=apo)
pmask_fn = fn.get_mask('P', apodized=apo)
tmask_sg = io.load_mask(tmask_fn)
pmask_sg = io.load_mask(pmask_fn)
tmask, pmask = dict(), dict()
for FREQ in csu.FREQ:
    if FREQ not in csu.FREQFILTER:
        tmask[FREQ] = tmask_sg
        pmask[FREQ] = pmask_sg


#TODO perhaps exclude LFIs for ell>1000, i.e. remove from covmatrix, instead of smoothing them..
#TODO cov2weight seems bugged, only works when np.nan_to_num(data) is passed. but nan's is what cov2weight wants for filtering..
def run_weight(path_name):
    """
    Calculates weights derived from data defined by freqdset attribute of config.
    No SMICA, straightforward weight derivation.
    Needed for combining maps without SMICA.
    """
    Cl_tot = io.load_data(fn.get_spectrum("T", "non-separated", simid=simid))

    covl_tot = cv.build_covmatrices(Cl_tot, "K_CMB", csu.freqcomb, csu.FREQ_f, csu.cutoff_freq, cutoff=1400)
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


def run_propag_mv():
    lmax_loc = 2000
    W_mv = io.load_data(fn.get_misc('w', simid=simid))
    W_total = W_mv

    nalm = int((lmax_loc+1)*(lmax_loc+2)/2) 
    alm = np.zeros(shape=(len(csu.FREQ_f),3,nalm))

    beamf = io.load_beamf(csu.freqcomb, csu.lmax, csu.freqdatsplit)
    
    maps = dict()
    for FREQ in csu.FREQ:
        if FREQ not in csu.FREQFILTER:
            inpath_map_pla_name = fn.get_d(FREQ, "T", simid=simid)
            print("inpath_map_pla_name: {}".format(inpath_map_pla_name))
            nside_out = csu.nside_out[0] if int(FREQ)<100 else csu.nside_out[1]
            maps[FREQ] = io.load_d(inpath_map_pla_name, field=(0,1,2), nside_out=nside_out)

    maps = mp.process_all(maps)

    for itf, freq in enumerate(csu.FREQ_f):
        print('freq: ', freq)
        ns = csu.nside_out[0] if int(freq) < 100 else csu.nside_out[1]
        if apo:
            alm[itf][1:] = hp.map2alm(np.array([n*hp.ud_grade(pmask[freq], nside_out=ns) for n in maps[freq]]), lmax_loc)[1:] # full sky QU->EB        #TODO no TT at the moment
        else:
            alm[itf][1:] = trsf.map2alm_spin(maps[freq], hp.ud_grade(pmask[freq], nside_out=ns), 2, lmax_loc) # full sky QU->EB        #TODO no TT at the moment
 
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    beam_e = hp.gauss_beam(np.radians(0.005/60), 4100, pol = True)[:,1]
    beam_b = hp.gauss_beam(np.radians(0.005/60), 4100, pol = True)[:,2]

    for itf, det in enumerate(csu.FREQ_f):
        print('freq: ', det)
        ns = csu.nside_out[0] if int(det) < 100 else csu.nside_out[1]
        # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
        combalmE += hp.almxfl(
            hp.almxfl(
                hp.almxfl(
                    alm[itf][1], np.nan_to_num(1/beamf[1,itf,itf,:lmax_loc])),
                beam_e[:lmax_loc]),
            np.squeeze(np.nan_to_num(W_total[1,itf,:lmax_loc])))
        combalmE = hp.almxfl(combalmE, 1/hp.pixwin(ns, pol=True)[0][:lmax_loc])
        combalmB += hp.almxfl(
            hp.almxfl(
                hp.almxfl(
                    alm[itf][2], np.nan_to_num(1/beamf[2,itf,itf,:lmax_loc])),
                    beam_b[:lmax_loc]),
            np.squeeze(np.nan_to_num(W_total[2,itf,:lmax_loc])))
        combalmB = hp.almxfl(combalmB, 1/hp.pixwin(ns, pol=True)[1][:lmax_loc])

    mapT_combined = hp.alm2map([np.zeros_like(combalmE), combalmE, combalmB], csu.nside_out[1])
    io.save_data(mapT_combined, fns.get_map('T', 'combined', simid=simid))
    ClT_combined = trsf.map2cls({'combined':mapT_combined}, {'combined':tmask[csu.FREQ_f[0]]}, {'combined':pmask[csu.FREQ_f[0]]}, csu.spectrum_type, lmax_loc, freqcomb=['combined-combined'], lmax_mask=csu.lmax_mask)
    io.save_data(ClT_combined, fns.get_spectrum('T', 'combined', simid=simid))

    maq_lpDXS = hp.smoothing(hp.ma(mapT_combined[1]), np.radians(1))
    mau_lpDXS = hp.smoothing(hp.ma(mapT_combined[2]), np.radians(1))

    mapT_combined_fn = fns.get_map('T', 'combined', simid=simid)
    mapT_combined_smoothed_fn = mapT_combined_fn.replace('.', 'smoothed.')

    io.save_data(np.array([np.zeros_like(maq_lpDXS),maq_lpDXS, mau_lpDXS]), mapT_combined_smoothed_fn)


if __name__ == '__main__':
    # set_logger(DEBUG)
    bool_weight = True
    bool_propagmv = True
    bool_crosscov = False
    bool_tf = False

    if bool_weight:
        run_weight(fn.get_misc('w', simid=simid))

    if bool_propagmv:
        run_propag_mv()

    if bool_crosscov:
        run_mappsmica2crosscov()

    if bool_tf:
        run_tf(io.fh.out_misc_path+"tf_{}".format(csu.binname) + "_" + filename)
