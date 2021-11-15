"""
run_powerspectrum.py: script for generating powerspectra from maps.
Depends on all maps being generated to begin with. Use ``run_map.py``, if e.g. noisemaps are missing.
"""

__author__ = "S. Belkner"

import os, sys

import healpy as hp
import numpy as np

from component_separation.cs_util import Config
from component_separation.cs_util import Filename_gen as fn_gen
from component_separation.io import IO

import component_separation.transformer as trsf
import component_separation.powerspectrum as pospec
import component_separation.map as mp

experiment = 'Planck'
csu = Config(experiment=experiment)
fn = fn_gen(csu, experiment=experiment)
io = IO(csu)


def run_map2cls(info_component):

    outpath_pow_sc_name = fn.get_spectrum(info_component)
    print("outpath_pow_sc_name: {}".format(outpath_pow_sc_name))

    maps = dict()
    for FREQ in csu.FREQ:
        if FREQ not in csu.FREQFILTER:
            inpath_map_pla_name = fn.get_d(FREQ, info_component)
            print("inpath_map_pla_name: {}".format(inpath_map_pla_name))
            maps[FREQ] = io.load_pla(inpath_map_pla_name, field=(0,1,2), ud_grade=(True, FREQ))
    maps = mp.process_all(maps)

    apo = csu.spectrum_type == 'pseudo'
    tmask_fn = fn.get_mask('T', apodized=apo)
    pmask_fn = fn.get_mask('P', apodized=apo)
    tmask_sg = io.load_mask(tmask_fn, stack=True)
    pmask_sg = io.load_mask(pmask_fn, stack=True)

    tmask, pmask = dict(), dict()
    for FREQ in csu.PLANCKMAPFREQ:
        if FREQ not in csu.freqfilter:
            tmask[FREQ] = tmask_sg
            pmask[FREQ] = pmask_sg

    Cl_usc = trsf.map2cls(maps, tmask, pmask, csu.spectrum_type, csu.lmax, csu.freqcomb, csu.nside_out, csu.lmax_mask)

    beamf_dict = fn.get_beamf()
    beamf = io.load_beamf(beamf_dict, csu.freqcomb)
    Cl_sc = pospec.process_all(Cl_usc, csu.freqcomb, beamf, csu.nside_out)
    io.save_data(Cl_sc, outpath_pow_sc_name)


def run_alm2cls(path_name):

    cmb_tlm, cmb_elm, cmb_blm = io.load_alms('cmb', 0)
    buffer = hp.alm2cl([cmb_tlm, cmb_elm, cmb_blm])[:,:csu.lmax+1]

    #TODO this 6 should depend on the number of speccomb, but something seems wrong in the config
    ClS_usc = np.zeros(shape=(len(csu.freqcomb),6,csu.lmax+1))
    for idfreqc in range(ClS_usc.shape[0]):
        ClS_usc[idfreqc] = buffer
    io.save_data(ClS_usc, path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)

    bool_alm2cls = False
    bool_map2cls = True
    bool_with_noise = True

    if bool_map2cls:
        run_map2cls('T')
        run_map2cls('N')

    if bool_alm2cls:
        run_alm2cls("/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy")