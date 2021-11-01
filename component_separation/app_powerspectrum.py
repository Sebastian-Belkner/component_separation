"""
run_powerspectrum.py: script for generating powerspectra from maps.
Depends on all maps being generated to begin with. Use ``run_map.py``, if e.g. noisemaps are missing.
"""

__author__ = "S. Belkner"

import os, sys
import copy

import healpy as hp
import numpy as np

import component_separation.cachechecker as cc

import component_separation.powerspectrum as pospec
import component_separation.map as mp

from component_separation.cs_util import Config
from component_separation.io import IO
import component_separation.transformer as trsf
from component_separation.cs_util import Filename_gen as fn_gen

csu = Config()
fn = fn_gen(csu)
io = IO(csu)


def run_map2cls(info_component):

    outpath_pow_sc_name = fn.get_spectrum(info_component)
    print("outpath_pow_sc_name: {}".format(outpath_pow_sc_name))

    maps = dict()
    for FREQ in csu.PLANCKMAPFREQ:
        if FREQ not in csu.freqfilter:
            inpath_map_pla_name = fn.get_pla(FREQ, info_component)
            print("inpath_map_pla_name: {}".format(inpath_map_pla_name))
            maps[FREQ] = io.load_pla(inpath_map_pla_name, field=(0,1,2), ud_grade=(True, FREQ))
    # maps = mp.process_all(maps)

    tmask_fn = fn.get_mask('T')
    pmask_fn = fn.get_mask('P')
    tmask_sg = io.load_mask(tmask_fn, stack=True)
    pmask_sg = io.load_mask(pmask_fn, stack=True)

    if csu.spectrum_type == 'pseudo':
        tmask_sg = mp.apodize_mask(tmask_sg)
        pmask_sg = mp.apodize_mask(pmask_sg)

    tmask, pmask = dict(), dict()
    for FREQ in csu.PLANCKMAPFREQ:
        if FREQ not in csu.freqfilter:
            tmask[FREQ] = tmask_sg
            pmask[FREQ] = pmask_sg

    Cl_usc = trsf.map2cls(maps, tmask, pmask)
    
    beamf_dict = fn.get_beamf()
    beamf = io.load_beamf(beamf_dict, csu.freqcomb)
    Cl_sc = pospec.process_all(Cl_usc, csu.freqcomb, beamf, csu.nside_out, csu.cf['pa']["Spectrum_scale"])
    io.save_data(Cl_sc, outpath_pow_sc_name)


@cc.alert_cached
def run_alm2cls(path_name, overw):
    cmb_tlm, cmb_elm, cmb_blm = io.load_alms('cmb', csu.sim_id)
    ClS_usc = np.array([hp.alm2cl([cmb_tlm, cmb_elm, cmb_blm])[:,:csu.cf['pa']['lmax']+1]])
    ClS = pospec.apply_scale(ClS_usc, csu.cf['pa']["Spectrum_scale"])
    io.save_data(ClS, path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_alm2spec = False

    bool_map2cls = True
    
    bool_with_noise = False
   
    # bool_map2spec = False
    # bool_map2psspec = False
    # if bool_map2spec:
    #     run_map2cls(bool_with_noise)

    # if bool_map2psspec:
    #     run_map2pcls(bool_with_noise)

    if bool_map2cls:
        run_map2cls('total')
        if bool_with_noise:
            run_map2cls('noise')

    if bool_alm2spec:
        run_alm2cls(csu.overwrite_cache)