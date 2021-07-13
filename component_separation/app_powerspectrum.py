"""
run_powerspectrum.py: script for generating powerspectra from maps.
Depends on all maps being generated to begin with. Use ``run_map.py``, if e.g. noisemaps are missing.

"""

__author__ = "S. Belkner"

import copy

import os
import sys
from typing import Dict, List, Optional, Tuple
import component_separation.interface as cslib

import healpy as hp
import numpy as np

import component_separation
import component_separation.io as io
import component_separation.powspec as pw
import component_separation.transform_map as trsf_m
import component_separation.transform_spec as trsf_s

from component_separation.cs_util import Config
from component_separation.cs_util import Helperfunctions as hpf

csu = Config()


def run_map2spec(bool_with_noise)
    @io.alert_cached(path_unsc)
    @io.alert_cached(path_sc)
    def r_m2s(cs, path_unsc, path_sc):        
        beamf = io.load_beamf(cs.freqcomb)
        cf_loc = cs.cf
        maps = io.load_plamap(cf_loc, field=(0,1,2), nside_out=cs.nside_out)
        maps = trsf_m.process_all(maps)
        tmask, pmask, pmask = io.load_one_mask_forallfreq()
        C_l_unsc = pw.tqupowerspec(data, tmask, pmask, cf['pa']["lmax"], cf['pa']["lmax_mask"])
        io.save_data(C_l_unsc, path_unsc)

        C_l = trsf_s.process_all(C_l_unsc, cf_loc, cs.freqcomb, cs.PLANCKSPECTRUM, cf_loc['pa']['Tscale'], beamf, cs.nside_out, cf_loc['pa']["Spectrum_scale"], cf_loc['pa']['smoothing_window'], cf_loc['pa']['max_polynom'])
        io.save_data(C_l, path_sc)

    if bool_map2spec_alsowithnoise:
        cf_n = copy.deepcopy(cf)
        cf_n['pa']['freqdset'] = cf['pa']['freqdset']+"_diff"
        csu_n = Config(cf_n)
        cslist = [csu, csu_n]
        path_name_list = [
            (io.spec_unsc_path_name, io.spec_sc_path_name),
            (io.noise_unsc_path_name, io.noise_sc_path_name)
        ]
    else:
        cslist = [csu]
        path_name_list = [(io.spec_unsc_path_name, io.spec_sc_path_name)]

    for cs, (path_unsc, path_sc) in zip(cslist, path_name_list):
        r_m2s(cs, path_unsc, path_sc)


@io.alert_cached(io.signal_sc_path_name)
def run_alm2spec()
    #TODO to process spectrum, need to know the beamfunction? is it 5arcmin?
    # beamf = {'12345-12345': hp.gauss_beam()}
    # C_lS_unsc = trsf_s.apply_beamf(C_lS_unsc, cf, ['12345-12345'], speccomb, beamf)
    io.alert_cached(io.signal_sc_path_name)
    cmb_tlm, cmb_elm, cmb_blm = cslib.load_alms('cmb', csu.sim_id)
    C_lS_unsc = np.array([hp.alm2cl([cmb_tlm, cmb_elm, cmb_blm])[:,:cf['pa']['lmax']+1]])
    C_lS = trsf_s.apply_scale(C_lS_unsc, cf['pa']["Spectrum_scale"]) 
    io.save_data(C_lS, io.signal_sc_path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_map2spec = True
    bool_alm2spec = False
    bool_with_noise = False

    if bool_map2spec:
        run_map2spec(bool_with_noise)

    if bool_alm2spec:
        run_alm2spec()