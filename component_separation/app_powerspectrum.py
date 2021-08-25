"""
run_powerspectrum.py: script for generating powerspectra from maps.
Depends on all maps being generated to begin with. Use ``run_map.py``, if e.g. noisemaps are missing.

"""

__author__ = "S. Belkner"

import copy
import os
import sys

import healpy as hp
import numpy as np

import component_separation.interface as cslib
import component_separation.io as csio
import component_separation.powspec as pw
import component_separation.transform_map as trsf_m
import component_separation.transform_spec as trsf_s
from component_separation.cs_util import Config
from component_separation.cs_util import Helperfunctions as hpf
from component_separation.io import IO

csu = Config()
io = IO(csu)

#TODO this structure doesnt serve the need anymore. simplify
def run_map2spec(bool_with_noise):
    @csio.alert_cached
    def r_m2s(path_unsc, overw, cs):        
        cf_loc = cs.cf
        maps = io.load_plamap(cf_loc, field=(0,1,2), nside_out=cs.nside_out)
        maps = trsf_m.process_all(maps)
        tmask, pmask, pmask = io.load_one_mask_forallfreq()
        C_l_unsc = pw.tqupowerspec(maps, tmask, pmask, cf_loc['pa']["lmax"], cf_loc['pa']["lmax_mask"], cs.nside_out, cs.freqcomb)
        io.save_data(C_l_unsc, path_unsc)
        return C_l_unsc

    @csio.alert_cached
    def r_s2sc(path_sc, overw, cs, C_l_unsc):
        beamf = io.load_beamf(cs.freqcomb)
        C_l = trsf_s.process_all(C_l_unsc, cs.freqcomb, beamf, cs.nside_out, cs.cf['pa']["Spectrum_scale"], cs.cf['pa']['smoothing_window'], cs.cf['pa']['max_polynom'])
        io.save_data(C_l, path_sc)

    if bool_with_noise:
        cf_n = copy.deepcopy(csu.cf)
        cf_n['pa']['freqdset'] = csu.cf['pa']['freqdset']+"_diff"
        csu_n = Config(cf_n)
        cslist = [csu, csu_n]
        path_name_list = [
            (io.fh.spec_unsc_path_name, io.fh.spec_sc_path_name),
            (io.fh.noise_unsc_path_name, io.fh.noise_sc_path_name)
        ]
    else:
        cslist = [csu]
        path_name_list = [(io.fh.spec_unsc_path_name, io.fh.spec_sc_path_name)]
    for cs, (path_unsc, path_sc) in zip(cslist, path_name_list):
        C_l_unsc = r_m2s(path_unsc, csu.overwrite_cache, cs)
        r_s2sc(path_sc, csu.overwrite_cache, cs, C_l_unsc)


@csio.alert_cached
def run_alm2spec(path_name, overw):
    #TODO to process spectrum, need to know the beamfunction? is it 5arcmin?
    # beamf = {'12345-12345': hp.gauss_beam()}
    # C_lS_unsc = trsf_s.apply_beamf(C_lS_unsc, cf, ['12345-12345'], speccomb, beamf)
    # io.alert_cached(io.signal_sc_path_name)
    cmb_tlm, cmb_elm, cmb_blm = cslib.load_alms('cmb', csu.sim_id)
    C_lS_unsc = np.array([hp.alm2cl([cmb_tlm, cmb_elm, cmb_blm])[:,:csu.cf['pa']['lmax']+1]])
    C_lS = trsf_s.apply_scale(C_lS_unsc, csu.cf['pa']["Spectrum_scale"])
    io.save_data(C_lS, path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_map2spec = True
    bool_alm2spec = False
    bool_with_noise = False

    if bool_map2spec:
        run_map2spec(bool_with_noise)

    if bool_alm2spec:
        run_alm2spec(io.fh.signal_sc_path_name, csu.overwrite_cache)

