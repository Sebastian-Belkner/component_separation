"""
run_powerspectrum.py: script for generating powerspectra from maps.
Depends on all maps being generated to begin with. Use ``run_map.py``, if e.g. noisemaps are missing.
"""

__author__ = "S. Belkner"

import copy

import healpy as hp
import numpy as np

import component_separation.interface as cslib
import component_separation.cachechecker as cc

import component_separation.powerspectrum as pospec
import component_separation.map as mp

from component_separation.cs_util import Config
from component_separation.io import IO
import component_separation.transformer as trsf
from component_separation.cs_util import Filename_gen as fn_gen

csu = Config()
io = IO(csu)

#TODO this structure doesnt serve the need anymore. simplify
def run_map2cls(bool_with_noise):
    @cc.alert_cached
    def r_m2s(path_unsc, overw, cs):        
        cf_loc = cs.cf
        maps = io.load_plamap(cf_loc, field=(0,1,2), nside_out=cs.nside_out)
        maps = mp.process_all(maps)
        tmask, pmask, pmask = io.load_one_mask_forallfreq()
        C_l_unsc = trsf.map2cls(maps, 'Chonetal', tmask, pmask)
        io.save_data(C_l_unsc, path_unsc)
        return C_l_unsc

    @cc.alert_cached
    def r_s2sc(path_sc, overw, cs, C_l_unsc):
        beamf = io.load_beamf(cs.freqcomb)
        C_l = pospec.process_all(C_l_unsc, cs.freqcomb, beamf, cs.nside_out, cs.cf['pa']["Spectrum_scale"])
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


#TODO this structure doesnt serve the need anymore. simplify
def run_map2pcls(bool_with_noise):
    @cc.alert_cached
    def r_m2ps(path_unsc, overw, cs):        
        cf_loc = cs.cf
        maps = io.load_plamap(cf_loc, field=(0,1,2), nside_out=cs.nside_out)
        maps = mp.process_all(maps)
        tmask, pmask, pmask = io.load_one_mask_forallfreq()
        C_l_unsc = trsf.map2pcls(maps, tmask, pmask)
        io.save_data(C_l_unsc, path_unsc)
        return C_l_unsc

    @cc.alert_cached
    def r_ps2psc(path_sc, overw, cs, C_l_unsc):
        beamf = io.load_beamf(cs.freqcomb)
        C_l = pospec.process_all(C_l_unsc, cs.freqcomb, beamf, cs.nside_out, cs.cf['pa']["Spectrum_scale"])
        io.save_data(C_l, path_sc)
        print('Saved to {}'.format(path_sc))

    if bool_with_noise:
        cf_n = copy.deepcopy(csu.cf)
        cf_n['pa']['freqdset'] = csu.cf['pa']['freqdset']+"_diff"
        csu_n = Config(cf_n)
        cslist = [csu, csu_n]
        path_name_list = [
            (io.fh.psspec_unsc_path_name, io.fh.psspec_sc_path_name),
            (io.fh.psnoise_unsc_path_name, io.fh.psnoise_sc_path_name)
        ]
    else:
        cslist = [csu]
        path_name_list = [(io.fh.psspec_unsc_path_name, io.fh.psspec_sc_path_name)]
    
    for cs, (path_unsc, path_sc) in zip(cslist, path_name_list):
        C_l_unsc = r_m2ps(path_unsc, csu.overwrite_cache, cs)
        r_ps2psc(path_sc, csu.overwrite_cache, cs, C_l_unsc)

        
def run_map2cls_new(info_component, info_combination, powerspectrum_type):
    """
    Map can either consist of,
    noise, foreground, signal, non-separated == info_component
    Map can either be,
    combined, perfreq == info_combination
    """
    if info_component == 'signal':
        assert 0, "To be implemented"
    elif info_component == 'noise':
        cf_n = copy.deepcopy(csu.cf)
        cf_n['pa']['freqdset'] = csu.cf['pa']['freqdset']+"_diff"
        csu_n = Config(cf_n)
        cf_loc = csu_n.cf
    elif info_component == 'foreground':
        assert 0, "To be implemented"
    elif info_component == 'non-separated':
        cf_loc = csu.cf

    if info_combination == 'combined':
        assert 0, "To be implemented"
    elif info_combination == 'perfreq':
        cf_loc = csu.cf
        
    outpath_pow_usc_name = fn_gen.get_name('powerspectrum', info_component, info_combination, powerspectrum_type, 'unscaled')
    outpath_pow_sc_name = fn_gen.get_name('powerspectrum', info_component, info_combination, powerspectrum_type, 'scaled')
        
    maps = io.load_plamap(cf_loc, field=(0,1,2), nside_out=csu.nside_out)
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    
    maps = mp.process_all(maps)
    Cl_usc = trsf.map2cls(maps, 'pseudo', tmask, pmask)
    io.save_data(Cl_usc, outpath_pow_usc_name)
    
    beamf = io.load_beamf(csu.freqcomb)
    Cl_sc = pospec.process_all(Cl_usc, csu.freqcomb, beamf, csu.nside_out, csu.cf['pa']["Spectrum_scale"])
    io.save_data(Cl_sc, outpath_pow_sc_name)
        

@cc.alert_cached
def run_alm2cls(path_name, overw):
    cmb_tlm, cmb_elm, cmb_blm = cslib.load_alms('cmb', csu.sim_id)
    ClS_usc = np.array([hp.alm2cl([cmb_tlm, cmb_elm, cmb_blm])[:,:csu.cf['pa']['lmax']+1]])
    ClS = pospec.apply_scale(ClS_usc, csu.cf['pa']["Spectrum_scale"])
    io.save_data(ClS, path_name)


if __name__ == '__main__':
    # hpf.set_logger(DEBUG)
    bool_alm2spec = False

    bool_map2spec = True
    bool_map2psspec = False

    bool_map2psspec_new = False
    
    bool_with_noise = True
   

    if bool_map2spec:
        run_map2cls(bool_with_noise)

    if bool_map2psspec:
        run_map2pcls(bool_with_noise)
    
    if bool_map2psspec_new:
        run_map2cls_new('freq')
        if bool_with_noise:
            run_map2cls_new('noise')

    if bool_alm2spec:
        run_alm2cls(csu.overwrite_cache)