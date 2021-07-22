#!/usr/local/bin/python
"""
run_map.py: script for generating empiric noisemaps or masks, which are based on maps.
Input are maps, input directory to be specified in the `rm_config.json`
output are maps or masks. output directory to be specified in `rm_config.json`

"""

__author__ = "S. Belkner"


import json
import os
import healpy as hp
from os import path
from typing import Dict, List, Optional, Tuple

import numpy as np

import component_separation
from component_separation.io import IO
import component_separation.io as csio
import component_separation.powspec as pw
import component_separation.transform_map as trsf
from component_separation.cs_util import Config

with open(os.path.dirname(component_separation.__file__)+'/config_rm.json', "r") as f:
    cf = json.load(f)
csu = Config(cf)
io = IO(csu)


@csio.alert_cached
def run_cmbmap(pathname, overwr):
    """
    Derives CMB powerspectrum directly from alm data of pure CMB.
    """
    nsi = csu.nside_out[1]
    cmb_tlm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(csu.sim_id), hdu=1)
    cmb_elm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(csu.sim_id), hdu=2)
    cmb_blm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(csu.sim_id), hdu=3)

    # TODO what is a reasonable nside for this?
    CMB = hp.alm2map([cmb_tlm, cmb_elm, cmb_blm], nsi)
    io.save_data(CMB, pathname)


def run_splitmaps2diffmap():
    """This routine loads the even-odd planck maps, takes the half-difference and
    saves it as a new map. 
    """
    def create_difference_map(data_hm1, data_hm2):
        def _difference(data1, data2):
            ret = dict()
            for freq, val in data1.items():
                ret.update({freq: (data1[freq] - data2[freq])/2.})
            return ret
        ret_data = _difference(data_hm1, data_hm2)
        ret_data = trsf.process_all(ret_data)
        return ret_data
    mch = csu.mch
    freqdset = csu.freqdset
    sim_id = csu.sim_id
    buff = csu.cf[mch][freqdset]['filename']
    nside_out = csu.nside_out

    @csio.alert_cached
    def wrap(outpath_filename, overw, freqf, filename_1of2, filename_2of2):
        cf['pa']["freqfilter"] = freqf
        cf[mch][freqdset]['filename'] = filename_1of2
        data_hm1 = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)

        cf[mch][freqdset]['filename'] = filename_2of2
        data_hm2 = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
        data_diff = create_difference_map(data_hm1, data_hm2)

        io.save_map(data_diff[FREQ], outpathfile_name)

    freqdatsplit = cf['pa']["freqdatsplit"]
    cf[mch][freqdset]['ap'] = cf[mch][freqdset]['ap']\
            .replace("{sim_id}", sim_id)\
            .replace("{split}", freqdatsplit if "split" in cf[mch][freqdset] else "")
    
    filename_1of2 = buff\
        .replace("{split}", freqdatsplit if "split" in cf[mch][freqdset] else "")\
        .replace("{even/odd/half1/half2}", "{even/half1}")\
        .replace("{n_of_2}", "1of2")

    filename_2of2 = cf[mch][freqdset]['filename'] = buff\
        .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")\
        .replace("{even/odd/half1/half2}", "{odd/half2}")\
        .replace("{n_of_2}", "2of2")
    
    outpath_name = cf[mch]["outdir_map_ap"]

    for FREQ in csu.PLANCKMAPFREQ_f:
        freqf = [f for f in csu.PLANCKMAPFREQ if f != FREQ]
        outpathfile_name = outpath_name+cf[mch][freqdset]["out_filename"]\
            .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
            .replace("{freq}", FREQ)\
            .replace("{nside}", str(nside_out[0]) if int(FREQ)<100 else str(nside_out[1]))\
            .replace("{00/1}", "00" if int(FREQ)<100 else "01")\
            .replace("{split}", freqdatsplit)\
            .replace("{sim_id}", sim_id)
        wrap(outpathfile_name, csu.overwrite_cache, freqf, filename_1of2, filename_2of2)


if __name__ == '__main__':
    bool_emp_noisemap = True
    bool_cmbmap = False
    bool_synmap = False

    if bool_emp_noisemap:
        run_splitmaps2diffmap()

    if bool_cmbmap:
        run_cmbmap(io.fh.map_cmb_sc_path_name, csu.overwrite_cache)

    if bool_synmap:
        run_cl2synmap()