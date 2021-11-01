#!/usr/local/bin/python
"""
app_map.py: script for generating empiric noisemaps or masks.
Input are maps, input directory to be specified in the `config_rm.json`
output are maps or masks. output directory to be specified in `config_rm.json`

"""

__author__ = "S. Belkner"

import numpy as np

from component_separation.io import IO
import component_separation.cachechecker as cc
from component_separation.cs_util import Config

import component_separation.transformer as trsf
import component_separation.map as mp

csu = Config()
io = IO(csu)


@cc.alert_cached
def run_cmbalm2map(pathname, overwr):
    """
    Derives map directly from alm data of pure CMB.
    """
    nsi = csu.nside_out[1]
    cmb_tlm, cmb_elm, cmb_blm = io.load_alms('cmb', csu.sim_id)
    CMB = trsf.alm_s2map(cmb_tlm, cmb_elm, cmb_blm, nsi)
    io.save_data(CMB, pathname)


def run_splitmaps2diffmap():
    """
    This routine loads the even-odd planck maps, takes the half-difference and
    saves it as a new map. 
    """
    cf = csu.cf
    mch = csu.mch
    freqdset = csu.freqdset
    sim_id = csu.sim_id
    buff = csu.cf[mch][freqdset]['filename']
    nside_out = csu.nside_out

    freqdatsplit = cf['pa']["freqdatsplit"]
    cf[mch][freqdset]['ap'] = cf[mch][freqdset]['ap']\
            .replace("{sim_id}", sim_id)\
            .replace("{split}", freqdatsplit if "split" in cf[mch][freqdset] else "")
    
    filename_1of2 = buff\
        .replace("{split}", freqdatsplit if "split" in cf[mch][freqdset] else "")\
        .replace("{even/odd/half1/half2}", "{even/half1}")\
        .replace("{n_of_2}", "1of2")

    filename_2of2 = cf[mch][freqdset]['filename'] = buff\
        .replace("{split}", freqdatsplit if "split" in cf[mch][freqdset] else "")\
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
        
        cf['pa']["freqfilter"] = freqf
        cf[mch][freqdset]['filename'] = filename_1of2
        data_hm1 = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)

        cf[mch][freqdset]['filename'] = filename_2of2
        data_hm2 = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
        
        data_diff = mp.create_difference_map(data_hm1, data_hm2)
        data_diff = trsf.process_all(data_diff)

        io.save_map(data_diff[FREQ], outpathfile_name)


if __name__ == '__main__':
    bool_emp_noisemap = True
    bool_cmbmap = False
    bool_synmap = False

    if bool_emp_noisemap:
        run_splitmaps2diffmap()

    if bool_cmbmap:
        run_cmbalm2map(io.fh.map_cmb_sc_path_name, csu.overwrite_cache)

    if bool_synmap:
        assert 0, 'To be implemented'
        # run_cl2synmap()