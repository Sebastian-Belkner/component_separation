#!/usr/local/bin/python
"""
run_map.py: script for generating empiric noisemaps or masks, which are based on maps.
Input are maps, input directory to be specified in the `rm_config.json`
output are maps or masks. output directory to be specified in `rm_config.json`

"""

__author__ = "S. Belkner"


import json
import os
import os.path
import platform
import healpy as hp
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from os import path
from typing import Dict, List, Optional, Tuple

import numpy as np

import component_separation
import component_separation.io as io
import component_separation.powspec as pw
import component_separation.transform_map as trsf
from component_separation.cs_util import Config
from component_separation.cs_util import Planckf, Plancks

with open(os.path.dirname(component_separation.__file__)+'/config_rm.json', "r") as f:
    cf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

freqdset = cf['pa']["freqdset"]
if "sim_id" in cf[mch][freqdset]:
        sim_id = cf[mch][freqdset]["sim_id"]
else:
    sim_id = ""

if cf["pa"]['nside_out'] is None:
    nside_out = cf["pa"]['nside_desc_map']
else:
    nside_out = cf["pa"]['nside_out']

freqfilter = cf['pa']["freqfilter"]
csu = Config(cf)

def create_difference_map(data_hm1, data_hm2):
    def _difference(data1, data2):
        ret = dict()
        for freq, val in data1.items():
            ret.update({freq: (data1[freq] - data2[freq])/2.})
        return ret
    ret_data = _difference(data_hm1, data_hm2)
    ret_data = trsf.process_all(ret_data)

    return ret_data


def cmblm2cmbmap(idx):
    CMB_in = dict()
    nsi = nside_out[1]
    cmb_tlm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=1)
    cmb_elm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=2)
    cmb_blm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=3)

    # TODO what is a reasonable nside for this?
    CMB_in["TQU"] = dict()
    CMB_in["TQU"] = hp.alm2map([cmb_tlm, cmb_elm, cmb_blm], nsi)
    return CMB_in["TQU"], nsi


def splitmaps2diffmap():
    """This routine loads the even-odd planck maps, takes the half-difference and
    saves it as a new map. 
    """
    buff = cf[mch][freqdset]['filename']
    freqfilter =  [
            '030',
            '044',
            '070',
            '100',
            '143',
            '217',
            '353',
            '545',
            '857',
        ]

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
        freqf = [f for f in freqfilter if f != FREQ]
        ns = nside_out[0] if int(FREQ)<100 else nside_out[1]
        outpathfile_name = outpath_name+cf[mch][freqdset]["out_filename"]\
            .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
            .replace("{freq}", FREQ)\
            .replace("{nside}", str(ns))\
            .replace("{00/1}", "00" if int(FREQ)<100 else "01")\
            .replace("{split}", freqdatsplit)\
            .replace("{sim_id}", sim_id)
        io.alert_cached(outpathfile_name)
        cf['pa']["freqfilter"] = freqf
        
        cf[mch][freqdset]['filename'] = filename_1of2
        data_hm1 = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)

        cf[mch][freqdset]['filename'] = filename_2of2
        data_hm2 = io.load_plamap(cf, field=(0,1,2), nside_out=nside_out)
        data_diff = create_difference_map(data_hm1, data_hm2)

        io.save_map(data_diff[FREQ], outpathfile_name)
        del data_diff


def map2mask():
    """This routine generates masks based on the standard SMICA or lensing masks and
        the noise variance due to the scanning strategy from planck
    """
    mask_path = cf[mch]['outdir_mask']
    freqfilter =  [
            '030',
            '044',
            '070',
            '100',
            '143',
            '217',
            '353',
            '545',
            '857',
        ]
    treshold = 3*1e-9
    freqdset = cf["pa"]['freqdset']
    maskbase = cf['pa']['mskset']
    for FREQ in csu.PLANCKMAPFREQ[:-2]:
        print(FREQ)
        freqf = [f for f in freqfilter if f != FREQ]
        cf['pa']["freqfilter"] = freqf
        noise_level = io.load_plamap_new(cf["pa"], field=7)
        noisevarmask = np.where(noise_level[FREQ]<treshold,True, False)
        if int(FREQ)<100:
            tmask, pmask, pmask = io.load_one_mask_forallfreq(1024)
        else:
            tmask, pmask, pmask = io.load_one_mask_forallfreq()
        comb_pmask =  pmask * noisevarmask
        comb_pmask_negated = pmask * ~noisevarmask

        comb_tmask =  pmask * noisevarmask
        comb_tmask_negated = pmask * ~noisevarmask
        print("Frequency:", FREQ)
        print("Mean noise,   sky coverage")
        print(30*"_")
        print(
            np.mean(noise_level[FREQ]), "1" ,"\n",
            np.mean(
                noise_level[FREQ] * pmask),
                np.sum(pmask/len(pmask)), "\n",
            np.mean(
                noise_level[FREQ] * comb_pmask),
                np.sum((comb_pmask)/len(pmask)), "\n",
            np.mean(
                noise_level[FREQ] * comb_pmask_negated),
                np.sum((comb_pmask_negated)/len(pmask)), "\n"
        )

        filename = "{LorH}_SkyMask_{freq}_{nside}_R3.{00/1}_full_{maskbase}_{p/t}mask_{s/l}patch.fits"\
                .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
                .replace("{freq}", FREQ)\
                .replace("{nside}", str(1024) if int(FREQ)<100 else str(2048))\
                .replace("{00/1}", "00" if int(FREQ)<100 else "01")\
                .replace("{maskbase}", maskbase)\
                .replace("{s/l}", 's')\
                .replace("{p/t}", 'p')
        io.save_map(comb_pmask, mask_path+filename)

        filename = "{LorH}_SkyMask_{freq}_{nside}_R3.{00/1}_full_{maskbase}_{p/t}mask_{s/l}patch.fits"\
                .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
                .replace("{freq}", FREQ)\
                .replace("{nside}", str(1024) if int(FREQ)<100 else str(2048))\
                .replace("{00/1}", "00" if int(FREQ)<100 else "01")\
                .replace("{maskbase}", maskbase)\
                .replace("{s/l}", 'l')\
                .replace("{p/t}", 'p')
        io.save_map(comb_pmask_negated, mask_path+filename)


        filename = "{LorH}_SkyMask_{freq}_{nside}_R3.{00/1}_full_{maskbase}_{p/t}mask_{s/l}patch.fits"\
                .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
                .replace("{freq}", FREQ)\
                .replace("{nside}", str(1024) if int(FREQ)<100 else str(2048))\
                .replace("{00/1}", "00" if int(FREQ)<100 else "01")\
                .replace("{maskbase}", maskbase)\
                .replace("{s/l}", 's')\
                .replace("{p/t}", 't')
        io.save_map(comb_tmask, mask_path+filename)

        filename = "{LorH}_SkyMask_{freq}_{nside}_R3.{00/1}_full_{maskbase}_{p/t}mask_{s/l}patch.fits"\
                .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
                .replace("{freq}", FREQ)\
                .replace("{nside}", str(1024) if int(FREQ)<100 else str(2048))\
                .replace("{00/1}", "00" if int(FREQ)<100 else "01")\
                .replace("{maskbase}", maskbase)\
                .replace("{s/l}", 'l')\
                .replace("{p/t}", 't')
        io.save_map(comb_tmask_negated, mask_path+filename)


if __name__ == '__main__':
    print(60*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(60*"$")

    run_emp_noisemap = True
    run_cmbmap = False
    run_mask = False
    run_synmap = False

    if run_emp_noisemap:
        print('Generating noise map from half ring maps..')
        splitmaps2diffmap()
        print('..Done')

    if run_cmbmap:
        """
        Derives CMB powerspectrum directly from alm data of pure CMB.
        """
        io.alert_cached(io.map_cmb_sc_path_name)
        CMB, nsi = cmblm2cmbmap(int(sim_id))  
        io.save_data(CMB, io.map_cmb_sc_path_name)

    if run_mask:
        map2mask()

    if run_synmap:
        cl2synmap()
