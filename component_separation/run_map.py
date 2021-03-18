#!/usr/local/bin/python
"""
run_map.py: script for generating maps

"""

__author__ = "S. Belkner"


import json
import logging
import logging.handlers
import os
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import healpy as hp
import functools

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks

with open('config.json', "r") as f:
    cf = json.load(f)


uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir_spectrum']
map_path = cf[mch]['outdir_map']
mask_path = cf[mch]['outdir_mask']
indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]

def create_difference_map(FREQ):
    def _difference(data1, data2):
        ret = dict()
        for freq, val in data1.items():
            ret.update({freq: (data1[freq] - data2[freq])/2.})
        return ret
    ret_data = _difference(data_hm1, data_hm2)
    ret_data = prep.preprocess_all(ret_data)

    return ret_data


if __name__ == '__main__':
    print(60*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(60*"$")

    empiric_noisemap = False
    make_mask = True

    if empiric_noisemap:
        """This routine loads the even-odd planck maps, takes the half-difference and
        saves it as a new map. 
        """
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
        for FREQ in PLANCKMAPFREQ[:-2]:
            freqf = [f for f in freqfilter if f != FREQ]
            cf['pa']["freqfilter"] = freqf
            print(freqf)
            cf['pa']["freqdset"] = "DX12-split1"
            data_hm1 = io.load_plamap_new(cf['pa'], field=(0,1,2))
            cf['pa']["freqdset"] = "DX12-split2"
            data_hm2 = io.load_plamap_new(cf['pa'], field=(0,1,2))
            data_diff = create_difference_map(data_hm1, data_hm2)
            filename = "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full-eohd.fits"\
                .replace("{LorH}", "LFI" if int(FREQ)<100 else "HFI")\
                .replace("{freq}", FREQ)\
                .replace("{nside}", str(1024) if int(FREQ)<100 else str(2048))\
                .replace("{00/1}", "00" if int(FREQ)<100 else "01")
            io.save_map(data_diff[FREQ], map_path+filename)
            del data_diff

    if make_mask:
        """This routine generates masks based on the standard SMICA or lensing masks and
            the noise variance due to the scanning strategy from planck
        """

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
        for FREQ in PLANCKMAPFREQ[:-2]:
            print(FREQ)
            freqf = [f for f in freqfilter if f != FREQ]
            cf['pa']["freqfilter"] = freqf
            noise_level = io.load_plamap_new(cf["pa"], field=7)
            noisevarmask = np.where(noise_level[FREQ]<treshold,True, False)
            tmask, pmask, pmask = io.load_mask(cf["pa"])
            comb_pmask =  pmask[FREQ] * noisevarmask
            comb_pmask_negated = pmask[FREQ] * ~noisevarmask

            comb_tmask =  pmask[FREQ] * noisevarmask
            comb_tmask_negated = pmask[FREQ] * ~noisevarmask
            print("Frequency:", FREQ)
            print("Mean noise,   sky coverage")
            print(30*"_")
            print(
                np.mean(noise_level[FREQ]), "1" ,"\n",
                np.mean(
                    noise_level[FREQ] * pmask[FREQ]),
                    np.sum(pmask[FREQ]/len(pmask[FREQ])), "\n",
                np.mean(
                    noise_level[FREQ] * pmask[FREQ] * noisevarmask),
                    np.sum((pmask[FREQ]*noisevarmask)/len(pmask[FREQ])), "\n"
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