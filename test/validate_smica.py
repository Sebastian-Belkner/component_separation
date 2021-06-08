"""
validate_transferfunction.py: compare CMB powerspectrum from smica to planck cmb simulation, using the testsuit

"""

import json
import logging
import logging.handlers
import os
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import healpy as hp

import component_separation
import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf
import healpy as hp

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]

def mv(icov_l):
    spec_pick = "EE"
    def _weightspec(icov_l):
        for specc, data in icov_l.items():
            if specc == spec_pick:
                icovsum = np.array([np.sum(icov_l[specc][l]) for l in range(lmax)])
                retspec = {specc: {'optimal-optimal': np.array([1/icovsum_l if icovsum_l is not None else 0 for icovsum_l in icovsum])}}
        return retspec
    return _weightspec(icov_l)


def calc_transferfunction(smica_cmb, C_lin):
    # emp_map = io.load_plamap(cf, field=(0,1,2))
    # Tsyn_map, Psyn_map, Psyn_map = io.load_data(io.synmap_sc_path_name+'_0.npy')

    ## either this, and use synfast for the smica output?
    
    # now, calculate transferfunction from c_lin_mv to smica_cmb
    # perhaps, use eq (9) from diffuse comp sep paper
    # tf = 1/(beamf * pixwindow) * sqrt(mean(c_lin_mv/smica_cmb))
    tf = 1.*np.sqrt(smica_cmb/C_lin)
    return tf


def calc_crosscorrelation(map_smica, map_in):
    def _ud_grade(data, FREQ):
        if int(FREQ)<100:
            return hp.pixelfunc.ud_grade(data, nside_out=1024)
        else:
            return hp.pixelfunc.ud_grade(data, nside_out=2048)
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    spectrum = ps.map2cls(
                tqumap=map_smica,
                tmask=_ud_grade(tmask, 100),
                pmask=_ud_grade(pmask, 100),
                lmax=lmax,
                lmax_mask=lmax_mask,
                tqumap2=map_in,
                tmask2=_ud_grade(tmask, 100),
                pmask2=_ud_grade(pmask, 100)
        )
    return spectrum


def bin_it_1D(data, bins, offset=0):
    ret = np.ones(len(bins))
    for k in range(bins.shape[0]):
        ret[k] = np.mean(np.nan_to_num(data[int(bins[k][0]):int(bins[k][1])]))
    return np.nan_to_num(ret)


if __name__ == '__main__':
    filename_raw = io.total_filename_raw
    filename = io.total_filename
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated filename(s) for this session: {}".format(filename_raw))
    print(filename)
    print(40*"$")
    bins = const.linear_equisized_bins_100 #const.SMICA_lowell_bins    #

    # Load smica spectrum of interest
    smica_spec = io.load_data(io.specsmica_sc_path_name)[0,0,:]

    # Load planck cmb simulation of interest
    #TODO this should be somehwat related to the SMICA output, but cannot find the NPIPE data atm. gotta ask Julien
    # for now, use anything as long as its a CMB map, preferably a E map.
    # if its Q,U, then transform first
    
    # cmb_map_in = hp.read_map("/global/cfs/cdirs/cmb/data/planck2018/pr3/cmbmaps/dx12_v3_smica_cmb_raw.fits", field=(0,1,2))
    # cmb_spec_in = hp.anafast(cmb_map_in)
    # io.save_data(cmb_spec_in, cf[mch]['outdir_misc_ap']+"cmb_spec_in.npy")
    cmb_spec_in = io.load_data(cf[mch]['outdir_misc_ap']+"cmb_spec_in.npy")
    
    #two paths possible from here
    #   1. compare smica_spec with hp.anafast(cmb_map_in) using eq 9 from diffuse comp sep paper
    #   beware, cmb_map_in most likely is IQU -> transform first? pol=true parameter takes care of it?
    cmb_spec_in_bnd = bin_it_1D(cmb_spec_in[1,:]*1e12, bins=bins)
    tf = calc_transferfunction(smica_spec, cmb_spec_in_bnd)   
    io.save_data(tf, cf[mch]['outdir_misc_ap']+"tf.npy")


    #   2. crosscorrelate hp.synfast(smica_spec) with cmb_map_in
    #   beware, cmb_map_in most likely is IQU -> transform first? pol=true parameter takes care of it?
    if False:
        crc = calc_crosscorrelation(hp.synfast(smica_spec, nside=2048), cmb_map_in)
        io.save_data(crc, cf[mch]['outdir_misc_ap']+"crc.npy")

    # this is a possible alternative to the above cmb_map_in, to be fixed
    # or load cmb sim map for each frequency and combine using mv,
    # C_lin_all = [hp.read_map("<path_to_sim_cmb_cls>{}".format(freq)) for freq in PLANCKMAPFREQ_f]
    # cov_lin_all = pw.build_covmatrices(C_lin_all, lmax, freqfilter, specfilter)
    # icov_lin_all = pw.invert_covmatrices(cov_lin_all)
    # c_lin_mv = mv(icov_lin_all)
    # smica_cmb_map = hp.synfast(smica_spec, nside=2048)


