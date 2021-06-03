"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"

# '/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx

import functools
import json
import os
import platform
import os.path
from os import path
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start

import component_separation
from component_separation.cs_util import Planckf, Planckr, Plancks

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = cf["pa"]['nside']
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

freqdset = cf["pa"]['freqdset']
PLANCKMAPFREQ_f = [FREQ for FREQ in PLANCKMAPFREQ
    if FREQ not in cf['pa']["freqfilter"]]

freqfilter = cf['pa']["freqfilter"]


def _multi(a, b):
    return a*b

def read_pf(mask_path, mask_filename):
    return {FREQ: hp.read_map(
        '{mask_path}{mask_filename}'
        .format(
            mask_path = mask_path,
            mask_filename = mask_filename
                .replace("{freqdset}", freqdset)
                .replace("{freq}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
            ), dtype=np.bool)
            for FREQ in PLANCKMAPFREQ_f
        }

def read_single(mask_path, mask_filename):
    return hp.read_map(
        '{mask_path}{mask_filename}'
        .format(
            mask_path = mask_path,
            mask_filename = mask_filename))


def make_filenamestring(cf_local, desc='scaled'):
    """Helper function for generating unique filenames given te current configuration

    Args:
        cf (Dict): Configuration file - in general conf.json from root directory

    Returns:
        str: unique filename which may be used for spectra, weights, maps, etc..
    """    
    spectrum_scale = cf['pa']["Spectrum_scale"]
    mskset = cf_local['pa']['mskset'] # smica or lens
    freqdset = cf_local['pa']['freqdset'] # DX12 or NERSC
    lmax = cf_local['pa']["lmax"]
    lmax_mask = cf_local['pa']["lmax_mask"]
    freqfilter = cf_local['pa']["freqfilter"]
    specfilter = cf_local['pa']["specfilter"]
    smoothing_window = cf_local['pa']["smoothing_window"]
    max_polynom = cf_local['pa']["max_polynom"]
    if "sim_id" in cf[mch][freqdset]:
        sim_id = cf[mch][freqdset]["sim_id"]
    else:
        sim_id = ""
    if desc == 'raw':
        return '{sim_id}_{spectrum_scale}_{freqdset}_{mskset}_{lmax}_{lmax_mask}_{freqs}_{spec}_{split}.npy'.format(
            sim_id = sim_id,
            spectrum_scale = spectrum_scale,
            freqdset = freqdset,
            mskset = mskset,
            lmax = lmax,
            lmax_mask = lmax_mask,
            spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
            freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
            split = "Full" if cf_local['pa']["freqdatsplit"] == "" else cf_local['pa']["freqdatsplit"])
    else:
        return '{sim_id}_{freqdset}_{mskset}_{lmax}_{lmax_mask}_{smoothing_window}_{max_polynom}_{freqs}_{spec}_{split}.npy'.format(
            sim_id = sim_id,
            freqdset = freqdset,
            mskset = mskset,
            lmax = lmax,
            lmax_mask = lmax_mask,
            spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
            freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
            split = "Full" if cf_local['pa']["freqdatsplit"] == "" else cf_local['pa']["freqdatsplit"],
            smoothing_window = smoothing_window,
            max_polynom = max_polynom)

def load_data(path_name: str) -> Dict[str, Dict]:
    if os.path.isfile(path_name):
        data = np.load(path_name, allow_pickle=True)
        print('loaded {}'.format(path_name))
        if data.shape == ():
            return data.item()
        else:
            return data
    else:
        print("no existing data at {}".format(path_name))
        return None


def load_plamap(cf_local, field):
    """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
    Map data in `PATH/map/`.
    Args:
        cf (Dict): Configuration as coming from conf.json
        mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
        
    Returns:
        List[Dict]: Planck maps (data and masks) and some header information

    Doctest:
    >>> get_data(cf: Dict, mch: str) 
    NotSureWhatToExpect
    """

    freqdset = cf_local["pa"]['freqdset'] # NPIPE or DX12
    freqfilter = cf_local["pa"]["freqfilter"]
    nside = cf_local["pa"]["nside"]

    abs_path = cf_local[mch][freqdset]['ap']
    freq_filename = cf_local[mch][freqdset]['filename']
    if "sim_id" in cf_local[mch][freqdset]:
        sim_id = cf_local[mch][freqdset]["sim_id"]
    else:
        sim_id = ""
    mappath = {
        FREQ:'{abs_path}{freq_filename}'
            .format(
                abs_path = abs_path\
                    .replace("{sim_id}", sim_id)\
                    .replace("{split}", cf_local['pa']['freqdatsplit'] if "split" in cf_local[mch][freqdset] else ""),
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside[0]) if int(FREQ)<100 else str(nside[1]))
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                    .replace("{even/half1}", "even" if int(FREQ)>=100 else "half1")
                    .replace("{odd/half2}", "odd" if int(FREQ)>=100 else "half2")
                    .replace("{sim_id}", sim_id)\
                    .replace("{split}", cf_local['pa']['freqdatsplit'] if "split" in cf_local[mch][freqdset] else "")
                )
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter}

    maps = {
        FREQ: hp.read_map(mappath[FREQ], field=field)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
    }
    print("loaded {}".format(mappath))
    return maps


def load_mask_per_freq(dg_to=1024):
    maskset = cf['pa']['mskset']
    freqfilter = cf['pa']["freqfilter"]

    pmask_path = cf[mch][maskset]['pmask']["ap"]
    pmask_filename = cf[mch][maskset]['pmask']['filename']
    pmasks = [
        read_pf(pmask_path, a)
        for a in pmask_filename]
    pmask = {
        FREQ: functools.reduce(
            _multi,
            [a[FREQ] for a in pmasks])
                for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    pmask = {FREQ: hp.pixelfunc.ud_grade(pmask[FREQ], nside_out=dg_to if int(FREQ)<100 else 2048)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
        }
    
    tmask_path = cf[mch][maskset]['tmask']["ap"]
    tmask_filename = cf[mch][maskset]['tmask']['filename']
    tmask = read_pf(tmask_path, tmask_filename)
    tmask = {FREQ: hp.pixelfunc.ud_grade(tmask[FREQ], nside_out=dg_to)
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }
    return tmask, pmask, pmask


def load_one_mask_forallfreq(udgrade=False):
    maskset = cf['pa']['mskset']
    pmask_path = cf[mch][maskset]['pmask']["ap"]
    pmask_filename = cf[mch][maskset]['pmask']['filename']
    print('loading mask {}'.format(pmask_filename))
    pmasks = [
                read_single(pmask_path, a)
                for a in pmask_filename]
    pmask = functools.reduce(
            _multi,
            [a for a in pmasks])

    tmask_path = cf[mch][maskset]['tmask']["ap"]
    tmask_filename = cf[mch][maskset]['tmask']['filename']
    tmask = read_single(tmask_path, tmask_filename)
    if udgrade:
        tmask = hp.ud_grade(tmask, nside_out=udgrade)
        pmask = hp.ud_grade(pmask, nside_out=udgrade)

    return tmask, pmask, pmask


def load_truthspectrum(abs_path=""):
    return pd.read_csv(
        abs_path+cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)


def load_weights_smica(path_name):
    return np.loadtxt(path_name).reshape(2,7,4001)


@log_on_start(INFO, "Starting to load weights from {path_name}")
@log_on_end(DEBUG, "Weights loaded successfully")
def load_weights(path_name: str, indir_root: str = None, indir_rel: str = None, in_desc: str = None, fname: str = None) -> Dict[str, Dict]:
    if path_name == None:
        fending = ".npy"
        path_name = indir_root+indir_rel+in_desc+fname+fending
    if os.path.isfile(path_name):
        data = np.load(path_name, allow_pickle=True)
        print( "loaded {}".format(path_name))
        return data
    else:
        print("no existing weights at {}".format(path_name))
        return None


@log_on_start(INFO, "Starting to load synmap from {path_name}")
@log_on_end(DEBUG, "Synmap loaded successfully")
def load_synmap(path_name: str, indir_root: str = None, indir_rel: str = None, in_desc: str = None, fname: str = None) -> Dict[str, Dict]:
    if path_name == None:
        fending = ".npy"
        path_name = indir_root+indir_rel+in_desc+fname+fending
    if os.path.isfile(path_name):
        data = np.load(path_name, allow_pickle=True)
        print( "loaded {}".format(path_name))
        return data
    else:
        print("no existing map at {}".format(path_name))
        return None


@log_on_start(INFO, "Starting to load beamf functions from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully")
def load_beamf(freqcomb: List) -> Dict:
    """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

    Args:
        cf (Dict): Configuration as coming from conf.json
        mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
        freqcomb (List): Frequency channels which are to be ignored

    Returns:
        Dict: Planck beamfunctions
    """
    beamf = dict()
    for freqc in freqcomb:
        freqs = freqc.split('-')
        if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
            beamf.update({
                freqc: {
                    "HFI": fits.open(
                        "{bf_path}{bf_filename}"
                        .format(
                            bf_path = cf[mch]["beamf"]["HFI"]['ap'],
                            bf_filename = cf[mch]["beamf"]["HFI"]['filename']
                                .replace("{freq1}", freqs[0])
                                .replace("{freq2}", freqs[1])
                        ))
                    }
                })
        elif int(freqs[0]) < 100 and int(freqs[1]) >= 100:
            beamf.update({
                freqc: {
                    "HFI": fits.open(
                        "{bf_path}{bf_filename}"
                        .format(
                            bf_path = cf[mch]["beamf"]["HFI"]['ap'],
                            bf_filename = cf[mch]["beamf"]["HFI"]['filename']
                                .replace("{freq1}", freqs[1])
                                .replace("{freq2}", freqs[1])
                    ))
                }
            })
            beamf[freqc].update({
                "LFI": fits.open(
                        "{bf_path}{bf_filename}"
                        .format(
                            bf_path = cf[mch]["beamf"]["LFI"]['ap'],
                            bf_filename = cf[mch]["beamf"]["LFI"]['filename']
                    ))
            })
        if int(freqs[0]) < 100 and int(freqs[1]) < 100:
             beamf.update({
                freqc: {
                    "LFI": fits.open(
                        "{bf_path}{bf_filename}"
                        .format(
                            bf_path = cf[mch]["beamf"]["LFI"]['ap'],
                            bf_filename = cf[mch]["beamf"]["LFI"]['filename']
                    ))
                }})
    return beamf


@log_on_start(INFO, "Saving to {path_name}")
@log_on_end(DEBUG, "Data saved successfully to {path_name}")
def save_data(data: Dict[str, Dict], path_name: str, filename: str = 'default'):
    if os.path.exists(path_name):
        os.remove(path_name)
    np.save(path_name, data)
    print('Data saved to {}'.format(path_name))


@log_on_start(INFO, "Saving to {path_name}")
@log_on_end(DEBUG, "Data saved successfully to {path_name}")
def save_figure(mp, path_name: str, outdir_root: str = None, outdir_rel: str = None, out_desc: str = None, fname: str = None):
    if path_name == None:
        fending = ".jpg"
        path_name = outdir_root+outdir_rel+out_desc+fname+fending
    mp.savefig(path_name, dpi=144)
    mp.close()


@log_on_start(INFO, "Saving to {path_name}")
@log_on_end(DEBUG, "Data saved successfully to {path_name}")
def save_map(data, path_name: str):
    hp.write_map(path_name, data, overwrite=True)
    print("saved map to {}".format(path_name))

def iff_make_dir(outpath_name):
    if path.exists(outpath_name):
        pass
    else:
        os.makedirs(outpath_name)


"""The following lines define naming conventions for the files and directories
"""

total_filename = make_filenamestring(cf)
total_filename_raw = make_filenamestring(cf, 'raw')

out_spec_path = cf[mch]['outdir_spectrum_ap'] + cf['pa']["freqdset"] + "/"
iff_make_dir(out_spec_path)

out_map_path = cf[mch]['outdir_map_ap'] + cf['pa']["freqdset"] + "/"
iff_make_dir(out_map_path)
map_sc_filename = "MAP" + total_filename
map_sc_path_name = out_map_path + map_sc_filename


out_synmap_path = cf[mch]['outdir_map_ap'] + cf['pa']["freqdset"] + "/"
iff_make_dir(out_synmap_path)
synmap_sc_filename = "SYNMAP" + cf[mch][freqdset]['sim_id'] + total_filename
synmap_sc_path_name = out_synmap_path + map_sc_filename


spec_unsc_filename = "SPEC-RAW_" + total_filename_raw
out_spec_unsc_path_name = out_spec_path + spec_unsc_filename
out_spec_unsc_path_name = out_spec_unsc_path_name

spec_sc_filename = "SPEC" + total_filename
spec_sc_path_name = out_spec_path + spec_sc_filename

weight_path = cf[mch]['outdir_weight_ap'] + cf['pa']["freqdset"] + "/"
iff_make_dir(weight_path)

weight_path_name = weight_path + "WEIG_" + cf['pa']["Tscale"] + "_" + total_filename


import copy
cf_copy = copy.deepcopy(cf)


### the following lines are only needed for run_smica part of the code
buff = cf['pa']['freqdset']
if "diff" in buff:
    pass
else:
    cf_copy['pa']['freqdset'] = buff+'_diff'
noise_filename = make_filenamestring(cf_copy)
noise_filename_raw = make_filenamestring(cf_copy, 'raw')
noise_path = cf_copy[mch]['outdir_spectrum_ap'] + cf_copy['pa']["freqdset"] + "/"

iff_make_dir(noise_path)
noise_unsc_path_name = noise_path + 'SPEC-RAW_' + noise_filename_raw
noise_sc_path_name = noise_path + "SPEC" + noise_filename

cf_copy['pa']['freqdset'] = buff

