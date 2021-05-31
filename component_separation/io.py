"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"

# '/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx

import functools
import json
import os
import platform
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

abs_path = cf[mch]['abs_path']
freqdset = cf["pa"]['freqdset']
PLANCKMAPFREQ_f = [FREQ for FREQ in PLANCKMAPFREQ
    if FREQ not in cf['pa']["freqfilter"]]

indir_path = cf[mch]['indir']
freqfilter = cf['pa']["freqfilter"]


def _multi(a, b):
    return a*b

def read_pf(mask_path, mask_filename):
    return {FREQ: hp.read_map(
        '{abs_path}{path}{mask_path}{mask_filename}'
        .format(
            abs_path = abs_path,
            path = indir_path,
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
        '{abs_path}{path}{mask_path}{mask_filename}'
        .format(
            abs_path = abs_path,
            path = indir_path,
            mask_path = mask_path,
            mask_filename = mask_filename))


def make_filenamestring(cf, desc='scaled'):
    """Helper function for generating unique filenames given te current configuration

    Args:
        cf (Dict): Configuration file - in general conf.json from root directory

    Returns:
        str: unique filename which may be used for spectra, weights, maps, etc..
    """    
    mskset = cf['pa']['mskset'] # smica or lens
    freqdset = cf['pa']['freqdset'] # DX12 or NERSC
    lmax = cf['pa']["lmax"]
    lmax_mask = cf['pa']["lmax_mask"]
    freqfilter = cf['pa']["freqfilter"]
    specfilter = cf['pa']["specfilter"]
    smoothing_window = cf['pa']["smoothing_window"]
    max_polynom = cf['pa']["max_polynom"]
    if desc == 'raw':
        return '{freqdset}-msk_{mskset}-lmax_{lmax}-lmaxmsk_{lmax_mask}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
            freqdset = freqdset,
            mskset = mskset,
            lmax = lmax,
            lmax_mask = lmax_mask,
            spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
            freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    else:
        return '{freqdset}-msk_{mskset}-lmax_{lmax}-lmaxmsk_{lmax_mask}-smoothing_{smoothing_window}_{max_polynom}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
            freqdset = freqdset,
            mskset = mskset,
            lmax = lmax,
            lmax_mask = lmax_mask,
            spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
            freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
            split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"],
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


def load_plamap(cf: Dict, field, split_desc=''):
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

    freqdset = cf["pa"]['freqdset'] # NPIPE or DX12
    freqfilter = cf["pa"]["freqfilter"]
    nside = cf["pa"]["nside"]

    freq_filename = cf[mch][freqdset]['filename']
    
   
    indir_path = cf[mch]['indir']
    freq_path = cf[mch][freqdset]['path']
    if mch == "XPS":
         abs_path = cf[mch]['abs_path']
    else:
        abs_path = ""



    if 'sim_id' in cf[mch][freqdset]:
        sim_id = cf[mch][freqdset]["sim_id"]
    else:
        sim_id = ""

    mappath = {
        FREQ:'{abs_path}{path}{freq_path}{freq_filename}'
            .format(
                abs_path = abs_path,
                path = indir_path,
                freq_path = freq_path
                    .replace("{sim_id}", sim_id)
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else ""),
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside[0]) if int(FREQ)<100 else str(nside[1]))
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                    .replace("{even/half1}", "even" if int(FREQ)>=100 else "half1")
                    .replace("{odd/half2}", "odd" if int(FREQ)>=100 else "half2")
                    .replace("{sim_id}", sim_id)
                    .replace("{n_of_2}", "1of2" if split_desc == "1" else "2of2")
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

    pmask_path = cf[mch][maskset]['pmask']["path"]
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
    
    tmask_path = cf[mch][maskset]['tmask']["path"]
    tmask_filename = cf[mch][maskset]['tmask']['filename']
    tmask = read_pf(tmask_path, tmask_filename)
    tmask = {FREQ: hp.pixelfunc.ud_grade(tmask[FREQ], nside_out=dg_to)
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }
    return tmask, pmask, pmask


def load_one_mask_forallfreq(udgrade=False):
    maskset = cf['pa']['mskset']
    pmask_path = cf[mch][maskset]['pmask']["path"]
    pmask_filename = cf[mch][maskset]['pmask']['filename']
    print('loading mask {}'.format(pmask_filename))
    pmasks = [
                read_single(pmask_path, a)
                for a in pmask_filename]
    pmask = functools.reduce(
            _multi,
            [a for a in pmasks])

    tmask_path = cf[mch][maskset]['tmask']["path"]
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
def load_beamf(freqcomb: List, abs_path: str = abs_path) -> Dict:
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
                        "{abs_path}{indir_path}{bf_path}{bf_filename}"
                        .format(
                            abs_path = abs_path,
                            indir_path = cf[mch]['indir'],
                            bf_path = cf[mch]["beamf"]["HFI"]['path'],
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
                        "{abs_path}{indir_path}{bf_path}{bf_filename}"
                        .format(
                            abs_path = abs_path,
                            indir_path = cf[mch]['indir'],
                            bf_path = cf[mch]["beamf"]["HFI"]['path'],
                            bf_filename = cf[mch]["beamf"]["HFI"]['filename']
                                .replace("{freq1}", freqs[1])
                                .replace("{freq2}", freqs[1])
                    ))
                }
            })
            beamf[freqc].update({
                "LFI": fits.open(
                        "{abs_path}{indir_path}{bf_path}{bf_filename}"
                        .format(
                            abs_path = abs_path,
                            indir_path = cf[mch]['indir'],
                            bf_path = cf[mch]["beamf"]["LFI"]['path'],
                            bf_filename = cf[mch]["beamf"]["LFI"]['filename']
                    ))
            })
        if int(freqs[0]) < 100 and int(freqs[1]) < 100:
             beamf.update({
                freqc: {
                    "LFI": fits.open(
                        "{abs_path}{indir_path}{bf_path}{bf_filename}"
                        .format(
                            abs_path = abs_path,
                            indir_path = cf[mch]['indir'],
                            bf_path = cf[mch]["beamf"]["LFI"]['path'],
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

component_separation_path = 'project/component_separation/'

total_filename = make_filenamestring(cf)
total_filename_raw = make_filenamestring(cf, 'raw')
spec_path = cf[mch]['outdir_spectrum'] + cf['pa']["freqdset"] + "/"
spec_unsc_path_name = spec_path + '-raw-' + total_filename_raw
spec_sc_filename = "-" + cf['pa']["Spectrum_scale"] + "-" + total_filename
spec_sc_path_name = spec_path + spec_sc_filename

weight_path = abs_path + component_separation_path + cf[mch]['outdir_weight'] + cf['pa']["freqdset"] + "/"
weight_path_name = weight_path + "-" + cf['pa']["Tscale"] + "-" + total_filename

buff = cf['pa']['freqdset']
cf['pa']['freqdset'] = buff+'-diff'
noise_filename = make_filenamestring(cf)
noise_filename_raw = make_filenamestring(cf, 'raw')
noise_path = cf[mch]['outdir_spectrum'] + cf['pa']["freqdset"] + "/"
noise_unsc_path_name = noise_path + '-raw-' + noise_filename
noise_sc_path_name = noise_path + "-" + cf['pa']["Spectrum_scale"] + "-" + noise_filename

cf['pa']['freqdset'] = buff