"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"

# '/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx

import functools
import json
import os
import os.path
import platform
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start
from scipy import stats

import component_separation
from component_separation.cs_util import Planckf, Planckr, Plancks



PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

compath = os.path.dirname(component_separation.__file__)[:-21]
with open('{}/config.json'.format(compath), "r") as f:
    cf = json.load(f)


def make_filenamestring(cf):
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
    
    return '{freqdset}-msk_{mskset}-lmax_{lmax}-lmaxmsk_{lmax_mask}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        mskset = mskset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])


def load_data(path_name: str) -> Dict[str, Dict]:
    if os.path.isfile(path_name):
        data = np.load(path_name, allow_pickle=True)
        return data
    else:
        print("no existing data at {}".format(path_name))
        return None


def load_plamap_new(pa: Dict):
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

    freqdset = pa['freqdset'] # NPIPE or DX12
    freqfilter = pa["freqfilter"]
    nside = pa["nside"]

    freq_filename = cf[mch][freqdset]['filename']
    
    indir_path = cf[mch]['indir']
    freq_path = cf[mch][freqdset]['path']

    mappath = {
        FREQ:'{path}{freq_path}{freq_filename}'
            .format(
                path = indir_path,
                freq_path = freq_path,
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside[0]) if int(FREQ)<100 else str(nside[1]))
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                    .replace("{even/half1}", "even" if int(FREQ)>=100 else "half1")
                    .replace("{odd/half2}", "odd" if int(FREQ)>=100 else "half2")
                )
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter}

    maps = {
        FREQ: hp.read_map(mappath[FREQ], field=(0,1,2))
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
    }
    return maps


def load_mask(pa: Dict):
    pass

#%% Collect maps
@log_on_start(INFO, "Starting to load pla maps")
@log_on_end(DEBUG, "pla maps loaded successfully")
def load_plamap(pa: Dict) -> List[Dict]:
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

    ### Grab all necessary parameters from config

    mskset = pa['mskset'] # smica or lens
    lownoise_patch = pa['lownoise_patch']
    freqdset = pa['freqdset'] # NPIPE or DX12
    freqfilter = pa["freqfilter"]
    specfilter = pa["specfilter"]
    nside = pa["nside"]

    indir_path = cf[mch]['indir']
    freq_path = cf[mch][freqdset]['path']
    
    if mskset == 'hitshist':
        tmask_path = cf[mch][mskset]['mask']["path"]
        pmask_path = cf[mch][mskset]['mask']["path"]
    
        tmask_filename = cf[mch][mskset]['mask']['filename']
        pmask_filename = cf[mch][mskset]['mask']['filename']
    else:
        tmask_path = cf[mch][mskset]['tmask']["path"]
        pmask_path = cf[mch][mskset]['pmask']["path"]

        tmask_filename = cf[mch][mskset]['tmask']['filename']
        pmask_filename = cf[mch][mskset]['pmask']['filename']

    freq_filename = cf[mch][freqdset]['filename']

    def _read(mask_path, mask_filename):
        tresh_low, tresh_up = 0.0, 0.1
        return {FREQ: hp.read_map(
                '{path}{mask_path}{mask_filename}'
                .format(
                    path = indir_path,
                    mask_path = mask_path,
                    mask_filename = mask_filename
                        .replace("{freqdset}", freqdset)
                        .replace("{freq}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                        .replace("{tresh_low}", str(tresh_low))
                        .replace("{tresh_up}", str(tresh_up))
                        .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                    ), dtype=np.bool)
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }
    def _multi(a,b):
        return a*b

    def _read_noisevarmask(FREQ):
        f = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/map/frequency/HFI_SkyMap_{}-field_2048_R3.01_full.fits".format(FREQ)
        boundary = None #such that it covers about 25% of pixels for all masks
        boundary= {
            "030": 2*1e-9,
            "044": 2*1e-9,
            "070": 2*1e-9,
            "100": 2*1e-9,
            "143": 2*1e-9,
            "217": 2*1e-9,
            "353": 2*1e-9
        }
        noise_level = hp.read_map(f, field=7)
        noisevarmask = np.where(noise_level<boundary[FREQ],True, False)
    ### Build paths and filenames from config information
    mappath = {
        FREQ:'{path}{freq_path}{freq_filename}'
            .format(
                path = indir_path,
                freq_path = freq_path,
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside[0]) if int(FREQ)<100 else str(nside[1]))
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                    .replace("{even/half1}", "even" if int(FREQ)>=100 else "half1")
                    .replace("{odd/half2}", "odd" if int(FREQ)>=100 else "half2")
                )
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    if tmask_filename is None:
        tmask = None
        tmask_d = None
    else:
        tmask = _read(tmask_path, tmask_filename)
        tmask_d = {FREQ: hp.pixelfunc.ud_grade(tmask[FREQ], nside_out=nside[0])
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }

    if pmask_filename is None:
        pmask = None
        pmask_d = None
    else:
        pmasks = [_read(pmask_path, a) for a in pmask_filename]
        pmask = {FREQ: functools.reduce(_multi, [a[FREQ] for a in pmasks])
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }
        if lownoise_patch == True: #this is only for masking high noise variance area
            for FREQ in PLANCKMAPFREQ:
                if FREQ not in freqfilter:
                    pmask[FREQ] = pmask[FREQ]*_read_noisevarmask(FREQ)
        pmask_d = {FREQ: hp.pixelfunc.ud_grade(pmask[FREQ], nside_out=nside[0])
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }


    ## Decide which maps to load 
    def _load_data(path_name):
        return load_data(path_name)

    diff_data = {}
    for FREQ in PLANCKMAPFREQ:
        if FREQ not in freqfilter:
            if mappath[FREQ].endswith('npy'):
                diff_data.update({FREQ: _load_data(mappath[FREQ])})
                
    flag = False
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter and ("T" in spec):
            flag = True
            break
    if flag and (tmask is not None):
        tmap = {
            FREQ: {
                "map": hp.read_map(mappath[FREQ], field=0)
                        if mappath[FREQ].endswith('fits')
                        else diff_data[FREQ][0][FREQ]["map"],
                "mask": tmask_d[FREQ] if int(FREQ)<100 else tmask[FREQ]
                }for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
        }
    elif flag and (tmask is None):
        print("Temperature spectra requested, but no Temperature mask provided. Spectra including temperature are skipped")
        tmap = None
    elif flag == False:
        tmap = None

    qmap = {
        FREQ: {
            "map": hp.read_map(mappath[FREQ], field=1)
                    if mappath[FREQ].endswith('fits')
                    else diff_data[FREQ][1][FREQ]["map"],
            "mask": pmask_d[FREQ] if int(FREQ)<100 else pmask[FREQ]
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    umap = {
        FREQ: {
            "map": hp.read_map(mappath[FREQ], field=2)
                    if mappath[FREQ].endswith('fits')
                    else diff_data[FREQ][2][FREQ]["map"],
            "mask": pmask_d[FREQ] if int(FREQ)<100 else pmask[FREQ]
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]


def load_truthspectrum(abspath=""):
    return pd.read_csv(
        abspath+cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)


def load_hitsvar(pa: Dict, field=7, abs_path=""):
    freqdset = pa['freqdset'] # NPIPE or DX12
    freqfilter = pa["freqfilter"]
    nside = pa["nside"]

    indir_path = abs_path+cf[mch]['indir']

    freq_path = cf[mch][freqdset]['path']

    freq_filename = cf[mch][freqdset]['filename']

    ### Build paths and filenames from config information
    mappath = {
        FREQ:'{path}{freq_path}{freq_filename}'
            .format(
                path = indir_path,
                freq_path = freq_path,
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside[0]) if int(FREQ)<100 else str(nside[1]))
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                )
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    hitsmap = {
        FREQ: hp.read_map(mappath[FREQ], field=field)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
        }
    return hitsmap


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
        return data.item()
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


@log_on_start(INFO, "Starting to load spectrum from {path_name}")
@log_on_end(DEBUG, "Spectrum loaded successfully")
def load_spectrum(path_name: str, indir_root: str = None, indir_rel: str = None, in_desc: str = None, fname: str = None) -> Dict[str, Dict]:
    if path_name == None:
        fending = ".npy"
        path_name = indir_root+indir_rel+in_desc+fname+fending
    if os.path.isfile(path_name):
        data = np.load(path_name, allow_pickle=True)
        print( "loaded {}".format(path_name))
        return data.item()
    else:
        print("no existing spectrum at {}".format(path_name))
        return None
        

@log_on_start(INFO, "Starting to load beamf functions from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully")
def load_beamf(freqcomb: List, abs_path: str = "") -> Dict:
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
def save_map(data: Dict[str, Dict], path_name: str):
    hp.write_map(path_name, data, overwrite=True)
    print("saved map to {}".format(path_name))