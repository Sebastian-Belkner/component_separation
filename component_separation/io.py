"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"

# '/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx

from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import functools
import os.path
import numpy as np
from typing import Dict, List, Optional, Tuple
from component_separation.cs_util import Planckf, Plancks, Planckr
import healpy as hp
from scipy import stats
import json
import platform
import component_separation
compath = os.path.dirname(component_separation.__file__)[:-21]

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

with open('{}/config.json'.format(compath), "r") as f:
    cf = json.load(f)


def make_filenamestring(cf):
    mskset = cf['pa']['mskset'] # smica or lens
    freqdset = cf['pa']['freqdset'] # DX12 or NERSC
    lmax = cf['pa']["lmax"]
    lmax_mask = cf['pa']["lmax_mask"]
    freqfilter = cf['pa']["freqfilter"]
    specfilter = cf['pa']["specfilter"]
    tscale = cf['pa']["Tscale"]
    
    return '{freqdset}-msk_{mskset}-lmax_{lmax}-lmaxmsk_{lmax_mask}-{Tscale}_tscale-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        mskset = mskset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        Tscale = tscale,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])


#%% Collect maps
@log_on_start(INFO, "Starting to load pla maps")
@log_on_end(DEBUG, "Data loaded successfully")
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
    freqdset = pa['freqdset'] # NPIPE or DX12
    freqfilter = pa["freqfilter"]
    specfilter = pa["specfilter"]
    nside = pa["nside"]
    tresh_low, tresh_up = 0.0, 0.1

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
        pmask_d = {FREQ: hp.pixelfunc.ud_grade(pmask[FREQ], nside_out=nside[0])
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
                }


    ## Decide which maps to load 
    flag = False
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter and ("T" in spec):
            flag = True
            break
    if flag and (tmask is not None):
        tmap = {
            FREQ: {
                "map": hp.read_map(mappath[FREQ], field=0),
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
            "map": hp.read_map(mappath[FREQ], field=1),
            "mask": pmask_d[FREQ] if int(FREQ)<100 else pmask[FREQ]
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    umap = {
        FREQ: {
            "map": hp.read_map(mappath[FREQ], field=2),
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


def load_hitsmaps(pa: Dict):
    freqdset = pa['freqdset'] # NPIPE or DX12
    freqfilter = pa["freqfilter"]
    nside = pa["nside"]

    indir_path = cf[mch]['indir']

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
        FREQ: hp.read_map(mappath[FREQ], field=3)
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


@log_on_start(INFO, "Trying to load spectrum from {path_name}")
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


# %%
def save_map(data: Dict[str, Dict], path: str, filename: str = 'default'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"map/"+filename, data)


def save_weights(data: Dict[str, Dict], path: str, filename: str = 'default'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"weights/"+filename, data)


@log_on_start(INFO, "Saving spectrum to {path}{filename}")
@log_on_end(DEBUG, "Spectrum saved successfully to {filename}")
def save_spectrum(data: Dict[str, Dict], path: str, filename: str = 'default'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"spectrum/"+filename, data)


@log_on_start(INFO, "Saving mask to {path_name}")
@log_on_end(DEBUG, "Mask saved successfully to {path_name}")
def save_mask(data: Dict[str, Dict], path_name: str):
    if os.path.exists(path_name):
        os.remove(path_name)
    np.save(path_name, data)


def save_figure(mp, path_name: str, outdir_root: str = None, outdir_rel: str = None, out_desc: str = None, fname: str = None):
    if path_name == None:
        fending = ".jpg"
        path_name = outdir_root+outdir_rel+out_desc+fname+fending
    mp.savefig(path_name, dpi=144)
    plt.close()
