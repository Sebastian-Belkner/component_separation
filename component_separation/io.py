"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"


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

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

with open('config.json', "r") as f:
    cf = json.load(f)


#%% Collect maps
@log_on_start(INFO, "Starting to load data")
@log_on_end(DEBUG, "Data loaded successfully: '{result}' ")
def load_tqumap() -> List[Dict]:
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
    indir_path = cf[mch]['indir']
    mskset = cf['pa']['mskset'] # smica or lens
    freqdset = cf['pa']['freqdset'] # NPIPE or DX12
    freqfilter = cf['pa']["freqfilter"]
    specfilter = cf['pa']["specfilter"]

    tmask_path = cf[mch][mskset]['tmask']["path"]
    pmask_path = cf[mch][mskset]['pmask']["path"]
    freq_path = cf[mch][freqdset]['path']

    tmask_filename = cf[mch][mskset]['tmask']['filename']
    pmask_filename = cf[mch][mskset]['pmask']['filename']
    freq_filename = cf[mch][freqdset]['filename']

    nside = cf['pa']["nside"]

    def _read(mask_path, mask_filename):
        return hp.read_map(
            '{path}{mask_path}{mask_filename}'
            .format(
                path = indir_path,
                mask_path = tmask_path,
                mask_filename = tmask_filename), dtype=np.bool)
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
                )
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    if tmask_filename is None:
        tmask = None
        tmask_d = None
    else:
        tmask = _read(tmask_path, tmask_filename)
        tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=nside[0])

    if pmask_filename is None:
        pmask = None
        pmask_d = None
    else:
        pmasks = [_read(pmask_path, a) for a in pmask_filename]
        pmask = functools.reduce(_multi, pmasks)
        pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=nside[0])


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
                "mask": tmask_d if int(FREQ)<100 else tmask
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
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    umap = {
        FREQ: {
            "map": hp.read_map(mappath[FREQ], field=2),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]


@log_on_start(INFO, "Trying to load spectrum {filename}")
@log_on_end(DEBUG, "{result} loaded")
def load_weights(path: str, filename: str = 'default.npy') -> Dict[str, Dict]:
    if os.path.isfile(path+"weights/"+filename):
        data = np.load(path+"weights/"+filename, allow_pickle=True)
        return data.item()
    else:
        print("no existing weights at {}".format(path+"weights/"+filename))
        return None


@log_on_start(INFO, "Trying to load spectrum {filename}")
@log_on_end(DEBUG, "{result} loaded")
def load_map(path: str, filename: str = 'default.npy') -> Dict[str, Dict]:
    if os.path.isfile(path+"map/"+filename):
        data = np.load(path+"map/"+filename, allow_pickle=True)
        print(data)
        return data
    else:
        print("no existing map at {}".format(path+"map/"+filename))
        return None


@log_on_start(INFO, "Trying to load spectrum {filename}")
@log_on_end(DEBUG, "{result} loaded")
def load_spectrum(path: str, filename: str = 'default.npy') -> Dict[str, Dict]:
    if os.path.isfile(path+"spectrum/"+filename):
        data = np.load(path+"spectrum/"+filename, allow_pickle=True)
        return data.item()
    else:
        print("no existing spectrum at {}".format(path+"spectrum/"+filename))
        return None
        

@log_on_start(INFO, "Starting to grab data from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully: '{result}' ")
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
                            bf_path = cf[mch]["beamf"]['path'],
                            bf_filename = cf[mch]["beamf"]['filename']
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
                            bf_path = cf[mch]["beamf"]['path'],
                            bf_filename = cf[mch]["beamf"]['filename']
                                .replace("{freq1}", freqs[1])
                                .replace("{freq2}", freqs[1])
                    ))
                }
            })
            beamf[freqc].update({
                "LFI": fits.open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/beamf/BeamWF_LFI/LFI_RIMO_R3.31.fits")
            })
        if int(freqs[0]) < 100 and int(freqs[1]) < 100:
             beamf.update({
                freqc: {
                    "LFI": fits.open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/beamf/BeamWF_LFI/LFI_RIMO_R3.31.fits")
                }})
    return beamf


# %% Plot weightings
def plotsave_weights(df: Dict, plotsubtitle: str = '', plotfilename: str = '', outdir_root: str = ''):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    plt.figure()
    for spec in df.keys():
        df[spec].columns[0:3]
        df[spec].plot(
            figsize=(8,6),
            ylabel='weigthing',
            # y = df[spec].columns.to_list()[0:3],
            # marker="x",
            # style= '--',
            grid=True,
            xlim=(0,4000),
            ylim=(-0.2,1.0),
            # logx=True,
            title="{} weighting - {}".format(spec, plotsubtitle))
        plt.savefig('{}vis/weighting/{}_weighting/{}_weighting--{}.jpg'.format(outdir_root, spec, spec, plotfilename))
        plt.close()


def plotsave_weights_binned(df: Dict, cf: Dict, specfilter: List[str], plotsubtitle: str = 'default', plotfilename: str = 'default', outdir_root: str = ''):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    lmax = cf['pa']['lmax']
    bins = np.arange(0, lmax+1, lmax/20)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = [np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        err = [np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        return mean, err

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            plt.figure(figsize=(8,6))
            plt.xlabel("Multipole l")
            plt.ylabel("Weighting")
            for name, data in df[spec].items():
                # plt.loglog(data, label=name)
                binmean, binerr = std_dev_binned(data)
                plt.errorbar(
                    0.5 * bl + 0.5 * br,
                    binmean,
                    yerr=binerr,
                    label=name,
                    capsize=3,
                    elinewidth=2
                    )
                plt.title("{} weighting - {}".format(spec, plotsubtitle))
                plt.xlim((1,4000))
                plt.ylim((-0.2,1.0))
            plt.grid(which='both', axis='x')
            plt.grid(which='major', axis='y')
            plt.legend()
            plt.savefig('{}vis/weighting/{}_weighting/{}_weighting_binned--{}.jpg'.format(outdir_root, spec, spec, plotfilename))
            plt.close()

# %% Plot weightings
def plotsave_map(data: Dict, plotsubtitle: str = '', plotfilename: str = '', outdir_root: str = ''):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    titstr = ["T", "Q", "U"]
    for idx, tqu in enumerate(data):
        for freq, val in tqu.items():
            plt.figure()
            hp.mollview(val["map"]*val["mask"], title=titstr[idx]+" @ "+freq+"GHz", norm="hist")
            plt.savefig('{}vis/map/{}_map--{}.jpg'.format(outdir_root, titstr[idx]+freq, plotfilename))
            plt.close()
# %%
def save_map(data: Dict[str, Dict], path: str, filename: str = 'default.npy'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"map/"+filename, data)


def save_weights(data: Dict[str, Dict], path: str, filename: str = 'default.npy'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"weights/"+filename, data)


@log_on_start(INFO, "Saving spectrum to {filename}")
@log_on_end(DEBUG, "Spectrum saved successfully to{filename}")
def save_spectrum(data: Dict[str, Dict], path: str, filename: str = 'default.npy'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"spectrum/"+filename, data)