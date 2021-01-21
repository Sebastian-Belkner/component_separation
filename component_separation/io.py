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
PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

#%% Collect maps
@log_on_start(INFO, "Starting to grab data with config {cf}")
@log_on_end(DEBUG, "Data loaded successfully: '{result}' ")
def get_data(cf: Dict, mch: str) -> List[Dict]:
    """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
    Map data in `PATH/map/`.
    Args:
        indir_path (str): path to root dir of the data. Must end with '/'
        freq_path (str): path to frequency maps
        freqfilter (List[str]): Frequency channels which are to be ignored
        tmask_filename (str): name of the mask for intensity maps
        pmask_filename (List[str]): list of names of the masks for polarisation maps. They will be reduced to one mask

    Returns:
        List[Dict]: Planck maps (data and masks) and some header information

    Doctest:
    >>> get_data(freqfilter=freqfilter) 
    NotSureWhatToExpect
    """
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
    ### build complete paths
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
        tmask = hp.read_map(
            '{path}{tmask_path}{tmask_filename}'
            .format(
                path = indir_path,
                tmask_path = tmask_path,
                tmask_filename = tmask_filename), field=2, dtype=np.float64)
        tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=nside[0])


    def multi(a,b):
        return a*b
    pmasks = [hp.read_map(
        '{path}{pmask_path}{pmask_filename}'
        .format(
            path = indir_path,
            pmask_path = pmask_path,
            pmask_filename = a), dtype=np.bool) for a in pmask_filename]
    pmask = functools.reduce(multi, pmasks)
    pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=nside[0])

    ret = []
    flag = False
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            if "T" in spec:
                flag = True
                break
    if flag and (tmask is not None):
        tmap = {
            FREQ: {
                "header": {
                    "nside" : nside[0] if int(FREQ)<100 else nside[1],
                    "freq" : FREQ,
                    "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
                },
                "map": hp.read_map(mappath[FREQ], field=0),
                "mask": tmask_d if int(FREQ)<100 else tmask
                }for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
        }
    elif flag and (tmask is None):
        print("Temperature spectra requested, but no Temperature masking provided. Spectra including temperature are skipped")
        tmap = None
    elif flag == False:
        tmap = None

    qmap = {
        FREQ: {
            "header": {
                "nside" : nside[0] if int(FREQ)<100 else nside[1],
                "freq" : FREQ,
                "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
            },
            "map": hp.read_map(mappath[FREQ], field=1),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    umap = {
        FREQ: {
            "header": {
                "nside" : nside[0] if int(FREQ)<100 else nside[1],
                "freq" : FREQ,
                "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
            },
            "map": hp.read_map(mappath[FREQ], field=2),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]

def save_spectrum(data: Dict[str, Dict], path: str, filename: str = 'default.npy'):
    np.save(path+filename, data)

def load_spectrum(path: str, filename: str = 'default.npy') -> Dict[str, Dict]:
    if os.path.isfile(path+filename):
        data = np.load(path+filename, allow_pickle=True)
        return data.item()
    else:
        print("no existing spectrum at {}".format(path+filename))
        return None

@log_on_start(INFO, "Starting to grab data from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully: '{result}' ")
def get_beamf(indir_path: str, freqcomb: List) -> Dict:
    """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

    Args:
        path (str): Relative path to root dir of the data. Must end with '/'
        freqcomb (List): Frequency channels which are to be ignored

    Returns:
        Dict: Planck beamfunctions
    """    
    beamf = dict()
    for fkey in freqcomb:
        freqs = fkey.split('-')
        beamf.update({
            fkey: fits.open(
                "{path}beamf/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_{freq1}x{freq2}.fits"
                .format(
                    path = indir_path,
                    freq1 = freqs[0],
                    freq2 = freqs[1])
                    )
            })
    return beamf


# %% Plot
def plotsave_powspec(df: Dict, specfilter: List[str], subtitle: str = '', filetitle: str = '') -> None:
    """Plotting

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns

        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
        subtitle (String, optional): Add some characters to the title. Defaults to ''.
    """

    spectrum_truth = pd.read_csv(
        'data/powspecplanck.txt',
        header=0,
        sep='    ',
        index_col=0)

    # plt.figure()
    idx=1
    idx_max = len(PLANCKSPECTRUM) - len(specfilter)
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            plt.figure()
            df[spec].plot(
                loglog=True,
                alpha=(idx_max-idx)/idx_max,
                xlim=(0,4000),
                ylim=(1e-3,1e5),
                ylabel="power spectrum",
                grid=True,
                title="{} spectrum - DX12 - {}".format(spec, subtitle))
            if "Planck-"+spec in spectrum_truth.columns:
                spectrum_truth["Planck-"+spec].plot(
                    loglog=True,
                    grid=True,
                    ylabel="power spectrum",
                    legend=True
                    )
            plt.savefig('vis/spectrum/DX12_{}_spectrum--{}--{}.jpg'.format(spec, subtitle, filetitle))

    # %% Compare to truth
    # plt.figure()


# %% Plot weightings
def plotsave_weights(df: Dict, subtitle: str = '', filetitle: str = ''):
    """Plotting
    Args:
        df (Dict): Data to be plotted
    """
    plt.figure()
    for spec in df.keys():
        df[spec].plot(
            ylabel='weigthing',
            # marker="x",
            # style= '--',
            grid=True,
            xlim=(0,4000),
            ylim=(-0.5,1.0),
            # logx=True,
            title="{} weighting - DX12 - {}".format(spec, subtitle))
        plt.savefig('vis/weighting/DX12_{}_weighting--{}--{}.jpg'.format(spec, subtitle, filetitle))
# %%
