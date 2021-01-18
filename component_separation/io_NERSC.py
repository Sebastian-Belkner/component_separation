"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"


from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
import os
import sys
import functools
import matplotlib.pyplot as plt
import os.path
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from component_separation.cs_util import Planckf, Plancks, Planckr
import healpy as hp
PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

#%% Collect maps
@log_on_start(INFO, "Starting to grab data without {freqfilter}")
@log_on_end(DEBUG, "Data without '{freqfilter}' loaded successfully: '{result}' ")
def get_data(path: str, freqfilter: List[str], tmask_filename: str, pmask_filename: List[str], nside: List[int] = PLANCKMAPNSIDE) -> List[Dict]:
    """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
    Map data in `PATH/map/`.
    Args:
        path (str): Relative path to root dir of the data. Must end with '/'
        freqfilter (List[str]): Frequency channels which are to be ignored
        tmask_filename (str): name of the mask for intensity maps
        pmask_filename (List[str]): list of names of the masks for polarisation maps. They will be reduced to one mask

    Returns:
        List[Dict]: Planck maps (data and masks) and some header information

    Doctest:
    >>> get_data(freqfilter=freqfilter) 
    NotSureWhatToExpect
    """
    
    print(PLANCKMAPFREQ)
    mappath = {
#         FREQ:'/project/projectdirs/planck/data/compsep/exchanges/dx12/maps/hfi/{freq}GHz_ful.all_ful.RD12_RC4.P.fits'
          FREQ: "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/npipe6v20{split}_{freq}_map.fits"
            .format(
                split = "A",
                freq = FREQ
                )
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    tmask = hp.read_map('{}'.format(tmask_filename), dtype=np.float64)
    tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=nside[0])

    def multi(a,b):
        return a*b
    pmasks = [hp.read_map('{}'.format(a), dtype=np.bool) for a in pmask_filename]
    pmask = functools.reduce(multi, pmasks)
    pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=nside[0])

    print(mappath)
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
        return None

@log_on_start(INFO, "Starting to grab data from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully: '{result}' ")
def get_beamf(path: str, freqcomb: List) -> Dict:
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
        beamf.update({fkey: fits.open("{}beamf/Bl_TEB_R3.01_fullsky_{}x{}.fits".format(path, *freqs))})
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
#                 y=["{}-{}".format(a, b) for a,b in [c.split('-') for c in df[spec].columns] if a==b],
                loglog=True,
                alpha=(idx_max-idx)/idx_max,
                xlim=(0,4000),
                ylim=(1e-3,1e5),
                ylabel="power spectrum",
                grid=True,
                title="{} spectrum - NPIPE - {}".format(spec, subtitle))
            if "Planck-"+spec in spectrum_truth.columns:
                spectrum_truth["Planck-"+spec].plot(
                    loglog=True,
                    grid=True,
                    ylabel="power spectrum",
                    legend=True
                    )
            plt.savefig('vis/spectrum/NPIPE_{}_spectrum--{}--{}.jpg'.format(spec, subtitle, filetitle))

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
            title="{} weighting - NPIPE - {}".format(spec, subtitle))
        plt.savefig('vis/weighting/NPIPE_{}_weighting--{}--{}.jpg'.format(spec, subtitle, filetitle))
# %%
