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
                    .replace("{split}", cf['pa'][freqdatsplit] if "split" in cf[mch][freqdset] else "")
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


    ## Decide on the spectra requests which maps to load 
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


@log_on_start(INFO, "Saving spectrum to {path}+{filename}")
@log_on_end(DEBUG, "Spectrum saved successfully to {path}+{filename}")
def save_spectrum(data: Dict[str, Dict], path: str, filename: str = 'default.npy'):
    np.save(path+filename, data)


@log_on_start(INFO, "Trying to load spectrum from {path}+{filename}")
@log_on_end(DEBUG, "{result} loaded")
def load_spectrum(path: str, filename: str = 'default.npy') -> Dict[str, Dict]:
    if os.path.isfile(path+filename):
        data = np.load(path+filename, allow_pickle=True)
        return data.item()
    else:
        print("no existing spectrum at {}".format(path+filename))
        return None
        

@log_on_start(INFO, "Starting to grab data from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully: '{result}' ")
def get_beamf(cf: dict, mch: str, freqcomb: List) -> Dict:
    """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

    Args:
        cf (Dict): Configuration as coming from conf.json
        mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
        freqcomb (List): Frequency channels which are to be ignored

    Returns:
        Dict: Planck beamfunctions
    """

    indir_path = cf[mch]['indir']
    bf_path = cf[mch]["beamf"]['path']
    bf_filename = cf[mch]["beamf"]['filename']
    beamf = dict()
    for fkey in freqcomb:
        freqs = fkey.split('-')
        beamf.update({
            fkey: fits.open(
                "{indir_path}{bf_path}{bf_filename}"
                .format(
                    indir_path = indir_path,
                    bf_path = bf_path,
                    bf_filename = bf_filename
                        .replace("{freq1}", freqs[0])
                        .replace("{freq2}", freqs[1])
                ))
            })

    return beamf


# %% Plot
def plotsave_powspec(df: Dict, specfilter: List[str], plotsubtitle: str = 'default', plotfilename: str = 'default') -> None:
    """Plotting

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
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
                title="{} spectrum - {}".format(spec, plotsubtitle))
            if "Planck-"+spec in spectrum_truth.columns:
                spectrum_truth["Planck-"+spec].plot(
                    loglog=True,
                    grid=True,
                    ylabel="power spectrum",
                    legend=True
                    )
            plt.savefig('vis/spectrum/{}_spectrum--{}.jpg'.format(spec, plotfilename))

    # %% Compare to truth
    # plt.figure()


# %% Plot
def plotsave_powspec_binned(df: Dict, cf: Dict, specfilter: List[str], plotsubtitle: str = 'default', plotfilename: str = 'default') -> None:
    """Plotting

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    from scipy.signal import savgol_filter
    lmax = cf['pa']['lmax']
    bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    bl = bins[:-1]
    br = bins[1:]

    spectrum_truth = pd.read_csv(
        'data/powspecplanck.txt',
        header=0,
        sep='    ',
        index_col=0)

    def std_dev_binned(d):
        mean = [np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        err = [np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        return mean, err

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            idx=0
            idx_max = len(df[spec].columns)
            plt.figure()
            plt.xscale("log", nonpositive='clip')
            plt.yscale("log", nonpositive='clip')
            plt.xlabel("Multipole l")
            plt.ylabel("Powerspectrum")
            for name, data in df[spec].items():
                idx+=1
                # plt.loglog(data, label=name)
                binmean, binerr = std_dev_binned(data)
                plt.errorbar(
                    0.5 * bl + 0.5 * br,
                    binmean,
                    # savgol_filter(binmean, int((len(binmean))/4-1.), 5),
                    yerr=binerr,
                    label=name,
                    capsize=3,
                    elinewidth=2,
                    fmt='none',
                    alpha=(2*idx_max-idx)/(2*idx_max))
                plt.title("{} spectrum - {}".format(spec, plotsubtitle))
                plt.xlim((1,4000))
                plt.ylim((1e-3,1e5))
            if "Planck-"+spec in spectrum_truth.columns:
                plt.plot(spectrum_truth["Planck-"+spec], label = "Planck-"+spec)
            plt.grid(which='both', axis='x')
            plt.grid(which='major', axis='y')
            plt.legend()
            plt.savefig('vis/spectrum/{}_spectrum_binned--{}.jpg'.format(spec, plotfilename))


# %% Plot weightings
def plotsave_weights(df: Dict, plotsubtitle: str = '', plotfilename: str = ''):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
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
            title="{} weighting - {}".format(spec, plotsubtitle))
        plt.savefig('vis/weighting/{}_weighting--{}.jpg'.format(spec, plotfilename))


def plotsave_weights_binned(df: Dict, cf: Dict, specfilter: List[str], plotsubtitle: str = 'default', plotfilename: str = 'default'):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    from scipy.signal import savgol_filter
    lmax = cf['pa']['lmax']
    bins = np.arange(0, lmax+1, 100)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = [np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        err = [np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        return mean, err

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            plt.figure()
            plt.xlabel("Multipole l")
            plt.ylabel("Weighting")
            for name, data in df[spec].items():
                # plt.loglog(data, label=name)
                binmean, binerr = std_dev_binned(data)
                plt.errorbar(
                    0.5 * bl + 0.5 * br,
                    binmean,
                    # savgol_filter(binmean, int((len(binmean))/4-1.), 5),
                    yerr=binerr,
                    label=name,
                    capsize=3,
                    elinewidth=2
                    )
                plt.title("{} weighting - {}".format(spec, plotsubtitle))
                plt.xlim((0,4000))
                plt.ylim((-0.5,1.0))
            plt.grid(which='both', axis='x')
            plt.grid(which='major', axis='y')
            plt.legend()
            plt.savefig('vis/weighting/{}_weighting_binned--{}.jpg'.format(spec, plotfilename))
# %%
