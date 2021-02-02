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


# %% Plot
def plotsave_powspec(df: Dict, specfilter: List[str], truthfile: str, plotsubtitle: str = 'default', plotfilename: str = 'default', outdir_root: str = '') -> None:
    """Plotting

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    spectrum_truth = pd.read_csv(
        truthfile,
        header=0,
        sep='    ',
        index_col=0)

    # plt.figure(figsize=(8,6))
    idx=1
    idx_max = len(PLANCKSPECTRUM) - len(specfilter)
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            plt.figure()
            df[spec].plot(
                figsize=(8,6),
                loglog=True,
                alpha=(idx_max-idx)/idx_max,
                xlim=(10,4000),
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
            plt.savefig('{}vis/spectrum/{}_spectrum--{}.jpg'.format(outdir_root, spec, plotfilename))
            plt.close()
    # %% Compare to truth
    # plt.figure(figsize=(8,6))


# %% Plot
def plotsave_powspec_binned(plt, data: Dict, cf: Dict, truthfile: str, spec: str, plotsubtitle: str = 'default', plotfilename: str = 'default', outdir_root: str = '', loglog: bool = True, color: List = None, alttext: str = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    koi = next(iter(data.keys()))
    specs = list(data[koi].keys())

    from scipy.signal import savgol_filter
    lmax = cf['pa']['lmax']
    if loglog:
        bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    else:
        bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    bl = bins[:-1]
    br = bins[1:]

    if loglog:
        spectrum_truth = pd.read_csv(
            truthfile,
            header=0,
            sep='    ',
            index_col=0)

    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        return mean, err

    flag = False
    if spec is None:
        flag = True
    else:
        specs = [spec]
    for spec in specs:
        if flag:
            plt.figure(figsize=(8,6))
        # fig = plt.figure(figsize=(8,6))
        plt.xscale("log", nonpositive='clip')
        if loglog:
            plt.yscale("log", nonpositive='clip')
        plt.xlabel("Multipole l")
        if alttext is None:
            plt.ylabel("Powerspectrum")
        else:
            plt.ylabel(alttext)
        plt.grid(which='both', axis='x')
        plt.grid(which='major', axis='y')
        idx=0
        idx_max = len(next(iter(data.keys())))
        if alttext is None:
            plt.title("{} spectrum - {}".format(spec, plotsubtitle))
        plt.xlim((10,4000))
        if loglog:
            plt.ylim((1e-3,1e5))
        else:
            plt.ylim((-0.5,0.5))
        for freqc, val in data.items():
            if "217" in freqc:
                binmean, binerr = std_dev_binned(data[freqc][spec])
                if color is None:
                    plt.errorbar(
                        0.5 * bl + 0.5 * br,
                        binmean,
                        yerr=binerr,
                        label=freqc,
                        capsize=2,
                        elinewidth=1,
                        fmt='x',
                        ms=4,
                        # errorevery=int(idx%2+1),
                        alpha=(4*idx_max-idx)/(2*idx_max)
                        )
                else:
                    plt.errorbar(
                        0.5 * bl + 0.5 * br,
                        binmean,
                        yerr=binerr,
                        label=freqc,
                        capsize=2,
                        elinewidth=1,
                        fmt='x',
                        ms=4,
                        barsabove=True,
                        # errorevery=int(idx%2+1),
                        alpha=(4*idx_max-idx)/(2*idx_max),
                        color=color[idx])
                idx+=1
        if loglog:
            if "Planck-"+spec in spectrum_truth.columns:
                plt.plot(spectrum_truth["Planck-"+spec], label = "Planck-"+spec)
        if alttext is None:
            plt.legend()
        plt.savefig('{}vis/spectrum/{}_spectrum/{}_binned--{}.jpg'.format(outdir_root, spec, spec, plotfilename))
        plt.close()

def plot_compare_powspec_binned(plt, data1: Dict, data2: Dict, cf: Dict, truthfile: str, spec: str, plotsubtitle: str = 'default', plotfilename: str = 'default', outdir_root: str = '', loglog: bool = True) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    from scipy.signal import savgol_filter
    lmax = cf['pa']['lmax']
    if loglog:
        bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    else:
        bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    bl = bins[:-1]
    br = bins[1:]

    if loglog:
        spectrum_truth = pd.read_csv(
            truthfile,
            header=0,
            sep='    ',
            index_col=0)

    def std_dev_binned(d):
        mean = [np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        err = [np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        return mean, err

    color = ['r', 'g', 'b', 'y']
    plt.xscale("log", nonpositive='clip')
    if loglog:
        plt.yscale("log", nonpositive='clip')
    # plt.xlabel("Multipole l")
    plt.ylabel("Powerspectrum")
    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    idx=0
    idx_max = len(next(iter(data2.keys())))
    plt.title("{} spectrum - {}".format(spec, plotsubtitle))
    plt.xlim((10,4000))
    if loglog:
        plt.ylim((1e-3,1e5))
    else:
        plt.ylim((-0.5,0.5))
    for freqc, val in data2.items():
        binmean1, binerr1 = std_dev_binned(data1[freqc][spec])
        plt.errorbar(
            0.5 * bl + 0.5 * br,
            binmean1,
            yerr=binerr1,
            label=freqc,
            capsize=3,
            elinewidth=2,
            fmt='none',
            color=color[idx],
            alpha=(idx_max-idx)/(2*idx_max))
        binmean2, binerr2 = std_dev_binned(data2[freqc][spec])
        plt.errorbar(
            0.5 * bl + 0.5 * br,
            binmean2,
            yerr=binerr2,
            label="syn "+ freqc,
            capsize=3,
            elinewidth=2,
            fmt='none',
            color=color[idx],
            alpha=(2*idx_max-idx)/(2*idx_max))
        idx+=1
    if loglog:
        if "Planck-"+spec in spectrum_truth.columns:
            plt.plot(spectrum_truth["Planck-"+spec], label = "Planck-"+spec)
    plt.legend()
    plt.savefig('{}vis/spectrum/{}_spectrum/{}_binned--{}.jpg'.format(outdir_root, spec, spec, plotfilename))
    plt.close()


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
            ylim=(-0.5,1.0),
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
                plt.ylim((-0.5,1.0))
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