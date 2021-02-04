"""
plot.py: Plotting functions

"""

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


def plotsave_beamwindowfunctions():
    import itertools
    import matplotlib.gridspec as gridspec
    TEB = {
            0: "T",
            1: "E",
            2: "B"
        }
    LFI_dict = {
            "030": 28,
            "044": 29,
            "070": 30
        }
    for p in itertools.product(freq, freq):
        if int(p[0])<int(p[1]):
            aa = "{}-{}".format(p[0], p[0])
            bb = "{}-{}".format(p[1], p[1])
            ab = "{}-{}".format(p[0], p[1])
            for field1 in [0,1,2]:
                for field2 in [0,1,2]:
                    if field2 >= field1:
                        plt.figure(figsize=(6.4,4.8))
                        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])                    
                        ax1 = plt.subplot(gs[0])
                        plt.ylabel('Windowfunction')
                        plt.yscale("log", nonpositive='clip')
                        plt.grid()
                        plt.xlim((0,4000))
                        plt.title(r"$W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{{{{b}}},{{{d}}}}} = B^{{\mathtt{{{a}}}}}_{{\mathtt{{{b}}}}}(l) B^{{\mathtt{{{c}}}}}_{{\mathtt{{{d}}}}}(l)$".format(
                            a=TEB[field1],
                            b="f_1",
                            c=TEB[field2],
                            d="f_2"))
                        l_00 = r"true $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[0])
                        l_11 = r"true $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[1],
                                c=TEB[field2],
                                d=p[1])
                        l_01 = r"true $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[1])
                        l_01e = r"estimated $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[1])

                        l_01d = r"($W(l)^{{\mathtt{{{{{a}}}{{{c}}}, estimate}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}} - W(l)^{{\mathtt{{{{{a}}}{{{c}}}, truth}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}) / W(l)^{{\mathtt{{{{{a}}}{{{c}}}, estimate}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[1])
                                    
                        if int(p[0])>=100 and int(p[1])>=100:
                            plt.ylim((1e-8,2))
                            ab2 = beamf[aa]["HFI"][1].data.field(field1) * beamf[bb]["HFI"][1].data.field(field2)                        
                            plt.plot(
                                beamf[aa]["HFI"][1].data.field(field1)*beamf[aa]["HFI"][1].data.field(field2),
                                label = l_00,
                                linewidth=1,
                                c='g')
                            plt.plot(
                                beamf[bb]["HFI"][1].data.field(field1)* beamf[bb]["HFI"][1].data.field(field2),
                                label = l_11,
                                linewidth=1,
                                c='r')
                            plt.plot(
                                beamf[ab]["HFI"][1].data.field(field1) * beamf[ab]["HFI"][1].data.field(field2),
                                label = l_01,
                                linewidth=3,
                                color= "#cc7000")
                            plt.plot(
                                ab2,
                                label = l_01e,
                                linewidth=3,
                                color = "#3169CD")
                            plt.legend()

                            ax2 = plt.subplot(gs[1])

                            l1, = plt.plot(
                                (ab2-beamf[ab]["HFI"][1].data.field(field1)*beamf[ab]["HFI"][1].data.field(field2))/ab2,
                                color = "#cc7000",
                                linewidth=2,
                                linestyle='-')
                            l2, = plt.plot(
                                (ab2-beamf[ab]["HFI"][1].data.field(field1)*beamf[ab]["HFI"][1].data.field(field2))/ab2,
                                "--",
                                color = "#3169CD",
                                linestyle=(2, (2, 2)),
                                linewidth=2)
                            plt.legend([(l1, l2)], [l_01d])
                            plt.ylim((1e-5,2e1))
                            plt.xlim((0,4000))

                        elif int(p[0]) < 100 and int(p[1]) < 100:
                            plt.ylim((1e-3,2))
                            ab2 = np.sqrt(beamf[aa]["LFI"][LFI_dict[p[0]]].data.field(0)) * np.sqrt(beamf[bb]["LFI"][LFI_dict[p[1]]].data.field(0))

                            plt.plot(
                                np.sqrt(beamf[aa]["LFI"][LFI_dict[p[0]]].data.field(0)) * np.sqrt(beamf[aa]["LFI"][LFI_dict[p[0]]].data.field(0)),
                                label = l_00,
                                linewidth=1,
                                c='g')
                            plt.plot(
                                np.sqrt(beamf[bb]["LFI"][LFI_dict[p[1]]].data.field(0)) * np.sqrt(beamf[bb]["LFI"][LFI_dict[p[1]]].data.field(0)),
                                label = l_11,
                                linewidth=1,
                                c='r')
                            plt.plot([0],[1], linewidth=0)
                            plt.plot(
                                ab2,
                                label = l_01e,
                                linewidth=3,
                                color = "#3169CD")
                            plt.legend()

                            ax2 = plt.subplot(gs[1])

                            l1, = plt.plot([0],[1], linewidth=0,
                                color = "#cc7000",
                                linestyle='-'
                            )
                            l2, = plt.plot([0],[1], "--", linewidth=0,
                                color = "#3169CD",
                                linestyle=(2, (2, 2)),
                            )
                            plt.legend(title= "No Data")
                            plt.ylim((1e-1,1e1))
                            plt.xlim((0,4000))

                        elif int(p[0]) < 100 and int(p[1]) >= 100:
                            plt.ylim((1e-8,2))
                            ab2 = np.sqrt(beamf[aa]["LFI"][LFI_dict[p[0]]].data.field(0)) * beamf[bb]["HFI"][1].data.field(field2)[:2049]

                            plt.plot(
                                np.sqrt(beamf[aa]["LFI"][LFI_dict[p[0]]].data.field(0)) * np.sqrt(beamf[aa]["LFI"][LFI_dict[p[0]]].data.field(0)),
                                label = l_00,
                                linewidth=1,
                                c='g')
                            plt.plot(
                                beamf[bb]["HFI"][1].data.field(field1) * beamf[bb]["HFI"][1].data.field(field2),
                                label = l_11,
                                linewidth=1,
                                c='r')
                            plt.plot([0],[1], linewidth=0)
                            plt.plot(
                                ab2,
                                label = l_01e,
                                linewidth=3,
                                color = "#3169CD")
                            plt.legend()

                            ax2 = plt.subplot(gs[1])

                            l1, = plt.plot([0],[1], linewidth=0,
                                color = "#cc7000",
                                linestyle='-'
                            )
                            l2, = plt.plot([0],[1], "--", linewidth=0,
                                color = "#3169CD",
                                linestyle=(2, (2, 2)),
                            )
                            plt.legend(title= "No Data")
                            plt.ylim((1e-1,2e1))
                            plt.xlim((0,4000))

                        
                        plt.yscale("log", nonpositive='clip')
                        plt.grid()
                        plt.xlabel("Multipole")
                        plt.ylabel('Rel. difference')
                        plt.savefig(
                            "vis/beamf/{a}{b}_beamf/{a}{b}_beamf_{c}.jpg".format(
                                a=TEB[field1],
                                b=TEB[field2],
                                c=ab),
                            dpi=144)

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

    lmax = cf['pa']['lmax']
    bins = np.logspace(np.log10(1), np.log10(lmax+1), 50)
    bl = bins[:-1]
    br = bins[1:]

    spectrum_truth = pd.read_csv(
        truthfile,
        header=0,
        sep='    ',
        index_col=0)

    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        return mean, err

    for spec in specs:
        plt.figure(figsize=(8,6))
        plt.xlabel("Multipole l")
        plt.ylabel("Powerspectrum")

        plt.grid(which='both', axis='x')
        plt.grid(which='major', axis='y')
        idx=0
        idx_max = len(next(iter(data.keys())))
   
    
        plt.title("{} spectrum - {}".format(spec, plotsubtitle))
        plt.xlim((10,4000))
        # plt.ylim((1e-20,1e1))
        plt.xscale("log", nonpositive='clip')
        plt.yscale("log", nonpositive='clip')
        for freqc, val in data.items():
            idx_max+=len(freqc)
            if True: #"070" in freqc:
                binmean, binerr = std_dev_binned(data[freqc][spec])
                binerr_low = np.array([0 if binerr[n]>binmean[n] else binerr[n] for n in range(len(binerr))])
                plt.errorbar(
                    # 0.5 * bl[binmean>1e-2] + 0.5 * br[binmean>1e-2],
                    # binmean[binmean>1e-2],
                    # yerr=(binerr_low[binmean>1e-2], binerr[binmean>1e-2]),
                    0.5 * bl + 0.5 * br,
                    binmean,
                    yerr=binerr,
                    label=freqc,
                    capsize=2,
                    elinewidth=1,
                    fmt='x',
                    # ls='-',
                    ms=4,
                    alpha=(2*idx_max-idx)/(2*idx_max)
                    )
                idx+=1
        if "Planck-"+spec in spectrum_truth.columns:
            plt.plot(
                spectrum_truth["Planck-"+spec],
                label = "Planck-"+spec,
                ls='-', marker='.',
                ms=0,
                lw=3)
        plt.legend()
        if "syn" in plotfilename:
            pf = plotfilename[4:]
        else:
            pf = plotfilename
        plt.savefig('{}vis/spectrum/{}_spectrum/{}_binned--{}.jpg'.format(outdir_root, spec, spec, pf))
        # plt.close()


def plotsave_powspec_diff_binned(plt, data: Dict, cf: Dict, truthfile: str, spec: str, plotsubtitle: str = 'default', plotfilename: str = 'default', outdir_root: str = '', loglog: bool = True, color: List = None, alttext: str = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    lmax = cf['pa']['lmax']
    bins = np.logspace(np.log10(1), np.log10(lmax+1), 50)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        return mean, err

    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')

    plt.xlabel("Multipole l")

    plt.ylabel(alttext)
    plt.grid(which='both', axis='x')
    idx=0
    idx_max = len(next(iter(data.keys())))
    plt.xlim((10,4000))
    ylim = (0.001,1e-1)
    # plt.ylim(ylim)
    # plt.yticks(np.arange(ylim[0],ylim[1], step=(-ylim[0]+ylim[1])/9))
    plt.grid(which='both', axis='y')

    for freqc, val in data.items():
        idx_max+=len(freqc)
        if True: #"070" in freqc:
            binmean, binerr = std_dev_binned(data[freqc][spec])
            plt.errorbar(
                # 0.5 * bl[binmean>1e-2] + 0.5 * br[binmean>1e-2],
                # binmean[binmean>1e-2],
                # yerr=(binerr_low[binmean>1e-2], binerr[binmean>1e-2]),
                0.5 * bl + 0.5 * br,
                binmean,
                yerr=binerr,
                label=freqc,
                capsize=2,
                elinewidth=1,
                fmt='x',
                # ls='-',
                ms=4,
                alpha=(2*idx_max-idx)/(2*idx_max),
                color=color[idx]
                )
            idx+=1

        plt.savefig('{}vis/spectrum/{}_spectrum/{}_binned--{}.jpg'.format(outdir_root, spec, spec, plotfilename))
        # plt.close()


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
    plt.ylim((1e-3,1e5))
    for freqc, val in data2.items():
        binmean1, binerr1 = std_dev_binned(data1[freqc][spec])
        plt.errorbar(
            0.5 * bl + 0.5 * br,
            binmean1,
            yerr=binerr1,
            label=freqc,
            capsize=3,
            elinewidth=2,
            fmt='x',
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
            fmt='x',
            color=color[idx],
            alpha=(2*idx_max-idx)/(2*idx_max))
        idx+=1
    if loglog:
        if "Planck-"+spec in spectrum_truth.columns:
            plt.plot(spectrum_truth["Planck-"+spec], label = "Planck-"+spec)
    plt.legend()
    plt.savefig('{}vis/spectrum/{}_spectrum/{}_binned--{}.jpg'.format(outdir_root, spec, spec, plotfilename))
    # plt.close()

