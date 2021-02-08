"""
plot.py: Plotting functions

"""

import functools
import itertools
import json
import os
import os.path
import platform
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import healpy as hp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from component_separation.cs_util import Planckf, Planckr, Plancks
from logdecorator import log_on_end, log_on_error, log_on_start
from scipy import stats

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


def plot_beamwindowfunction(beamf, aa, bb, ab, field1, field2, p):
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
    return plt


# %% Plot
def plotsave_powspec(df: Dict, specfilter: List[str], truthfile: str, plotsubtitle: str = 'default', plotfilename: str = 'default') -> None:
    """Plotting

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    fending = '.jpg'
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
            plt.savefig('{}vis/spectrum/{}_spectrum--{}.jpg'.format(outdir_root, spec, plotfilename, fending))
            plt.close()
    # %% Compare to truth
    # plt.figure(figsize=(8,6))


# %% Plot
def plot_powspec_binned(data: Dict, lmax: Dict, title_string: str, truthfile = None, truth_label: str = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    # koi = next(iter(data.keys()))
    # specs = list(data[koi].keys())

    bins = np.logspace(np.log10(1), np.log10(lmax+1), 50)
    bl = bins[:-1]
    br = bins[1:]



    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        return mean, err

    
    plt.figure(figsize=(8,6))
    plt.xlabel("Multipole l")
    plt.ylabel("Powerspectrum")

    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    idx=0
    idx_max = len(next(iter(data.keys())))


    plt.title(title_string)
    plt.xlim((10,4000))
    # plt.ylim((1e-20,1e1))
    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    for freqc, val in data.items():
        idx_max+=len(freqc)
        if True: #"070" in freqc:
            binmean, binerr = std_dev_binned(data[freqc])
            binerr_low = np.array([binmean[n]*0.01 if binerr[n]>binmean[n] else binerr[n] for n in range(len(binerr))])
            plt.errorbar(
                0.5 * bl + 0.5 * br,
                binmean,
                yerr=(binerr_low, binerr),
                # 0.5 * bl + 0.5 * br,
                # binmean,
                # yerr=binerr,
                label=freqc,
                capsize=2,
                elinewidth=1,
                fmt='x',
                # ls='-',
                ms=4,
                alpha=(2*idx_max-idx)/(2*idx_max)
                )
            idx+=1
    if truthfile is not None:
        plt.plot(
            truthfile,
            label = truth_label,
            ls='-', marker='.',
            ms=0,
            lw=3)
    plt.legend()
    return plt


def plot_powspec_diff_binned(plt, data: Dict, lmax: int, plotsubtitle: str = 'default', plotfilename: str = 'default', color: List = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    bins = np.logspace(np.log10(1), np.log10(lmax+1), 50)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        return mean, err

    plt.xscale("log", nonpositive='clip')
    plt.yscale("linear")

    plt.xlabel("Multipole l")
    plt.ylabel('Rel. difference')

    plt.grid(which='both', axis='x')
    idx=0
    idx_max = len(next(iter(data.keys())))
    plt.xlim((10,4000))
    plt.ylim((-0.01,0.01))
    plt.grid(which='both', axis='y')

    for freqc, val in data.items():
        idx_max+=len(freqc)
        if True: #"070" in freqc:
            binmean, binerr = std_dev_binned(data[freqc])
            plt.errorbar(
                0.5 * bl + 0.5 * br,
                binmean,
                yerr=binerr,
                label=freqc,
                capsize=2,
                elinewidth=1,
                fmt='x',
                # ls='-',
                ms=4,
                alpha=0.9,
                color=color[idx]
                )
            idx+=1
    return plt


def plot_compare_powspec_binned(plt, data1: Dict, data2: Dict, lmax: int, title_string: str, truthfile: str, truth_label: str, color: List, plotsubtitle: str = 'default', plotfilename: str = 'default') -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    lmax = cf['pa']['lmax']
    bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = [np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        err = [np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        return mean, err

    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    # plt.xlabel("Multipole l")
    plt.ylabel("Powerspectrum")
    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    idx=0
    idx_max = len(next(iter(data2.keys())))
    plt.title(title_string)
    plt.xlim((10,4000))
    plt.ylim((1e-3,1e5))
    for freqc, val in data2.items():
        binmean1, binerr1 = std_dev_binned(data1[freqc])
        plt.errorbar(
            0.5 * bl + 0.5 * br,
            binmean1,
            yerr=binerr1,
            label=freqc,
            capsize=3,
            elinewidth=2,
            fmt='x',
            color=color[idx],
            alpha=0.9)
        binmean2, binerr2 = std_dev_binned(data2[freqc])
        plt.errorbar(
            0.5 * bl + 0.5 * br,
            binmean2,
            yerr=binerr2,
            label="syn "+ freqc,
            capsize=3,
            elinewidth=2,
            fmt='x',
            color=color[idx],
            alpha=0.3)
        idx+=1

    if truthfile is not None:
        plt.plot(truthfile, label = truth_label)
    plt.legend()
    return plt



def plot_weights_binned(weights: pd.DataFrame, lmax: int, title_string: str):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    bins = np.arange(0, lmax+1, lmax/20)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = [np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        err = [np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]
        return mean, err

    plt.figure(figsize=(8,6))
    plt.xlabel("Multipole l")
    plt.ylabel("Weighting")

    for name, data in weights.items():
        binmean, binerr = std_dev_binned(data)
        plt.errorbar(
            0.5 * bl + 0.5 * br,
            binmean,
            yerr=binerr,
            label=name,
            capsize=3,
            elinewidth=2
            )
        plt.title(title_string)
        plt.xlim((1,4000))
        plt.ylim((-0.2,1.0))
    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    plt.legend()
    return plt


# %% Plot weightings
def plot_map(data: Dict, title_string: str = ''):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    plt.figure()
    hp.mollview(data["map"]*data["mask"], title=title_string, norm="hist")
    return plt


# %% Plot weightings
def plotsave_weights(df: Dict, plotsubtitle: str = '', plotfilename: str = '', outdir_root: str = ''):
    fending = ".jpg"
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
        plt.savefig('{}vis/weighting/{}_weighting/{}_weighting--{}{}'.format(
            outdir_root,
            spec,
            spec,
            plotfilename,
            fending))
        plt.close()