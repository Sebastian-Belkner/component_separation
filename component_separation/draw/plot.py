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
import matplotlib
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

import component_separation
compath = os.path.dirname(component_separation.__file__)
with open('{}/draw/draw.json'.format(compath), "r") as f:
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


def plot_diff_binned(data1, data2, lmax, title_string: str, ylim: tuple = (1e-3,1e6), truthfile = None, truth_label: str = None):

    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    # koi = next(iter(data.keys()))
    # specs = list(data[koi].keys())
    # bins = np.arange(0, lmax+1, lmax/200)
    bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    bl = bins[:-1]
    br = bins[1:]



    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array(np.sqrt([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]))/np.sqrt(100)
        return mean, err

    
    plt.figure(figsize=(8,6))
    plt.xlabel("Multipole l")
    plt.ylabel("Powerspectrum")

    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    idx=0


    plt.title(title_string)
    plt.xlim((10,4000))
    plt.ylim(0.001, 1)
    plt.xscale("log", nonpositive='clip')
    plt.yscale("linear", nonpositive='clip')



    binmean, binerr = std_dev_binned(data1)
    binerr_low = np.array([binmean[n]*0.01 if binerr[n]>binmean[n] else binerr[n] for n in range(len(binerr))])
    plt.errorbar(
        0.5 * bl + 0.5 * br,
        binmean,
        yerr=(binerr_low, binerr),
        # 0.5 * bl + 0.5 * br,
        # binmean,
        # yerr=binerr,
        label='optimal NPIPE',
        capsize=2,
        elinewidth=1,
        fmt='x',
        # ls='-',
        ms=4,
        alpha=0.5
        )


    binmean, binerr = std_dev_binned(data2)
    binerr_low = np.array([binmean[n]*0.01 if binerr[n]>binmean[n] else binerr[n] for n in range(len(binerr))])
    plt.errorbar(
        0.5 * bl + 0.5 * br,
        binmean,
        yerr=(binerr_low, binerr),
        # 0.5 * bl + 0.5 * br,
        # binmean,
        # yerr=binerr,
        label='optimal DX12',
        capsize=2,
        elinewidth=1,
        fmt='x',
        # ls='-',
        ms=4,
        alpha=0.5
        )
    


    binmean, binerr = std_dev_binned((data2-data1)/data2)
    binerr_low = np.array([binmean[n]*0.01 if binerr[n]>binmean[n] else binerr[n] for n in range(len(binerr))])
    plt.errorbar(
        0.5 * bl + 0.5 * br,
        binmean,
        yerr=binerr,#(binerr_low, binerr),
        # 0.5 * bl + 0.5 * br,
        # binmean,
        # yerr=binerr,
        label='difference',
        capsize=2,
        elinewidth=1,
        fmt='x',
        # ls='-',
        ms=4
        )

    plt.plot(
        0.5 * bl + 0.5 * br,
        binmean,
        color = 'green'
        )


    if truthfile is not None:
        plt.plot(
            truthfile,
            label = truth_label,
            ls='-', marker='.',
            ms=0,
            lw=3)
    plt.legend()
    return plt


def plot_powspec_binned(data: Dict, lmax: Dict, title_string: str, ylim: tuple = (1e-3,1e6), truthfile = None, truth_label: str = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    # koi = next(iter(data.keys()))
    # specs = list(data[koi].keys())

    bins = np.logspace(np.log10(1), np.log10(lmax+1), 250)
    bl = bins[:-1]
    br = bins[1:]



    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array(np.sqrt([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]))
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
    plt.ylim(ylim)
    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    for freqc, val in data.items():
        idx_max+=len(freqc)
        freqs = freqc.split("-")
        # if "353" in freqs:
        if freqs[0]  == freqs[1]:
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


def plot_powspec_binned_bokeh(data: Dict, lmax: Dict, title_string: str, truthfile = None, truth_label: str = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    # koi = next(iter(data.keys()))
    # specs = list(data[koi].keys())

    bins = np.logspace(np.log10(1), np.log10(lmax+1), 250)
    bl = bins[:-1]
    br = bins[1:]



    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array(np.sqrt([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]))
        return mean, err

    from bokeh.plotting import figure as bfigure
    plt = bfigure(title=title_string, y_axis_type="log", x_axis_type="log",
           x_range=(10,4000), y_range=(1e-3,1e6),
           background_fill_color="#fafafa")
    # plt.xlabel("Multipole l")
    # plt.ylabel("Powerspectrum")

    idx=0
    idx_max = 8
    from bokeh.palettes import inferno
    for freqc, val in data.items():
        col = inferno(idx_max)
        freqs = freqc.split("-")
        if freqs[0]  == freqs[1]:
            binmean, binerr = std_dev_binned(data[freqc])
            binerr_low = np.array([binmean[n]*0.01 if binerr[n]>binmean[n] else binerr[n] for n in range(len(binerr))])
            plt.line(
                0.5 * bl + 0.5 * br,
                np.nan_to_num(binmean),
                legend_label=freqc+ " Channel",
                # yerr=(binerr_low, binerr),
                # # 0.5 * bl + 0.5 * br,
                # # binmean,
                # # yerr=binerr,
                # label=freqc,
                # capsize=2,
                # elinewidth=1,
                # fmt='x',
                # # ls='-',
                # ms=4,
                # alpha=(2*idx_max-idx)/(2*idx_max)
                color = col[idx],
                line_width=3,
                muted_alpha=0.2)
            plt.multi_line(#[(bm, bm) for bm in np.nan_to_num(binmean)]
                [(bx, bx) for bx in 0.5 * bl + 0.5 * br],
                [(bm-br, bm+br) for bm, br in zip(np.nan_to_num(binmean), np.nan_to_num(binerr))],
                legend_label=freqc+ " Channel",
                # yerr=(binerr_low, binerr),
                # # 0.5 * bl + 0.5 * br,
                # # binmean,
                # # yerr=binerr,
                # label=freqc,
                # capsize=2,
                # elinewidth=1,
                # fmt='x',
                # # ls='-',
                # ms=4,
                # alpha=(2*idx_max-idx)/(2*idx_max)
                line_color = col[idx],
                line_width=2,
                muted_alpha=0.2)
            idx+=1
            plt.xaxis.axis_label = "Multipole l"
            plt.yaxis.axis_label = "Powerspectrum"
    
    plt.line(
        np.arange(0,4000,1),
        truthfile,
        color='red',
        line_width=4,
        legend_label="Best Planck EE",
        muted_alpha=0.2)
        # label = truth_label,
        # ls='-', marker='.',
        # ms=0,
        # lw=3)
    plt.legend.location = "top_left"
    plt.legend.click_policy="mute"
    return plt


def plot_powspec_diff_binned(plt, data: Dict, lmax: int, plotsubtitle: str = 'default', plotfilename: str = 'default', color: List = None) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    bins = np.logspace(np.log10(1), np.log10(lmax+1), 100)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        if type(d) != np.ndarray:
            val = np.nan_to_num(d.to_numpy())[:lmax]
        else:
            val = np.nan_to_num(d)[:lmax]
        n, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins)
        sy, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val)
        sy2, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val * val)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        return mean, std, _

    plt.xscale("log", nonpositive='clip')
    plt.yscale("linear")

    plt.xlabel("Multipole l")
    plt.ylabel('Rel. difference')

    plt.grid(which='both', axis='x')
    idx=0
    idx_max = len(next(iter(data.keys())))
    plt.xlim((10,4000))
    plt.ylim((0,0.7))
    plt.grid(which='both', axis='y')

    for freqc, val in data.items():
        # if "070" not in freqc and "030" not in freqc and "044" not in freqc:
        # if "070" in freqc or "030" in freqc or "044" in freqc:
            idx_max+=len(freqc)
            mean, std, _ = std_dev_binned(data[freqc])
            plt.errorbar(
                (_[1:] + _[:-1])/2,
                mean,
                yerr=std,
                label=freqc,
                capsize=2,
                elinewidth=1,
                fmt='x',
                # ls='-',
                ms=4,
                alpha=0.9,
                color=CB_color_cycle[idx]
                )
            idx+=1
    return plt


def plot_compare_powspec_binned(plt, data1: Dict, data2: Dict, lmax: int, title_string: str, truthfile: str, truth_label: str, plotfilename: str = 'default') -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    base = 2
    nbins=150
    bins = np.logspace(np.log(1)/np.log(base), np.log(lmax+1)/np.log(base), nbins, base=base)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        if type(d) != np.ndarray:
            val = np.nan_to_num(d.to_numpy())[:lmax]
        else:
            val = np.nan_to_num(d)[:lmax]
        n, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins)
        sy, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val)
        sy2, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val * val)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        return mean, std, _

    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    # plt.xlabel("Multipole l")
    plt.ylabel("Powerspectrum")
    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    idx=0
    plt.title(title_string)
    plt.xlim((10,4000))
    plt.ylim((1e-2,1e3))

    for freqc, val in data2.items():
        # if "070" not in freqc and "030" not in freqc and "044" not in freqc:
            mean, std, _ = std_dev_binned(data1[freqc])
            plt.errorbar(
                (_[1:] + _[:-1])/2,
                mean,
                yerr=std,
                label="Optimal NPIPE lens spectrum",# + freqc,
                capsize=3,
                elinewidth=2,
                fmt='x',
                color=CB_color_cycle[idx],
                alpha=0.9)
            mean, std, _ = std_dev_binned(data2[freqc])
            plt.errorbar(
                (_[1:] + _[:-1])/2,
                mean,
                yerr=std,
                label="Optimal NPIPE smica spectrum",# + freqc,
                capsize=3,
                elinewidth=2,
                fmt='x',
                color=CB_color_cycle[idx+1],
                alpha=0.6)
            idx+=1

    if truthfile is not None:
        plt.plot(truthfile, label = truth_label, color = 'black')
    plt.legend()
    return plt


def plot_compare_weights_binned(plt, data1: Dict, data2: Dict, lmax: int, title_string: str) -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    base = 2
    plt.xscale("log", base=base)
    lmax = 3000
    nbins=75
    bins = np.logspace(np.log(1)/np.log(base), np.log(lmax+1)/np.log(base), nbins, base=base)
    plt.ylabel("Weights")
    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')

    idx=0
    plt.title(title_string)
    plt.ylim((-0.1,1))
    plt.xlim((20, 4000))

    def std_dev_binned(d, bins):
        if type(d) == np.ndarray:
            val = d
        else:
            val = np.nan_to_num(d.to_numpy())
        n, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins)
        sy, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val)
        sy2, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val * val)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        return mean, std, _




    # plt.xlabel("Multipole l")
    

    for freqc, val in data1.items():
        if "070" not in freqc and "030" not in freqc and "044" not in freqc:
            mean, std, _ = std_dev_binned(data1[freqc], bins)
            base_line = plt.errorbar(
                (_[1:] + _[:-1])/2,
                mean,
                yerr=std,
                label=freqc,
                capsize=3,
                elinewidth=2,
                fmt='x',
                alpha=0.9,
                color = CB_color_cycle[idx])

            mean, std, _ = std_dev_binned(data2[idx+3], bins)
            if idx == 0:
                plt.plot(
                    (_[1:] + _[:-1])/2,
                    mean,
                    label="smica channels",
                    alpha=0.8,
                    color = 'black')
            else:
                plt.plot(
                    (_[1:] + _[:-1])/2,
                    mean,
                    alpha=0.8,
                    color = 'black')
            idx+=1


    return plt


def plot_weights_diff_binned(plt, data: Dict, lmax: int, plotsubtitle: str = 'default', plotfilename: str = 'default') -> None:
    """Plotting

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    base=2
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    # plt.gca().set_prop_cycle(None)
    plt.xscale("log", base=base)
    lmax = cf['pa']['lmax']
    nbins=75
    # bins = np.logspace(np.log2(1), np.log2(lmax+1), nbins)\
    base = 2
    bins = np.logspace(np.log(1)/np.log(base), np.log(lmax+1)/np.log(base), nbins, base=base)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        mean = np.array([np.mean(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))])
        err = np.array(np.sqrt([np.std(d[int(bl[idx]):int(br[idx])]) for idx in range(len(bl))]))/np.sqrt(2)
        return mean, err

    plt.yscale("linear")

    plt.xlabel("Multipole l")
    plt.ylabel('Rel. difference')

    plt.grid(which='both', axis='x')
    idx=0
    idx_max = len(next(iter(data.keys())))
    plt.xlim((100,4000))
    plt.ylim((-0.1,0.1))
    plt.grid(which='both', axis='y')

    for freqc, val in data.items():
        if "070" not in freqc and "030" not in freqc and "044" not in freqc:
        # if "070" in freqc or "030" in freqc or "044" in freqc:
            idx_max+=len(freqc)
            binmean, binerr = std_dev_binned(data[freqc])
            plt.plot(
                0.5 * bl + 0.5 * br,
                binmean,
                # yerr=binerr,
                label=freqc,
                # capsize=2,
                # elinewidth=1,
                # fmt='x',
                # ls='-',
                ms=4,
                alpha=0.9,
                color = CB_color_cycle[idx]
                )
            idx+=1
    return plt


def plot_weights_binned(plt, weights: pd.DataFrame, lmax: int, title_string: str, alpha=1.0, legendstr = "", ls='-'):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    import matplotlib.ticker as mticker
    nbins=250
    # bins = np.logspace(np.log2(1), np.log2(lmax+1), nbins)\
    base = 2
    bins = np.logspace(np.log(1)/np.log(base), np.log(lmax+1)/np.log(base), nbins, base=base)
    #np.arange(0, lmax+1, lmax/20)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        val = np.nan_to_num(d.to_numpy())
        n, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins)
        sy, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val)
        sy2, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val * val)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        return mean, std, _



    for name, data in weights.items():
        # binmean, binerr = std_dev_binned(data)
        # plt.errorbar(
        #     0.5 * bl + 0.5 * br,
        #     binmean,
        #     yerr=binerr,
        #     label=name,
        #     capsize=3,
        #     elinewidth=2
        #     )
        mean, std, _ = std_dev_binned(data)
        plt.errorbar(
            (_[1:] + _[:-1])/2,
            mean,
            yerr=std,
            label=name+legendstr,
            capsize=2,
            elinewidth=1,
            alpha=alpha,
            ls=ls
            )
        plt.title(title_string)
        plt.xlim((100,4000))
        plt.ylim((-1.0,1.5))

    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    plt.legend()
    plt.get
    return plt


def plot_noiselevel_binned(plt, noise: pd.DataFrame, lmax: int, title_string: str, alpha=1.0, legendstr = "", ls='-'):
    """Plotting
    Args:
        df (Dict): Data to be plotted
        plotsubtitle (str, optional): Add characters to the title. Defaults to 'default'.
        plotfilename (str, optional): Add characters to the filename. Defaults to 'default'
    """
    nbins=250
    # bins = np.logspace(np.log2(1), np.log2(lmax+1), nbins)\
    base = 2
    bins = np.logspace(np.log(1)/np.log(base), np.log(lmax+1)/np.log(base), nbins, base=base)
    #np.arange(0, lmax+1, lmax/20)
    bl = bins[:-1]
    br = bins[1:]

    def std_dev_binned(d):
        val = np.nan_to_num(d)
        n, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins)
        sy, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val)
        sy2, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val * val)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        return mean, std, _



       
    mean, std, _ = std_dev_binned(noise)
    plt.errorbar(
        (_[1:] + _[:-1])/2,
        mean,
        yerr=std,
        label="EE"+legendstr,
        capsize=2,
        elinewidth=1,
        alpha=alpha,
        ls=ls
        )
    plt.title(title_string)
    plt.xlim((100,4000))
    plt.ylim((-4.0,1.0))

    plt.grid(which='both', axis='x')
    plt.grid(which='major', axis='y')
    plt.legend()
    plt.get
    return plt


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