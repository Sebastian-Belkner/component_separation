"""
run.py: script for executing main functionality of component_separation.draw module

"""

__author__ = "S. Belkner"

import json
import logging
import itertools
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import numpy as np
import seaborn as sns
from component_separation.cs_util import Planckf, Plancks
from component_separation.draw import plot as cplt

with open('config.json', "r") as f:
    cf = json.load(f)

with open('component_separation/draw/draw.json', "r") as f:
    dcf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

mskset = dcf['pa']['mskset'] # smica or lens
freqdset = dcf['pa']['freqdset'] # DX12 or NERSC

lmax = dcf['pa']["lmax"]
lmax_mask = dcf['pa']["lmax_mask"]

freqfilter = dcf['pa']["freqfilter"]
specfilter = dcf['pa']["specfilter"]
split = "Full" if dcf['pa']["freqdatsplit"] == "" else dcf['pa']["freqdatsplit"]


def _reorder_spectrum_dict(spectrum):
    spec_data = dict()
    for f in spectrum.keys():
        for s in spectrum[f].keys():
            if s in spec_data:
                spec_data[s].update({
                    f: spectrum[f][s]})
            else:
                spec_data.update({s:{}})
                spec_data[s].update({
                    f: spectrum[f][s]
                })
    return spec_data


def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)


def plot_maps(fname):
    dc = dcf["plot"]["maps"]
    if dc["type"] == "syn":
        inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
        maps = io.load_synmap(path_name=inpath_name)
    else:
        maps = io.load_plamap(dcf["pa"])

    tqu = ["T", "Q", "U"]
    for data in maps:
        idx=0
        for freq, val in data.items():
            outpath_name = dc["outdir_root"]+dc["outdir_rel"]+tqu[idx]+freq+"-"+dc["out_desc"]+"-"+fname+".jpg"
            mp = cplt.plot_map(
                data = val,
                title_string = tqu[idx] +" @ "+freq+"GHz")
            io.save_figure(
                mp = mp,
                path_name = outpath_name)
            idx+=1


def plot_weights(fname):
    dc = dcf["plot"]["weights"]
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
        
    weights = io.load_weights(inpath_name, fname)

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            title_string = "{} weigthts - {}".format(spec, plotsubtitle)
            mp = cplt.plot_weights_binned(
                weights[spec],
                lmax = dcf['pa']['lmax'],
                title_string = title_string,
                )

            outpath_name = \
                dc["outdir_root"] + \
                dc["outdir_rel"] + \
                spec+"_weights/" + \
                spec+"_weights" + "-" + \
                dc["out_desc"] +"-" + \
                fname + ".jpg"        

            io.save_figure(
                mp = mp,
                path_name = outpath_name)


def plot_beamwindowfunctions():
    dc = dcf["plot"]["beamf"]
    FREQS = [FR for FR in PLANCKMAPFREQ if FR not in freqfilter]
    freqcomb =  ["{}-{}".format(p[0],p[1])
            for p in itertools.product(FREQS, FREQS)
            if (int(p[1])>=int(p[0]))]

    beamf = io.load_beamf(freqcomb=freqcomb)

    TEB = {
            0: "T",
            1: "E",
            2: "B"
    }
    for p in itertools.product(FREQS, FREQS):
        if int(p[0])<int(p[1]):
            aa = "{}-{}".format(p[0], p[0])
            bb = "{}-{}".format(p[1], p[1])
            ab = "{}-{}".format(p[0], p[1])
            for field1 in [0,1,2]:
                for field2 in [0,1,2]:
                    if field2 >= field1:
                        mp = cplt.plot_beamwindowfunction(
                            beamf,
                            aa = aa,
                            bb = bb,
                            ab = ab,
                            field1 = field1,
                            field2 = field2,
                            p = p)
                        outpath_name = \
                            dc["outdir_root"] + \
                            dc["outdir_rel"] + \
                            TEB[field1]+TEB[field2]+"_beamf/" + \
                            TEB[field1]+TEB[field2]+"_beamf-" + \
                            dc["out_desc"]+ ab + ".jpg"       
                        io.save_figure(
                            mp = mp,
                            path_name = outpath_name)


def plot_spectrum_bias(fname):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    dc = dcf["plot"]["spectrum_bias"]
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    inpath_name_syn = dc["indir_root_syn"]+dc["indir_rel_syn"]+dc["in_desc_syn"]+fname
    
    spectrum = io.load_spectrum(inpath_name, fname)
    syn_spectrum = io.load_spectrum(inpath_name_syn, fname)

    spectrum_re = _reorder_spectrum_dict(spectrum)
    syn_spectrum_re = _reorder_spectrum_dict(syn_spectrum)

    lmax = dcf['pa']['lmax']

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    diff_spectrum = dict()
    for specc, va in syn_spectrum_re.items():
        for freqc, val in syn_spectrum_re[specc].items():
            if specc not in diff_spectrum.keys():
                diff_spectrum.update({specc: {}})
            diff_spectrum[specc].update({freqc: 
                (syn_spectrum_re[specc][freqc]-spectrum_re[specc][freqc])/spectrum_re[specc][freqc]})
    spectrum_truth = io.load_truthspectrum()

    for specc, diff_data in diff_spectrum.items():
        color = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple']
        title_string = specc+"-spectrum - " + plotsubtitle
        if "Planck-"+specc in spectrum_truth.columns:
            spectrum_trth = spectrum_truth["Planck-"+specc]

        plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        ax1 = plt.subplot(gs[0])

        mp = cplt.plot_compare_powspec_binned(plt,
            spectrum_re[specc],
            syn_spectrum_re[specc],
            lmax,
            title_string = title_string,
            truthfile = spectrum_trth,
            truth_label = "Planck-"+specc,
            plotsubtitle = plotsubtitle,
            color = color)
        
        ax2 = mp.subplot(gs[1])
        mp = cplt.plot_powspec_diff_binned(
            mp,
            diff_spectrum[specc],
            lmax,
            color = color)
            
        outpath_name = \
            dc["outdir_root"] + \
            dc["outdir_rel"] + \
            specc+"_spectrum/" + \
            specc+"_spectrumbias" + "-" + \
            dc["out_desc"] + "-" + \
            fname + ".jpg"
        io.save_figure(
            mp = mp,
            path_name = outpath_name)


def plot_spectrum(fname):
    dc = dcf["plot"]["spectrum"]
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    spectrum = io.load_spectrum(inpath_name, fname)
    lmax = dcf['pa']['lmax']

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)
   
    spec_data = _reorder_spectrum_dict(spectrum)

    spectrum_truth = io.load_truthspectrum()
    for specc, data in spec_data.items():
        title_string = "{} spectrum - {}".format(specc, plotsubtitle)
        if "Planck-"+specc in spectrum_truth.columns:
            spectrum_trth = spectrum_truth["Planck-"+specc]
        else:
            spectrum_trth = None
        mp = cplt.plot_powspec_binned(
            data,
            lmax,
            title_string = title_string,
            truthfile = spectrum_trth,
            truth_label = "Planck-"+specc
            )
        outpath_name = \
            dc["outdir_root"] + \
            dc["outdir_rel"] + \
            specc+"_spectrum/" + \
            specc+"_spectrum" + "-" + \
            dc["out_desc"] + "-" + \
            fname + ".jpg"
        io.save_figure(
            mp = mp,
            path_name = outpath_name)    

def plot_weighted_spectrum(fname):
    def _weightspec(weights, spectrum):
        import copy
        retspec = copy.deepcopy(spectrum)
        for specc, data in spectrum.items():

            buff = np.zeros_like(spectrum[specc]["030-030"][:lmax])
            for freqc, val in data.items():
                freqs = freqc.split("-")
                if freqs[0] == freqs[1]:
                    # probably, i need to take the variance, not the mean value.. see wikipedia..
                    b = np.nan_to_num(weights[specc].to_numpy())
                    normaliser = np.sum(b, axis=1)
                    # buff += spectrum[specc][freqc][:lmax] * spectrum[specc][freqc][:lmax] * weights[specc]["channel @{}GHz".format(freqs[0])].to_numpy()[:lmax] * weights[specc]["channel @{}GHz".format(freqs[0])].to_numpy()[:lmax]/(normaliser * normaliser)
                    buff += (1)/(spectrum[specc][freqc][:lmax])
            retspec[specc].update({'optimal-optimal': 1/buff})
        return retspec

    dc = dcf["plot"]["weights"]
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    weights = io.load_weights(inpath_name, fname)

    dc = dcf["plot"]["spectrum"]
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    spectrum = io.load_spectrum(inpath_name, fname)
    lmax = dcf['pa']['lmax']
    spec_data = _reorder_spectrum_dict(spectrum)
    spectrum_truth = io.load_truthspectrum()
    spec_data_wweighted = _weightspec(weights, spec_data)

    for specc, data in spec_data_wweighted.items():
        title_string = "{} spectrum - {}".format(specc, "optimal spectrum included")
        if "Planck-"+specc in spectrum_truth.columns:
            spectrum_trth = spectrum_truth["Planck-"+specc]
        else:
            spectrum_trth = None

        mp = cplt.plot_powspec_binned(
            data,
            lmax,
            title_string = title_string,
            truthfile = spectrum_trth,
            truth_label = "Planck-"+specc
            )

        outpath_name = \
            dc["outdir_root"] + \
            dc["outdir_rel"] + \
            specc+"_spectrum/" + \
            specc+"_spectrum" + "-" + \
            "withoptimal" + "-" + \
            fname + ".jpg"
        io.save_figure(
            mp = mp,
            path_name = outpath_name)


if __name__ == '__main__':
    set_logger(DEBUG)
    fname = io.make_filenamestring(dcf)

    if dcf["plot"]["maps"]["do_plot"]:
        plot_maps(
            fname = fname
            )
    
    if dcf["plot"]["beamf"]["do_plot"]:
        print("plotting beam windowfunctions")
        plot_beamwindowfunctions()

    if dcf["plot"]["weights"]["do_plot"]:
        print("plotting weights")
        plot_weights(fname = fname)

    if dcf["plot"]["spectrum"]["do_plot"]:
        print("plotting spectrum")
        plot_spectrum(fname = fname)

    if dcf["plot"]["spectrum_bias"]["do_plot"]:
        print("plotting spectrum bias")
        plot_spectrum_bias(fname = fname)

    plot_weighted_spectrum(fname)