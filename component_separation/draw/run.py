"""
run.py: script for executing main functionality of component_separation.draw module

"""

__author__ = "S. Belkner"

import json
import logging
import itertools
import matplotlib
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
from component_separation.cs_util import Planckf, Plancks
from component_separation.draw import plot as cplt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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

    for idx,spec in enumerate(PLANCKSPECTRUM):
        if spec not in specfilter:
            fig, ax = plt.subplots(figsize=(8,6))
            base=2
            plt.xlabel("Multipole l")
            plt.xscale("log", base=base)
            title_string = "{} weigthts - {}".format(spec, plotsubtitle)
            mp = cplt.plot_weights_binned(plt,
                weights[idx],
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
            print(spec + "processed " + outpath_name)
            io.save_figure(
                mp = mp,
                path_name = outpath_name)
    return outpath_name


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


def plot_weights_bias(fname):
    lmax = dcf['pa']['lmax']


    inpath_name_smica = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/ext/smica_propagation/weights_EB_smica_R3.00.txt"
    smica_weights = io.load_weights_smica(inpath_name_smica)[0,:,:lmax]
    import healpy as hp
    beam = [
        hp.gauss_beam(val, 2999, pol = True)[:,1]
        for val in [
            0.00930842,
            0.00785398,
            0.00378155,
            0.002807071,
            0.002106031,
            0.00145444,
            0.00140499,
        ]
    ]
    beam5 = hp.gauss_beam(0.00145444, 2999, pol = True)[:,1]

    for idx, weight in enumerate(smica_weights):
        # if idx>2:
            weight *= beam[idx]
            weight /= beam5


    dc = dcf["plot"]["weights_bias"]

    dcf["pa"]["mskset"] = "smica"
    dcf["pa"]["freqdset"] = "DX12"
    fname = io.make_filenamestring(dcf)
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    weights1 = io.load_weights(inpath_name, fname)

    dcf["pa"]["mskset"] = "lens"
    dcf["pa"]["freqdset"] = "NPIPE"
    fname = io.make_filenamestring(dcf)
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    weights2 = io.load_weights(inpath_name, fname)
    ret = [[np.nan_to_num(weights2[sp][freq].to_list()) for freq, val in weights2[sp].items()] for sp in ["TT", "EE", "BB", "TE"]]
    import pickle
    pickle.dump(ret, open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/tmp/weights/weightsK_CMBNPIPE-msk_lens-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE.pkl", 'wb'))
    print(len(weights2['EE']['channel @070GHz'].to_numpy()))
    np.save("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/tmp/weights/weightsK_CMBNPIPE-msk_lens-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE.npy", ret)
    sys.exit()
    # ["030", "044", "070", "100", "143", "217","353", "030", "044", "070", "100", "143", "217", "353"]
    
    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = "public smica weights vs NPIPE-lens",
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    diff_weights = dict()
    for specc, va in weights1.items():
        if specc == "EE":
            if specc not in diff_weights.keys():
                diff_weights.update({specc: {}})
            # diff_weights[specc] = (weights[specc] - smica_weights.T)/weights[specc]
                diff_weights[specc] = (weights1[specc] - weights2[specc] )/weights1[specc]

    for specc, diff_data in diff_weights.items():
        if specc == "EE":
            title_string = "Weights - " + plotsubtitle

            plt.figure(figsize=(8,6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
            ax1 = plt.subplot(gs[0])

            mp = cplt.plot_compare_weights_binned(plt,
                weights2[specc],
                smica_weights,
                # weights1[specc],
                # weights2[specc],
                lmax,
                title_string = title_string)
            
            # ax1.xaxis.set_major_locator(plt.MultipleLocator(np.log(2)))
            # ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
            plt.gca().set_xticks([25, 100, 400, 900, 1600, 2500, 3600])
            plt.gca().set_xticklabels(["25", "100", "400", "900", "1600", "2500", "3600"])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # plt.xlim((20,4000))

            ax2 = mp.subplot(gs[1])
            mp = cplt.plot_weights_diff_binned(
                mp,
                diff_weights[specc],
                lmax)

            plt.gca().set_xticks([25, 100, 400, 900, 1600, 2500, 3600])
            plt.gca().set_xticklabels(["25", "100", "400", "900", "1600", "2500", "3600"])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            plt.xlim((20,4000))
                
            outpath_name = \
                dc["outdir_root"] + \
                dc["outdir_rel"] + \
                specc+"_weights/" + \
                specc+"_weightsbias-smicapublic-smica" + "-" + \
                dc["out_desc"] + "-" + \
                fname + ".jpg"
            io.save_figure(
                mp = mp,
                path_name = outpath_name)


def plot_spectrum_new(fname):
    ylim = {
        "TT": (1e2, 1e5),
        "EE": (1e-3, 1e6),
        "BB": (1e-5, 1e5),
        "TE": (1e-2, 1e4),
        "TB": (-1e-3, 1e3),
        "EB": (-1e-3, 1e3),
        "ET": (1e-2, 1e5),
        "BT": (-1e-3, 1e3),
        "BE": (-1e-3, 1e3)
    }
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    
    dc = dcf["plot"]["spectrum"]
    def _inpathname(freqc,spec):
        return  "/global/u2/s/sebibel/tmp/spectrum/"+freqdset+"/"+spec+freqc+"-"+dc["in_desc"]+fname
    speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
    spec_data = {spec: {
        freqc: np.array(io.load_cl(_inpathname(freqc,spec)))
        for freqc in freqcomb} for 
        spec in speccs}
    lmax = dcf['pa']['lmax']

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)
    spectrum_truth = io.load_truthspectrum()
    for specc, data in spec_data.items():
        # for freq in PLANCKMAPFREQ:
            title_string = "{} spectrum - {}".format(specc, plotsubtitle)
            if "Planck-"+specc in spectrum_truth.columns:
                spectrum_trth = spectrum_truth["Planck-"+specc]
            else:
                spectrum_trth = None

            mp = cplt.plot_powspec_binned(
                data,
                lmax,
                title_string = title_string,
                ylim = ylim[specc],
                truthfile = spectrum_trth,
                truth_label = "Planck-"+specc,
                # filter = freq
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
            print("spectrum saved to {}".format(outpath_name))  


def plot_spectrum(fname):
    ylim = {
        "TT": (1e2, 1e5),
        "EE": (1e-3, 1e6),
        "BB": (1e-5, 1e5),
        "TE": (1e-2, 1e4),
        "TB": (-1e-3, 1e3),
        "EB": (-1e-3, 1e3),
        "ET": (1e-2, 1e5),
        "BT": (-1e-3, 1e3),
        "BE": (-1e-3, 1e3)
    }
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
            ylim = ylim[specc],
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
        print("spectrum saved to {}".format(outpath_name))


def plot_compare_optimalspectrum(fname):
    spec_pick = "EE"
    def _weightspec(icov_l, spectrum):
        import copy
        # retspec = copy.deepcopy(spectrum)
        for specc, data in spectrum.items():
            if specc == spec_pick:
                icovsum = np.array([np.sum(icov_l[specc][l]) for l in range(lmax)])
                    # buff += spectrum[specc][freqc][:lmax] * spectrum[specc][freqc][:lmax] * weights[specc]["channel @{}GHz".format(freqs[0])].to_numpy()[:lmax] * weights[specc]["channel @{}GHz".format(freqs[0])].to_numpy()[:lmax]/(normaliser * normaliser)
                retspec = {specc: {'optimal-optimal': np.array([1/icovsum_l if icovsum_l is not None else 0 for icovsum_l in icovsum])}}
        return retspec
    fname = io.make_filenamestring(dcf)
    lmax = dcf['pa']['lmax']
    freqcomb =  [
    "{}-{}".format(FREQ,FREQ2)
        for FREQ in PLANCKMAPFREQ
        if FREQ not in freqfilter
        for FREQ2 in PLANCKMAPFREQ
        if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    
    dc = dcf["plot"]["spectrum"]
    def _inpathname(freqc,spec):
        return  dc["indir_root"]+dc["indir_rel"]+spec+freqc+"-"+dc["in_desc"]+fname
    speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
    spectrum = {freqc: {
        spec: np.array(io.load_cl(_inpathname(freqc,spec)))
        for spec in speccs}  
        for freqc in freqcomb}

    cov = pw.build_covmatrices(spectrum, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    icov_l = pw.invert_covmatrices(cov, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)

    spec_data = _reorder_spectrum_dict(spectrum)
    spec_data_wweighted = _weightspec(icov_l, spec_data)

    spectrum_truth = io.load_truthspectrum(abspath='/mnt/c/Users/sebas/OneDrive/Desktop/Uni/')

    

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
    print("spectrum saved to {}".format(outpath_name))
    
        

def plot_noise_comparison():
    import matplotlib.ticker as mticker
    dc = dcf["plot"]["noise_comparison"]

    dcf["pa"]["mskset"] = "lens"
    dcf["pa"]["freqdset"] = "NPIPE"
    fname = io.make_filenamestring(dcf)
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    weights_NPIPE = io.load_weights(inpath_name, fname)

    dcf["pa"]["mskset"] = "smica"
    dcf["pa"]["freqdset"] = "DX12"
    fname = io.make_filenamestring(dcf)
    inpath_name = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
    weights_DX12 = io.load_weights(inpath_name, fname)

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)

    noiselevels = {
        "NPIPE":{
            "EE": {
                '030': 0.9*9*1e-3,
                '044': 0.9*9*1e-3,
                '070': 0.9*9*1e-3,
                '100': 0.9*2*1e-4,
                '143': 0.9*2*1e-4,
                '217': 0.9*3*1e-4,
                '353': 0.9*7*1e-3
            }
        },
        "DX12":{
            "EE": {
                '030': 9*1e-3,
                '044': 9*1e-3,
                '070': 9*1e-3,
                '100': 2*1e-4,
                '143': 2*1e-4,
                '217': 3*1e-4,
                '353': 7*1e-3
            }
        }
    }
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            if spec=="EE":
                # print(weights_NPIPE[spec])
                helper = np.array([np.nan_to_num(weights_NPIPE[spec]["channel @"+freq+"GHz"]) * noiselevels["NPIPE"][spec][freq]
                    for freq in PLANCKMAPFREQ
                    if freq not in freqfilter])
                noise_NPIPE = np.sum(np.abs(helper), axis=0)/np.sqrt(7)

                helper = np.array([np.nan_to_num(weights_DX12[spec]["channel @"+freq+"GHz"]) * noiselevels["DX12"][spec][freq]
                    for freq in PLANCKMAPFREQ
                    if freq not in freqfilter])
                noise_DX12 = np.sum(np.abs(helper), axis=0)/np.sqrt(7)


    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            if spec=="EE":
                title_string = "{} noise - {}".format(spec, plotsubtitle)
                fig, ax = plt.subplots(figsize=(8,6))
                base=2
                plt.xlabel("Multipole l")
                plt.ylabel("noiseleveldiff")
                plt.xscale("log", base=base)

                mp = cplt.plot_noiselevel_binned(plt,
                    noise_NPIPE,
                    lmax = dcf['pa']['lmax'],
                    title_string = title_string,
                    alpha=0.5,
                    legendstr = ' NPIPE',
                    ls="--"
                    )

                mp = cplt.plot_noiselevel_binned(plt,
                    noise_DX12,
                    lmax = dcf['pa']['lmax'],
                    title_string = title_string,
                    alpha=0.5,
                    legendstr = ' DX12',
                    ls="--"
                    )
                mp = cplt.plot_noiselevel_binned(plt,
                    (noise_NPIPE-noise_DX12)/noise_NPIPE,
                    lmax = dcf['pa']['lmax'],
                    title_string = title_string,
                    alpha=1.0,
                    legendstr = ' DX12npiperel',
                    ls="--"
                    )

                outpath_name = \
                    dc["outdir_root"] + \
                    spec+"_noisecomparison/" + \
                    spec+"_noisecomparison" + "-" + \
                    dc["out_desc"] +"-" + \
                    fname + ".jpg"   
                print(outpath_name)     

                io.save_figure(
                    mp = mp,
                    path_name = outpath_name)

            # fig, ax = plt.subplots(figsize=(8,6))
            # title_string = "{} weigthts - {}".format(spec, plotsubtitle)
            # mp = cplt.plot_weights_binned(plt,
            #     weights_NPIPE[spec],
            #     lmax = dcf['pa']['lmax'],
            #     title_string = title_string,
            #     alpha=0.5,
            #     legendstr = ' NPIPE',
            #     ls="--"
            #     )
            # plt.gca().set_prop_cycle(None)
            # mp = cplt.plot_weights_binned(plt,
            #     weights_DX12[spec],
            #     lmax = dcf['pa']['lmax'],
            #     title_string = title_string,
            #     alpha=0.9,
            #     legendstr = ' DX12',
            #     ls="-"
            #     )

            # ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
            # ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

            # outpath_name = \
            #     dc["outdir_root"] + \
            #     dc["outdir_rel"] + \
            #     spec+"_weightcomparison/" + \
            #     spec+"_weightcomparison" + "-" + \
            #     dc["out_desc"] +"-" + \
            #     fname + ".jpg"        

            # io.save_figure(
            #     mp = mp,
            #     path_name = outpath_name)


def plot_noise(fname):
    ylim = {
        "TT": (1e2, 1e5),
        "EE": (1e-3, 1e6),
        "BB": (1e-5, 1e5),
        "TE": (1e-2, 1e4),
        "TB": (-1e-3, 1e3),
        "EB": (-1e-3, 1e3),
        "ET": (1e-2, 1e5),
        "BT": (-1e-3, 1e3),
        "BE": (-1e-3, 1e3)
    }
    dc = dcf["plot"]["noise"]
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
        title_string = "{} Noise spectrum - {}".format(specc, plotsubtitle)
        if "Planck-"+specc in spectrum_truth.columns:
            spectrum_trth = spectrum_truth["Planck-"+specc]
        else:
            spectrum_trth = None

        mp = cplt.plot_powspec_binned(
            data,
            lmax,
            title_string = title_string,
            ylim = ylim[specc],
            truthfile = spectrum_trth,
            truth_label = "Planck-"+specc
            )
        outpath_name = \
            dc["outdir_root"] + \
            dc["outdir_rel"] + \
            specc+"_noisespectrum/" + \
            specc+"_noisespectrum" + "-" + \
            dc["out_desc"] + "-" + \
            fname + ".jpg"
        io.save_figure(
            mp = mp,
            path_name = outpath_name)    


def plot_spec_nonoise():
    dc = dcf["plot"]["spec_w/noise"]
    ylim = {
        "TT": (1e2, 1e5),
        "EE": (1e-3, 1e6),
        "BB": (1e-5, 1e5),
        "TE": (1e-2, 1e4),
        "TB": (-1e-3, 1e3),
        "EB": (-1e-3, 1e3),
        "ET": (1e-2, 1e5),
        "BT": (-1e-3, 1e3),
        "BE": (-1e-3, 1e3)
    }

    fname1 = io.make_filenamestring(dcf)
    inpath_name1 = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname1
    spectrum = io.load_spectrum(inpath_name1, fname)

    dcf['pa']['freqdset'] = 'DX12-diff'
    fname2 = io.make_filenamestring(dcf)
    inpath_name2 = dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname2
    noise = io.load_spectrum(inpath_name2, fname2)

    for freqc, val in spectrum.items():
        for specc, va in spectrum[freqc].items():
            if specc not in spectrum.keys():
                spectrum[freqc][specc] = spectrum[freqc][specc] - noise[freqc][specc]

    spec_data = _reorder_spectrum_dict(spectrum)
    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)

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
            ylim = ylim[specc],
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

def plot_variance():
    ylim = {
        "TT": (1e2, 1e5),
        "EE": (1e-10, 1e-2),
        "BB": (1e-5, 1e5),
        "TE": (1e-2, 1e4),
        "TB": (-1e-3, 1e3),
        "EB": (-1e-3, 1e3),
        "ET": (1e-2, 1e5),
        "BT": (-1e-3, 1e3),
        "BE": (-1e-3, 1e3)
    }
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    
    dc = dcf["plot"]["spectrum"]
    def _inpathname(freqc,spec):
        return  dc["indir_root"]+dc["indir_rel"]+spec+freqc+"-"+dc["in_desc"]+fname
    speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
    # spec_data = {spec: {
    #     freqc: np.array(io.load_cl(_inpathname(freqc,spec)))
    #     for freqc in freqcomb} for 
    #     spec in speccs}

    spec_pick = "EE"
    def _weightspec(icov_l, spectrum):
        import copy
        # retspec = copy.deepcopy(spectrum)
        for specc, data in spectrum.items():
            if specc == spec_pick:
                icovsum = np.array([np.sum(icov_l[specc][l]) for l in range(lmax)])
                    # buff += spectrum[specc][freqc][:lmax] * spectrum[specc][freqc][:lmax] * weights[specc]["channel @{}GHz".format(freqs[0])].to_numpy()[:lmax] * weights[specc]["channel @{}GHz".format(freqs[0])].to_numpy()[:lmax]/(normaliser * normaliser)
                retspec = {specc: {'optimal-optimal': np.array([1/icovsum_l if icovsum_l is not None else 0 for icovsum_l in icovsum])}}
        return retspec

    spectrum = {freqc: {
        spec: np.array(io.load_cl(_inpathname(freqc,spec)))
        for spec in speccs}  
        for freqc in freqcomb}
    lmax = dcf['pa']['lmax']
    npatch = 1
    ll = np.arange(0,lmax,1)
    # fsky = np.zeros((npatch, npatch), float)
    # np.fill_diagonal(fsky, 1/npatch*0.73)#*np.ones((npatch)))

    cov = pw.build_covmatrices(spectrum, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)
    icov_l = pw.invert_covmatrices(cov, lmax=lmax, freqfilter=freqfilter, specfilter=specfilter)

    spec_data = _reorder_spectrum_dict(spectrum)
    spec_data_wweighted = _weightspec(icov_l, spec_data)

    def dl2cl(dl):
        return dl/(ll*(ll+1))*2*np.pi

    def cl2var(cov_ltot_min, fsky):
        return 2 * cov_ltot_min * cov_ltot_min/((2*ll+1)*fsky)
    var_data = {spec: {
        freqc: dl2cl(spec_data_wweighted[spec][freqc])
            for freqc in ['optimal-optimal']} for 
        spec in [spec_pick]}
    var_data[spec_pick].update({'-2cl**2/nmode-': cl2var(dl2cl(spec_data_wweighted[spec_pick]['optimal-optimal']),0.96)})
    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)
   
    spectrum_truth = io.load_truthspectrum(abspath='/mnt/c/Users/sebas/OneDrive/Desktop/Uni/')
    for specc, data in var_data.items():
        title_string = "{} variance - {}".format(specc, plotsubtitle)
        if "Planck-"+specc in spectrum_truth.columns:
            pass
            spectrum_trth = spectrum_truth["Planck-"+specc]
        else:
            spectrum_trth = None

        mp = cplt.plot_variance_binned(
            data,
            lmax,
            title_string = title_string,
            ylim = ylim[specc],
            truthfile = dl2cl(spectrum_trth[:lmax]),
            truth_label = "Planck-"+specc
            )

        outpath_name = \
            dc["outdir_root"] + \
            dc["outdir_rel"] + \
            specc+"_spectrum/" + \
            specc+"_variance" + "-" + \
            dc["out_desc"] + "-" + \
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
        outpath_name = plot_weights(fname = fname)
        print("weights saved to {}".format(outpath_name))
    
    if dcf["plot"]["noise"]["do_plot"]:
        print("plotting noise")
        plot_noise(fname = fname)

    if dcf["plot"]["spectrum"]["do_plot"]:
        print("plotting spectrum")
        plot_spectrum_new(fname = fname)

    if dcf["plot"]["spectrum_bias"]["do_plot"]:
        print("plotting spectrum bias")
        plot_spectrum_bias(fname = fname)

    if dcf["plot"]["weights_bias"]["do_plot"]:
        print("plotting weights bias")
        plot_weights_bias(fname = fname)

    if dcf["plot"]["noise_comparison"]["do_plot"]:
        print("plotting noise_comparison")
        plot_noise_comparison()

    if dcf["plot"]["spec_w/noise"]["do_plot"]:
        print("plotting spectrum with noise subtracted")
        plot_spec_nonoise()


    if dcf["plot"]["variance"]["do_plot"]:
        print("plotting variance")
        plot_variance()

    # if dcf["plot"]["S/N"]["do_plot"]:
    #     print("plotting S/N")
    #     plot_variance()


    # plot_compare_optimalspectrum(fname)