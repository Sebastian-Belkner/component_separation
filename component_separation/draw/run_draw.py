"""
run.py: script for executing main functionality of component_separation.draw module

"""

__author__ = "S. Belkner"

import itertools
import json
import os
import platform
import copy
import sys

import component_separation
import component_separation.io as io
import component_separation.powspec as pw
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from component_separation.cs_util import Helperfunctions as hpf, Config as csu
from component_separation.cs_util import Planckf, Plancks
from component_separation.draw import plot as cplt

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

with open(os.path.dirname(component_separation.__file__)+'/draw/d_config.json', "r") as f:
    dcf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

mskset = dcf['pa']['mskset'] # smica or lens
freqdset = dcf['pa']['freqdset'] # DX12 or NERSC

lmax = dcf['pa']["lmax"]

freqfilter = dcf['pa']["freqfilter"]
specfilter = dcf['pa']["specfilter"]
split = "Full" if dcf['pa']["freqdatsplit"] == "" else dcf['pa']["freqdatsplit"]


def plot_beamwindowfunctions():
    dc = dcf["plot"]["beamf"]
    FREQS = [FR for FR in csu.PLANCKMAPFREQ if FR not in freqfilter]
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


def plot_spectrum(fname):
    
    if dcf['pa']["Spectrum_scale"] == "D_l":
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
    else:
       ylim = {
            "TT": (1e2, 1e5),
            "EE": (1e-6, 1e1),
            "BB": (1e-5, 1e5),
            "TE": (1e-2, 1e4),
            "TB": (-1e-3, 1e3),
            "EB": (-1e-3, 1e3),
            "ET": (1e-2, 1e5),
            "BT": (-1e-3, 1e3),
            "BE": (-1e-3, 1e3) 
       }
    
    dc = dcf["plot"]["spectrum"]
    lmax = dcf['pa']['lmax']

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)
    spec_data = io.load_data(io.spec_sc_path_name)
    # print(spec_data)
    spectrum_truth = io.load_truthspectrum()
    spec_data = hpf.reorder_spectrum_dict(spec_data)
    for specc, data in spec_data.items():
        # for freq in PLANCKMAPFREQ:
            title_string = "{} spectrum - {}".format(specc, plotsubtitle)
            if "Planck-"+specc in spectrum_truth.columns:
                spectrum_trth = spectrum_truth["Planck-"+specc]
                if dcf['pa']["Spectrum_scale"] == "D_l":
                    pass
                else:
                    spectrum_trth = spectrum_trth/hpf.llp1e12(np.array([l for l in range(len(spectrum_trth))]))*1e12
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
                dcf['pa']["Spectrum_scale"] + "-" + \
                fname + ".jpg"
            io.save_figure(
                mp = mp,
                path_name = outpath_name)
            print("spectrum saved to {}".format(outpath_name))  


def plot_spectrum_comparison(fname):
    """
    Creates a plot with two diagrams, comparing two sets of spectra with one another.
    This can be used to validate_spectra and validate_smica, and just compare two spectra, i.e. compare,
        1. synspectra to spectra
        2. smica spectra to input spectra
        3. two different spectra (e.g. varying in masks)
    """

    name1 = "SPEC"+io.make_filenamestring(dcf)
    path1 = io.out_spec_path
    pathname1 = path1 + name1

    dcf2 = copy.deepcopy(dcf)
    dcf2['pa']['smoothing_window'] = 0
    dcf2['pa']['max_polynom'] = 0

    name2 = "SPEC"+io.make_filenamestring(dcf2)
    path2 = io.out_spec_path
    pathname2 = path2 + name2

    spectrum1 = io.load_data(path_name=pathname1)
    spectrum2 = io.load_data(path_name=pathname2)


    spectrum1_re = hpf.reorder_spectrum_dict(spectrum1)
    spectrum2_re = hpf.reorder_spectrum_dict(spectrum2)

    lmax = dcf['pa']['lmax']

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])
    
    diff_spectrum = dict()
    for specc, va in spectrum1_re.items():
        for freqc, val in spectrum1_re[specc].items():
            if specc not in diff_spectrum.keys():
                diff_spectrum.update({specc: {}})
            diff_spectrum[specc].update({freqc: 
                (spectrum1_re[specc][freqc]-spectrum2_re[specc][freqc])/spectrum1_re[specc][freqc]})
    spectrum_truth = io.load_truthspectrum()
    

    for specc, diff_data in diff_spectrum.items():
        color = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple']
        title_string = specc+"-spectrum - " + plotsubtitle
        if "Planck-"+specc in spectrum_truth.columns:
            spectrum_trth = spectrum_truth["Planck-"+specc][:lmax+1]/(hpf.llp1e12(np.array([range(lmax+1)])))[0]*1e12

        plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        ax1 = plt.subplot(gs[0])

        mp = cplt.plot_compare_powspec_binned(plt,
            spectrum1_re[specc],
            spectrum2_re[specc],
            lmax,
            title_string = title_string,
            truthfile = spectrum_trth,
            truth_label = "Planck-"+specc)
        
        ax2 = mp.subplot(gs[1])
        mp = cplt.plot_powspec_diff_binned(
            mp,
            diff_spectrum[specc],
            lmax,
            color = color)
        outpath = "/global/cscratch1/sd/sebibel/vis/" + \
            specc+"_spectrum/"
        io.iff_make_dir(outpath)
        outpath_name = outpath + \
            specc+"_spectrumbias" + "-" + ".jpg"

        

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

    spec_data = hpf.reorder_spectrum_dict(spectrum)
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


def plot_weights(fname):
    dc = dcf["plot"]["weights"]
    total_filename = io.make_filenamestring(dcf)
    weight_path_name = io.weight_path_name
    weights = io.load_weights(weight_path_name, fname)

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


def plot_weights_comparison():
    ylim = {
        "TT": (1e2, 1e5),
        "EE": (1e-6, 1e1),
        "BB": (1e-5, 1e5),
        "TE": (1e-2, 1e4),
        "TB": (-1e-3, 1e3),
        "EB": (-1e-3, 1e3),
        "ET": (1e-2, 1e5),
        "BT": (-1e-3, 1e3),
        "BE": (-1e-3, 1e3) 
    }
    
    dc = dcf["plot"]["weights_comparison"]
    lmax = dcf['pa']['lmax']

    plotsubtitle = '{freqdset}"{split}" dataset - {mskset} masks'.format(
        mskset = mskset,
        freqdset = freqdset,
        split = split)

    cf['pa']['mskset'] = dc['mskset1']
    cf['pa']['freqdset'] = dc['freqdset1']
    cf['pa']['smoothing_window'] = dc['smoothing_window1']
    cf['pa']['max_polynom'] = dc['max_polynom1'] 
    filename1 = io.make_filenamestring(cf)
    weight_path1 = cf[mch]['abs_path'] + io.component_separation_path + cf[mch]['outdir_weight'] + dc["freqdset1"] + "/"
    weight_path_name1 = weight_path1 + "-" + cf['pa']["Tscale"] + "-" + filename1
    weight_data1 = io.load_data(weight_path_name1)

    cf['pa']['mskset'] = dc['mskset2']
    cf['pa']['freqdset'] = dc['freqdset2']
    cf['pa']['smoothing_window'] = dc['smoothing_window2']
    cf['pa']['max_polynom'] = dc['max_polynom2'] 
    filename2 = io.make_filenamestring(cf)
    weight_path2 = cf[mch]['abs_path'] + io.component_separation_path + cf[mch]['outdir_weight'] + dc["freqdset1"] + "/"
    weight_path_name2 = weight_path2 + "-" + cf['pa']["Tscale"] + "-" + filename2
    weight_data2 = io.load_data(weight_path_name2)

    print(weight_data2)
    
    diff_weights = np.nan_to_num(np.array([(weight_data1[idx] - weight_data2[idx] )/weight_data1[idx] for idx,specc in enumerate(weight_data1)]))

    for idx,spec in enumerate(PLANCKSPECTRUM):
        if spec not in specfilter:
            plt.figure(figsize=(8,6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
            ax1 = plt.subplot(gs[0])
          
            base=2
            plt.xlabel("Multipole l")
            plt.xscale("log", base=base)
            title_string = "{} weigthts - {}".format(spec, plotsubtitle)
            mp = cplt.plot_compare_weights_binned(plt,
                weight_data2[idx],
                weight_data1[idx],
                lmax,
                title_string = title_string)
            ax2 = mp.subplot(gs[1])
            mp = cplt.plot_weights_diff_binned(
                mp,
                diff_weights[idx],
                lmax)

            outpath_name = \
                dc["outdir_root"] + \
                dc["outdir_rel"] + \
                spec+"_weights/" + \
                spec+"_weights_comparison" + "-" + \
                dc["out_desc"] +"-" + \
                (filename1+filename2)[::3] + ".jpg"        
            print(spec + "processed " + outpath_name)
            io.save_figure(
                mp = mp,
                path_name = outpath_name)


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

    spec_data = hpf.reorder_spectrum_dict(spectrum)
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


if __name__ == '__main__':
    fname = io.make_filenamestring(dcf)
    
    if dcf["plot"]["beamf"]["do_plot"]:
        print("plotting beam windowfunctions")
        plot_beamwindowfunctions()

    if dcf["plot"]["maps"]["do_plot"]:
        print("plotting maps")
        plot_maps(fname = fname)

    # TODO adapt to latest structure
    if dcf["plot"]["spectrum"]["do_plot"]:
        print("plotting spectrum")
        plot_spectrum(fname = fname)

    # TODO adapt to latest structure
    if dcf["plot"]["spectrum_comparison"]["do_plot"]:
        print("plotting spectrum comparison")
        plot_spectrum_comparison(fname = fname)

    if dcf["plot"]["variance"]["do_plot"]:
        print("plotting variance")
        plot_variance()

    if dcf["plot"]["weights"]["do_plot"]:
        print("plotting weights")
        outpath_name = plot_weights(fname = fname)
        print("weights saved to {}".format(outpath_name))

    if dcf["plot"]["weights_comparison"]["do_plot"]:
        print("plotting comparison between 2 sets of weights")
        plot_weights_comparison()


    # if dcf["plot"]["S/N"]["do_plot"]:
    #     print("plotting S/N")
    #     plot_variance()

    # plot_compare_optimalspectrum(fname)
