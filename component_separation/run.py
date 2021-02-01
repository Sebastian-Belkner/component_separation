"""
run.py: script for executing main functionality of component_separation

"""

__author__ = "S. Belkner"


# TODO add LFI beam window functions to calculation
# why is there no lines @ unscaled DX12?
# use, in addition to the current datasets, cross and diff datasets
# cleanup plotting configuration etc. (maybe have a plot.config?)
# analytic expression for weight estimates

import json
import logging
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from functools import reduce
import numpy as np
import seaborn as sns

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks

with open('config.json', "r") as f:
    cf = json.load(f)

LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]


mskset = cf['pa']['mskset'] # smica or lens
freqdset = cf['pa']['freqdset'] # DX12 or NERSC

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir']
indir_path = cf[mch]['indir']
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]


def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)


def spec2synmap(spectrum, freqcomb):
    return pw.create_synmap(spectrum, cf, mch, freqcomb, specfilter) 


def map2spec(maps, freqcomb):
    if maps[0] == None:
        maps = maps[1:]
    elif maps[1] == None:
        maps = [maps[0]]
    maps_hpcorrected = pw.hphack(maps)
    # tqumap_hpcorrected = tqumap
    if len(maps) == 3:
        spectrum = pw.tqupowerspec(maps_hpcorrected, lmax, lmax_mask, freqcomb, specfilter)
    elif len(maps) == 2:
        spectrum = pw.qupowerspec(maps_hpcorrected, lmax, lmax_mask, freqcomb, specfilter)
    elif len(maps) == 1:
        print("Only TT spectrum caluclation requested. This is currently not supported.")
    return spectrum


def spec2specsc(spectrum):
    # df = pw.create_df(spectrum, cf["pa"]["offdiag"], freqfilter, specfilter)
    spec_sc = pw.apply_scale(spectrum, llp1=llp1)
    if bf:
        beamf = io.load_beamf(freqcomb=freqcomb)
        spec_scbf = pw.apply_beamfunction(spec_sc, beamf, lmax, specfilter)
    else:
        spec_scbf = spec_sc
    return spec_scbf


def specsc2weights(spectrum, diag):
    cov = pw.build_covmatrices(spectrum, diag, lmax, freqfilter, specfilter)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)
    return weights


def syn_spectrum_average():
    # Load all syn spectra
    spectrum = {
        i: io.load_spectrum(spec_path, str(i)+'_SYNscaled'+filename)
        for i in range(num_sim)}

    # sum all syn spectra
    spectrum_avg = dict()
    for FREQC in freqcomb:
        for spec in speccomb:
            if FREQC in spectrum_avg.keys():
                pass
            else:
                spectrum_avg.update({FREQC: {}})
            if spec in spectrum_avg[FREQC].keys():
                pass
            else:
                spectrum_avg[FREQC].update({spec: []})
                    
            spectrum_avg[FREQC][spec] = np.array(list(reduce(lambda x, y: x+y, [
                spectrum[idx][FREQC][spec]
                    for idx, _ in spectrum.items()
                ])))
            spectrum_avg[FREQC][spec] /= num_sim
    return spectrum_avg
                

if __name__ == '__main__':
    set_logger(DEBUG)
    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

    filename = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        mskset = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
        split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

    if cf['pa']['new_spectrum']:
        spectrum = map2spec(io.load_tqumap(), freqcomb)
        io.save_spectrum(spectrum, spec_path, 'unscaled'+filename)
    else:
        spectrum = io.load_spectrum(spec_path, 'unscaled'+filename)
    if spectrum is None:
        spectrum = map2spec(io.load_tqumap(), freqcomb)
        io.save_spectrum(spectrum, spec_path, 'unscaled'+filename)

    # spectrum_scaled = spec2specsc(spectrum)
    # io.save_spectrum(spectrum_scaled, spec_path, 'scaled'+filename)

    # weights = specsc2weights(spectrum_scaled, cf["pa"]["offdiag"])
    # io.save_weights(weights, spec_path, 'weights'+filename)
    

    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)==int(FREQ))]

    if cf['pa']["run_sim"]:
        for i in range(num_sim):
            print("Starting simulation {} of {}.".format(i+1, num_sim))
            synmaps = spec2synmap(spectrum, freqcomb)
            io.save_map(synmaps, spec_path, str(i)+"_synmaps"+filename)

            syn_spectrum = map2spec(synmaps, freqcomb)
            io.save_spectrum(syn_spectrum, spec_path, str(i)+"_SYNunscaled"+filename)

            syn_spectrum_scaled = spec2specsc(syn_spectrum)
            io.save_spectrum(syn_spectrum_scaled, spec_path, str(i)+"_SYNscaled"+filename)
    
    syn_spectrum_scaled = syn_spectrum_average()
    io.save_spectrum(syn_spectrum_scaled, spec_path, "SYNscaled_average"+filename)

    # weights = specsc2weights(syn_spectrum_scaled, False)
    # io.save_weights(weights, spec_path, "SYNweights"+filename)