#!/usr/local/bin/python
"""
run_powerspectrum.py: script for executing main functionality of component_separation

"""

__author__ = "S. Belkner"


# TODO
# pospace: is the second mask added correctly?
# how does binning work? do i have to take 2l+1 into account as described in https://arxiv.org/pdf/0803.1814.pdf on page9?
# monopole and dipole regression on apodized galmask (available in healpy ?) (healpy.pixelfunc.remove_dipole)
# use jackknives to compute a noise estimate (half mission)

import json
import logging
import logging.handlers
import os
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks

with open('config.json', "r") as f:
    cf = json.load(f)

LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=0
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
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir_spectrum']
weight_path = cf[mch]['outdir_weight']
indir_path = cf[mch]['indir']

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]

def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)
    logging.StreamHandler(sys.stdout)


def spec2synmap(spectrum, freqcomb):
    return pw.create_synmap(spectrum, cf, mch, freqcomb, specfilter) 


def map2spec(maps, freqcomb):
    # tqumap_hpcorrected = tqumap
    if len(maps) == 3:
        spectrum = pw.tqupowerspec(maps, lmax, lmax_mask, freqcomb, specfilter)
    elif len(maps) == 2:
        spectrum = pw.qupowerspec(maps, lmax, lmax_mask, freqcomb, specfilter)
    elif len(maps) == 1:
        print("Only TT spectrum caluclation requested. This is currently not supported.")
    return spectrum


def specsc2weights(spectrum, Tscale):
    cov = pw.build_covmatrices(spectrum, lmax, freqfilter, specfilter, Tscale)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter, Tscale)
    return weights


def synmaps2average(fname):
    # Load all syn spectra
    def _synpath_name(i):
        return spec_path + 'syn/scaled-{}_synmap-'.format(str(i)) + filename
    spectrum = {
        i: io.load_spectrum(path_name=_synpath_name(i))
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


def spec_weight2weighted_spec(spectrum, weights):
    alms = pw.spec2alms(spectrum)
    alms_w = pw.alms2almsxweight(alms, weights)
    spec = pw.alms2cls(alms_w)
    return spec


def postprocess_spectrum(data, freqcomb):
    spec_sc = pw.apply_scale(data, llp1=llp1)
    if bf:
        beamf = io.load_beamf(freqcomb=freqcomb)
        spec_scbf = pw.apply_beamfunction(spec_sc, beamf, lmax, specfilter)
    else:
        spec_scbf = spec_sc
    return spec_scbf

def mapmask2maskedarray(data):
    pass

if __name__ == '__main__':
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(40*"$")

    set_logger(DEBUG)

    freqcomb =  [
        "{}-{}".format(FREQ,FREQ2)
            for FREQ in PLANCKMAPFREQ
            if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ
            if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

    filename = io.make_filenamestring(cf)

    if cf['pa']['new_spectrum']:
        data = io.load_plamap_new(cf['pa'], field=(0,1,2))
        data = preprocess_map(data)
        mask = io.load_mask()
        data = mapmask2maskedarray()

        spectrum = map2spec(data, freqcomb)
        io.save_data(spectrum, spec_path+'unscaled'+filename)
    else:
        path_name = spec_path + 'unscaled' + filename
        spectrum = io.load_spectrum(path_name=path_name)
    if spectrum is None:
        print("couldn't find spectrum with given specifications at {}. Exiting..".format(path_name))
        sys.exit()

    spectrum_scaled = postprocess_spectrum(spectrum, freqcomb)
    io.save_data(spectrum_scaled, spec_path+'scaled'+filename)

    weights = specsc2weights(spectrum_scaled, cf['pa']["Tscale"])
    io.save_data(weights, weight_path+cf['pa']["Tscale"]+filename)

    # weighted_spec = spec_weight2weighted_spec(spectrum, weights)
