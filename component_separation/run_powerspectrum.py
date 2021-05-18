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
from component_separation.cs_util import Config as csu
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import os

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Planckf, Plancks

import component_separation

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

LOGFILE = 'data/tmp/logging/messages.log'
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

num_sim = cf['pa']["num_sim"]
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


def map2spec(data, tmask, pmask, freqcomb):
    # tqumap_hpcorrected = tqumap
    # if len(data) == 3:
    #     spectrum = pw.tqupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # elif len(data) == 2:
    #     spectrum = pw.qupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # elif len(data) == 1:
    #     print("Only TT spectrum caluclation requested. This is currently not supported.")

    spectrum = pw.tqupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    return spectrum


def specsc2weights(spectrum, Tscale):
    cov = pw.build_covmatrices(spectrum, lmax, freqfilter, specfilter, Tscale)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter, Tscale)
    return weights


def synmaps2average(fname):
    # Load all syn spectra
    def _synpath_name(i):
        return io.spec_path + 'syn/scaled-{}_synmap-'.format(str(i)) + filename
    spectrum = {
        i: io.load_spectrum(path_name=_synpath_name(i))
        for i in range(num_sim)}

    # sum all syn spectra
    spectrum_avg = dict()
    for FREQC in csu.freqcomb:
        for spec in csu.speccomb:
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
    spec_sc = pw.apply_scale(data, scale=cf['pa']["Spectrum_scale"])
    beamf = io.load_beamf(freqcomb=freqcomb)
    spec_scbf = pw.apply_beamfunction(spec_sc, beamf, lmax, specfilter)
    return spec_scbf


if __name__ == '__main__':
    filename = io.make_filenamestring(cf)
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated filename for this session: {}".format(filename))
    print(40*"$")
    set_logger(DEBUG)

    if cf['pa']['new_spectrum']:
        data = io.load_plamap_new(cf, field=(0,1,2))
        data = prep.preprocess_all(data)
        tmask, pmask, pmask = io.load_one_mask_forallfreq(cf["pa"])

        spectrum = map2spec(data, tmask, pmask, csu.freqcomb)
        io.save_data(spectrum, io.spec_unsc_path_name)
    else:
        spectrum = io.load_data(path_name=io.spec_unsc_path_name)

    if spectrum is None:
        print("couldn't find spectrum with given specifications at {}. Exiting..".format(io.spec_unsc_path_name))
        sys.exit()

    spectrum_scaled = postprocess_spectrum(spectrum, csu.freqcomb)
    io.save_data(spectrum_scaled, io.spec_sc_path_name)

    weights = specsc2weights(spectrum_scaled, cf['pa']["Tscale"])
    io.save_data(weights, io.weight_path_name)

    # weighted_spec = spec_weight2weighted_spec(spectrum, weights)
