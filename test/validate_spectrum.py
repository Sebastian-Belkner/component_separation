"""
validate_spectrum.py: compare powerspectrum as being calculated by `powspec.py` to planck simulations for deviation

 - take calculated powerspectra, C_out
 - use C_out to generate set of syntethic maps using hp.synfast(), C_out_syn
 - compare C_out to C_out_syn


"""
import json
import logging
import logging.handlers
import os
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import healpy as hp

import component_separation
import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
import component_separation.preprocess as prep
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf
from component_separation.cs_util import Planckf, Planckr, Plancks
import healpy as hp

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
freqfilter = cf['pa']["freqfilter"]
specfilter = cf['pa']["specfilter"]
num_sim = cf['pa']["num_sim"]

def spec2synmap(C_ltot):
    return pw.create_synmap(C_ltot, cf, mch, csu.freqcomb, specfilter) 


def cl2synmap():
    start = 0
    C_ltot = io.load_data(path_name=io.spec_sc_path_name)
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    for i in range(start, num_sim):
        print("Starting simulation {} of {}.".format(i+1, num_sim))
        
        syn_map = spec2synmap(C_ltot)
        io.save_data(syn_map, io.synmap_sc_path_name+'_'+str(i))
    

def synmaps2average(fname):
    def _synpath_name(i):
        return io.synmap_sc_path_name+'_'+str(i)
    spectrum = {
        i: io.load_data(path_name=_synpath_name(i))
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


if __name__ == '__main__':
    print(60*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(60*"$")

    #TODO fix this pipeline


    #load calculated spectrum
    # io.load_data()

    #generate set of syn maps
    cl2synmap()

    #calculate syn spectrum
    # syn_spectrum = pw.tqupowerspec(syn_map, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # io.save_data(syn_spectrum, io.out_spec_pathspec_path, "syn/unscaled-"+str(i)+"_synspec-"+filename)
    # syn_spectrum_scaled = spec2specsc(syn_spectrum, freqcomb)
    # io.save_spectrum(syn_spectrum_scaled, spec_path, "syn/scaled-"+str(i)+"_synmap-"+filename)

    #calculate syn mean spectrum
    # syn_spectrum_avgsc = synmaps2average(filename)
    # io.save_spectrum(syn_spectrum_avgsc, spec_path, "syn/scaled-" + "synavg-"+ filename)

    #compare syn mean spectrum to calculated spectrum, (i.e. spectrum transferfunction?)