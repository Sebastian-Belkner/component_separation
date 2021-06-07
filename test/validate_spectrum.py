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
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np

import component_separation
import component_separation.io as io
import component_separation.powspec as pw
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Constants as const
from component_separation.cs_util import Helperfunctions as hpf

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]

specfilter = cf['pa']["specfilter"]
num_sim = cf['pa']["num_sim"]


def spec2mapsyn(C_ltot):
    return pw.create_mapsyn(C_ltot, cf) 


def cl2mapsyn():
    start = 0
    C_ltot = io.load_data(path_name=io.spec_sc_path_name)
    for i in range(start, num_sim):
        if os.path.exists(io.mapsyn_sc_path_name+'_'+str(i)+'.npy'):
            print('mapsyn exists, skipping..')
            pass
        else:
            print("Starting simulation {} of {}.".format(i+1, num_sim))    
            map_syn = spec2mapsyn(C_ltot)
            io.save_data(map_syn, io.mapsyn_sc_path_name+'_'+str(i))


def mapsyn2clsyn():
    tmask, pmask, pmask = io.load_one_mask_forallfreq()
    for i in range(num_sim):
        if os.path.exists(io.specsyn_sc_path_name+'_'+str(i)+'.npy'):
            print('clsyn exists, skipping..')
            pass
        else:
            print("Calculating C_l for simulation {} of {}.".format(i+1, num_sim)) 
            map_syn = io.load_data(path_name=io.mapsyn_sc_path_name+'_'+str(i)+'.npy')
            C_ltot_syn = pw.tqupowerspec(map_syn, tmask, pmask, lmax, lmax_mask)
            io.save_data(C_ltot_syn, io.specsyn_sc_path_name+'_'+str(i))


if __name__ == '__main__':
    print(60*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print(60*"$")

    #load spectrum
    #generate set of syn maps
    cl2mapsyn()

    #calculate syn spectrum
    mapsyn2clsyn()

    #calculate syn mean spectrum
    C_ltot_syn_avg = io.load_data(io.specsyn_sc_path_name+'_'+str(0)+'.npy')
    for i in range(1, num_sim):
        C_ltot_syn_avg += io.load_data(io.specsyn_sc_path_name+'_'+str(i)+'.npy')
    C_ltot_syn_avg /= num_sim
    io.save_spectrum(C_ltot_syn_avg, io.specsyn_sc_path_name+'_{}mean'.format(num_sim))

    #compare syn mean spectrum to calculated spectrum, (i.e. spectrum transferfunction?)