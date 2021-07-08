"""
run_powerspectrum.py: script for generating powerspectra from maps.
Depends on all maps being generated to begin with. Use ``run_map.py``, if e.g. noisemaps are missing.

"""

__author__ = "S. Belkner"

import copy
import json
import logging
import logging.handlers
import os
import platform
import sys
from functools import reduce
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import healpy as hp
import numpy as np
from healpy.sphtfunc import smoothing

import component_separation
import component_separation.io as io
import component_separation.powspec as pw
import component_separation.transform_map as trsf_m
import component_separation.transform_spec as trsf_s
from component_separation.cs_util import Config
from component_separation.cs_util import Helperfunctions as hpf
from component_separation.cs_util import Planckf, Plancks

with open(os.path.dirname(component_separation.__file__)+'/config_ps.json', "r") as f:
    cf = json.load(f)
csu = Config(cf)

# LOGFILE = 'data/tmp/logging/messages.log'
# logger = logging.getLogger("")
# handler = logging.handlers.RotatingFileHandler(
#         LOGFILE, maxBytes=(1048576*5), backupCount=0
# )
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

def set_logger(loglevel=logging.INFO):
    logger.setLevel(loglevel)
    logging.StreamHandler(sys.stdout)


@hpf.deprecated
def cmbmaps2C_lS():
    "Deprecated. Uses trivial weights for the combination of cmb maps"
    C_lS = dict()
    cmb_map = dict()
    for det in detectors:
        if int(det)<100:
            nside_desc = cf['pa']['nside_desc_map'][0]
        else:
            nside_desc = cf['pa']['nside_desc_map'][1]
        hdul = fits.open("/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20_sim/0200/input/ffp10_cmb_{det}_alm_mc_0200_nside{nside}_quickpol.fits".format(det=det, nside=nside_desc))
        cmb_map[det] = np.array([hp.ud_grade(hdul[1].data.field(spec).reshape(-1), nside_out = nside_out, order_in = 'NESTED', order_out='RING')
                            for spec in [0,1,2]])
    # C_lS = pw.tqupowerspec(cmb_map, tmask, pmask, lmax, lmax_mask)

    for det in detectors:
        alms = pw.map2alm_spin(cmb_map[det], pmask[det], 2, lmax) # full sky QU->EB
        # almT[det] = alms[0]
        almE[det] = alms[0]
        almB[det] = alms[1]

    CMB_in = dict()
    nalm = int((lmax+1)*(lmax+2)/2)
    signalW = np.ones(shape=(lmax+1,7))*1/len(detectors)
    beamf = io.load_beamf(csu.freqcomb)
    # combalmT = np.zeros((nalm), dtype=np.complex128)
    combalmE = np.zeros((nalm), dtype=np.complex128)
    combalmB = np.zeros((nalm), dtype=np.complex128)
    for m,det in zip(range(len(detectors)),detectors):
        # combalmT += hp.almxfl(almT[name], np.squeeze(W[0,m,:]))
        combalmE += hp.almxfl(hp.almxfl(almE[det],1/beamf[str(det)+'-'+str(det)]["HFI"][1].data.field(1)[0:lmax]), np.squeeze(signalW[:,m]))
        combalmB += hp.almxfl(hp.almxfl(almB[det],1/beamf[str(det)+'-'+str(det)]["HFI"][1].data.field(2)[0:lmax]), np.squeeze(signalW[:,m]))

    CMB_in["EE"] = hp.almxfl(combalmE, 1/hp.pixwin(nside_out, pol=True)[0][0:lmax])
    CMB_in["BB"] = hp.almxfl(combalmB, 1/hp.pixwin(nside_out, pol=True)[1][0:lmax])

    CMB_in["TQU"] = dict()
    CMB_in["TQU"] = hp.alm2map([np.zeros_like(CMB_in["EE"]), CMB_in["EE"], CMB_in["BB"]], nside_out)
    return CMB_in["TQU"]


def map2spec(data, tmask, pmask):
    # tqumap_hpcorrected = tqumap
    # if len(data) == 3:
    #     spectrum = pw.tqupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # elif len(data) == 2:
    #     spectrum = pw.qupowerspec(data, tmask, pmask, lmax, lmax_mask, freqcomb, specfilter)
    # elif len(data) == 1:
    #     print("Only TT spectrum calculation requested. This is currently not supported.")
    spectrum = pw.tqupowerspec(data, tmask, pmask, cf['pa']["lmax"], cf['pa']["lmax_mask"])
    return spectrum


if __name__ == '__main__':
    filename_raw = io.total_filename_raw
    filename = io.total_filename
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated raw filename(s) for this session: {}".format(filename_raw))
    print("Generated filename(s) for this session: {}".format(filename))
    print(filename)
    print(40*"$")
    # set_logger(DEBUG)
    run_map2spec = False
    run_alm2spec = True
    run_map2spec_alsowithnoise = True

    if run_map2spec:
        nside_out = cf['pa']['nside_out'] if cf['pa']['nside_out'] is not None else cf['pa']['nside_desc_map']
        if run_map2spec_alsowithnoise:
            cf_n = copy.deepcopy(cf)
            cf_n['pa']['freqdset'] = cf['pa']['freqdset']+"_diff"
            csu_n = Config(cf_n)
            cslist = [csu, csu_n]
            path_name_list = [(io.spec_unsc_path_name, io.spec_sc_path_name), (io.noise_unsc_path_name, io.noise_sc_path_name)]
        else:
            cslist = [csu]
            path_name_list = [(io.spec_unsc_path_name, io.spec_sc_path_name)]
        for cs, (path_unsc, path_sc) in zip(cslist, path_name_list):        
            beamf = io.load_beamf(cs.freqcomb)
            cf_loc = cs.cf
            maps = io.load_plamap(cf_loc, field=(0,1,2), nside_out=cf_loc['pa']["nside_out"])
            maps = trsf_m.process_all(maps)
            tmask, pmask, pmask = io.load_one_mask_forallfreq()
            C_l_unsc = map2spec(maps, tmask, pmask)
            io.save_data(C_l_unsc, path_unsc)

            C_l_unsc = io.load_data(path_unsc)
            C_l = trsf_s.process_all(C_l_unsc, cf_loc, cs.freqcomb, cs.PLANCKSPECTRUM, cf_loc['pa']['Tscale'], beamf, nside_out, cf_loc['pa']["Spectrum_scale"], cf_loc['pa']['smoothing_window'], cf_loc['pa']['max_polynom'])
            io.save_data(C_l, path_sc)

    if run_alm2spec:
        nside_out = cf['pa']['nside_out'] if cf['pa']['nside_out'] is not None else cf['pa']['nside_desc_map']
        idx = 200
        beamf = io.load_beamf(csu.freqcomb)
        #TODO adapt for any data, not hardcoded
        cmb_tlm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=1)
        cmb_elm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=2)
        cmb_blm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=3)
        buff = hp.alm2cl([cmb_tlm, cmb_elm, cmb_blm])[:,:cf['pa']['lmax']+1]
        C_lS_unsc = np.array([[buff]])
        #TODO to process spectrum, need to know the beamfunction. is it 5arcmin?
        # C_lS = trsf_s.process_all(C_lS_unsc, cf, cf['pa']['Tscale'], beamf, nside_out, cf['pa']["Spectrum_scale"], cf['pa']['smoothing_window'], cf['pa']['max_polynom'])
        C_lS = C_lS_unsc
        io.save_data(C_lS, "/global/cscratch1/sd/sebibel/misc/C_lS_in.npy")
        # spectrum = io.load_data(path_name=io.spec_unsc_path_name)
