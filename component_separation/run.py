"""
run.py: runner script for calling component_separation

"""

__author__ = "S. Belkner"


import component_separation.MSC.MSC.pospace as ps
from component_separation.cs_util import Planckf, Plancks
import logging
import logging.handlers
from logging import DEBUG, ERROR, INFO, CRITICAL
import os, sys
import component_separation.powspec as pw
import component_separation.io as io
from typing import Dict, List, Optional, Tuple
import json
import platform

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
LOGFILE = 'messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

with open('config.json', "r") as f:
    cf = json.load(f)


def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)
    

def general_pipeline():
    mskset = cf['pa']['mskset'] # smica or lens
    freqdset = cf['pa']['freqdset'] # DX12 or NERSC
    set_logger(DEBUG)

    lmax = cf['pa']["lmax"]
    lmax_mask = cf['pa']["lmax_mask"]
    llp1 = cf['pa']["llp1"]
    bf = cf['pa']["bf"]


    spec_path = cf[mch]['outdir']
    indir_path = cf[mch]['indir']
    freqfilter = cf['pa']["freqfilter"]
    specfilter = cf['pa']["specfilter"]
    spec_filename = '{freqdset}_lmax_{lmax}-lmax_mask_{lmax_mask}-tmask_{tmask}-pmask_{pmask}-freqs_{freqs}_specs-{spec}.npy'.format(
        freqdset = freqdset,
        lmax = lmax,
        lmax_mask = lmax_mask,
        tmask = mskset,
        pmask = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]))
    spectrum = io.load_spectrum(spec_path, spec_filename)
    if spectrum is None:
        tqumap = io.get_data(cf, mch = mch)

        if tqumap[0] == None:
            tqumap = tqumap[1:]
        elif tqumap[1] == None:
            tqumap = [tqumap[0]]
        tqumap_hpcorrected = pw.hphack(tqumap)
        # tqumap_hpcorrected = tqumap
        if len(tqumap) == 3:
            spectrum = pw.tqupowerspec(tqumap_hpcorrected, lmax, lmax_mask, freqfilter, specfilter)
        elif len(tqumap) == 2:
            spectrum = pw.qupowerspec(tqumap_hpcorrected, lmax, lmax_mask, freqfilter, specfilter)
        elif len(tqumap) == 1:
            print("Only TT spectrum caluclation requested. This is currently not supported.")
        io.save_spectrum(spectrum, spec_path, spec_filename)


    df = pw.create_df(spectrum, freqfilter, specfilter)

    df_sc = pw.apply_scale(df, specfilter, llp1=llp1)
    
    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
                        for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                        for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    print(freqcomb)
    
    if bf:
        beamf = io.get_beamf(indir_path=indir_path, freqcomb=freqcomb)
        df_scbf = pw.apply_beamfunction(df_sc, beamf, lmax, specfilter) #df_sc #
    else:
        df_scbf = df_sc

    filetitle = '_{lmax}_{lmax_mask}_{tmask}_{pmask}_{freqs}_{spec}'.format(
        lmax = lmax,
        lmax_mask = lmax_mask,
        tmask = mskset,
        pmask = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]))
    subtitle = '{} masks'.format(mskset)
    io.plotsave_powspec(
        df_scbf,
        specfilter,
        subtitle=subtitle,
        filetitle=filetitle)
    # print(df_scbf)
    cov = pw.build_covmatrices(df_scbf, lmax, freqfilter, specfilter)
    # print(cov)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    # print(cov_inv_l)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)

    filetitle = '_{lmax}_{lmax_mask}_{tmask}_{pmask}_{freqs}_{spec}'.format(
        lmax = lmax,
        lmax_mask = lmax_mask,
        tmask = mskset,
        pmask = mskset,
        spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter])
        )
    io.plotsave_weights(
        weights,
        subtitle=subtitle,
        filetitle=filetitle)

if __name__ == '__main__':
    general_pipeline()