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


PLANCKMAPFREQ = [p.value for p in list(Planckf)]
LOGFILE = 'messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)
    

def general_pipeline():

    freqfilter = [
        Planckf.LFI_1.value,
        Planckf.LFI_2.value,
        # Planckf.HFI_1.value,
        Planckf.HFI_2.value,
        Planckf.HFI_3.value,
        Planckf.HFI_4.value,
        Planckf.HFI_5.value,
        Planckf.HFI_6.value
        ]
    specfilter = [
        Plancks.TE.value,
        Plancks.TB.value,
        Plancks.ET.value,
        Plancks.BT.value,
        Plancks.TT.value,
        Plancks.BE.value,
        Plancks.EB.value
        ]
    lmax = 400
    lmax_mask = 800

    set_logger(DEBUG)
    
    path = 'data/'
    tqumap = io.get_data(path=path, freqfilter=freqfilter, nside=[1024,2048])
    logger.log(DEBUG, tqumap)
    tqumap_hpcorrected = pw.hphack(tqumap)
    spectrum = pw.powerspectrum(tqumap_hpcorrected, lmax, lmax_mask, freqfilter, specfilter)
    
    df = pw.create_df(spectrum, freqfilter, specfilter)
    df_sc = pw.apply_scale(df, specfilter, llp1=False)
    
    
    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
                        for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                        for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    print(freqcomb)
    
    beamf = io.get_beamf(path=path, freqcomb=freqcomb)

    df_scbf = pw.apply_beamfunction(df_sc, beamf, lmax, specfilter)
    pw.plot_powspec(df, specfilter, subtitle='unscaled, w/ beamfunction')
    pw.plot_powspec(df_scbf, specfilter, subtitle='scaled, w beamfunction')
    
    cov = pw.build_covmatrices(df_scbf, lmax, freqfilter, specfilter)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)
    pw.plotsave_weights(weights)

if __name__ == '__main__':
    general_pipeline()