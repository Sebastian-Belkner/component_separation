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
        # Planckf.HFI_2.value,
        # Planckf.HFI_3.value,
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
    lmax = 4000
    lmax_mask = 8000
    llp1=True   

    set_logger(DEBUG)
    
    path = 'data/'
    spec_path = 'data/tmp/spectrum/'
    
    tmask_filename = 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
    # HFI_Mask_GalPlane-apo0_128_R2.00.fits

    pmask_filename = ['psmaskP_2048.fits.gz', 'gmaskP_apodized_0_2048.fits.gz']
    # ['HFI_Mask_GalPlane-apo0_2048_R2.00.fits']
    # ['psmaskP_2048.fits.gz', 'gmaskP_apodized_0_2048.fits.gz']

    spec_filename = 'lmax_{lmax}-lmax_mask_{lmax_mask}-tmask_{tmask}-pmask_{pmask}-freqs_{freqs}_.npy'.format(
        lmax = lmax,
        lmax_mask = lmax_mask,
        tmask = tmask_filename[::5],
        pmask = ','.join([pmsk[::5] for pmsk in pmask_filename]),
        freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter])
    )

    spectrum = io.load_spectrum(spec_path, spec_filename)
    if spectrum is None:
        tqumap = io.get_data(
            path=path,
            freqfilter=freqfilter,
            tmask_filename=tmask_filename,
            pmask_filename=pmask_filename,
            nside=[1024,2048])
        logger.log(DEBUG, tqumap)
        tqumap_hpcorrected = pw.hphack(tqumap)
        spectrum = pw.powerspectrum(tqumap_hpcorrected, lmax, lmax_mask, freqfilter, specfilter)
        io.save_spectrum(spectrum, spec_path, spec_filename)
    df = pw.create_df(spectrum, freqfilter, specfilter)
    df_sc = pw.apply_scale(df, specfilter, llp1=llp1)
    
    
    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
                        for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                        for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
    print(freqcomb)
    
    beamf = io.get_beamf(path=path, freqcomb=freqcomb)

    df_scbf = pw.apply_beamfunction(df_sc, beamf, lmax, specfilter)
    # pw.plot_powspec(df, specfilter, subtitle='unscaled, w/ beamfunction')
    pw.plot_powspec(df_scbf, specfilter, subtitle='scaled, {} l(l+1), w beamfunction'.format('w' if llp1 else 'w/'))
    # print(df_scbf)
    cov = pw.build_covmatrices(df_scbf, lmax, freqfilter, specfilter)
    # print(cov)
    cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)
    # print(cov_inv_l)
    weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)
    pw.plotsave_weights(weights)

if __name__ == '__main__':
    general_pipeline()