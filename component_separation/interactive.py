"""
interactive.py: runner script for calling component_separation using jupyter

"""

__author__ = "S. Belkner"

#%%
import component_separation.MSC.MSC.pospace as ps
from component_separation.cs_util import Planckf, Plancks
import logging
import numpy as np
import logging.handlers
from logging import DEBUG, ERROR, INFO, CRITICAL
import os, sys
import component_separation.powspec as pw
import component_separation.io as io
from typing import Dict, List, Optional, Tuple

PLANCKSPECTRUM = [p.value for p in list(Plancks)]
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
    
#%%
freqfilter = [
    Planckf.LFI_1.value,
    Planckf.LFI_2.value,
    Planckf.LFI_3.value,
    # Planckf.HFI_1.value,
    # Planckf.HFI_2.value,
    # Planckf.HFI_3.value,
    # Planckf.HFI_4.value,
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
bf=True

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

#%%
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

#%%
df = pw.create_df(spectrum, freqfilter, specfilter)
df_sc = pw.apply_scale(df, specfilter, llp1=llp1)
freqcomb =  ["{}-{}".format(FREQ,FREQ2)
                    for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                    for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]

#%%
if bf:
    beamf = io.get_beamf(path=path, freqcomb=freqcomb)
    df_scbf = pw.apply_beamfunction(df_sc, beamf, lmax, specfilter) #df_sc #
else:
    df_scbf = df_sc

#%%
filetitle = '_{lmax}_{lmax_mask}_{tmask}_{pmask}'.format(
    lmax = lmax,
    lmax_mask = lmax_mask,
    tmask = tmask_filename[::5],
    pmask = ','.join([pmsk[::5] for pmsk in pmask_filename]))
subtitle = 'scaled, {} l(l+1), {} beamfunction'.format('w' if llp1 else 'wout', 'w' if bf else 'wout')

#%%
pw.plotsave_powspec(
    df_scbf,
    specfilter,
    subtitle=subtitle,
    filetitle=filetitle)

#%%
def build_covmatrices(df: Dict, lmax: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, np.ndarray]:
    """Calculates the covariance matrices from the data

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict[str, np.ndarray]: The covariance matrices of Dimension [Nspec,Nspec,lmax]
    """
    NFREQUENCIES = len([FREQ for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter])
    cov = {spec: np.zeros(shape=(NFREQUENCIES, NFREQUENCIES, lmax+1))
                for spec in PLANCKSPECTRUM if spec not in specfilter}

    ifreq, ifreq2, ispec = -1, -1, 0
    for FREQ in PLANCKMAPFREQ:
        ifreq2 = -1
        if FREQ not in freqfilter:
            ifreq+=1
            for FREQ2 in PLANCKMAPFREQ:
                if FREQ2 not in freqfilter:
                    ifreq2+=1
                    if int(FREQ2) >= int(FREQ):
                        ispec=-1
                        print(ifreq, ifreq2)
                        for spec in PLANCKSPECTRUM:
                            if spec not in specfilter:
                                ispec+=1
                                cov[spec][ifreq][ifreq2] = df[spec][FREQ+'-'+FREQ2]
                                cov[spec][ifreq2][ifreq] = df[spec][FREQ+'-'+FREQ2]
    print('\n\nCovariance matrix EE:', cov['EE'].shape)
    # print(cov['EE'])
    print(cov['EE'][:,:,3])
    print(df['EE'].head())
    return cov
cov = build_covmatrices(df_scbf, lmax, freqfilter, specfilter)

#%%
cov_inv_l = pw.invert_covmatrices(cov, lmax, freqfilter, specfilter)

#%%
weights = pw.calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)


#%%
plt.plot(weights['EE'])

#%%
filetitle = '_{lmax}_{lmax_mask}_{tmask}_{pmask}_{freqs}'.format(
    lmax = lmax,
    lmax_mask = lmax_mask,
    tmask = tmask_filename[::5],
    pmask = ','.join([pmsk[::5] for pmsk in pmask_filename]),
    freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter])
    )
pw.plotsave_weights(
    weights,
    subtitle=subtitle,
    filetitle=filetitle)