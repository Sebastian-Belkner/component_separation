# pixel window function needs to be taken into account,  see Healpy.pixwin()


"""
The ``powspec`` module
======================
Helper functions to utilise MSC.pospace for calculating the powerspectra and calculate the weightings for the frequency channels of the
Planck satellite and for the powerspectra of interest

Doctest
---------
To use the following lines as a ``test``, execute,

``python3 -m doctest component_separation/powspec.py -v``

in your command line.

    General purpose doctest

    .. code-block:: python

        >>> tqumap = get_data('test/', freqfilter=freqfilter) # doctest: +SKIP 
        >>> spectrum = powerspectrum(tqumap, lmax, lmax_mask, freqfilter, specfilter)
        >>> df = create_df(spectrum, freqfilter, specfilter)
        >>> df_sc = apply_scale(df, specfilter)
        >>> df_scbf = apply_beamfunction(df_sc, lmax, specfilter)
        >>> plot_powspec(df, specfilter, subtitle='unscaled, w/ beamfunction')
        >>> plot_powspec(df_scbf, specfilter, subtitle='scaled, w beamfunction')
        >>> cov = build_covmatrices(df_scbf, lmax, freqfilter, specfilter)
        >>> cov_inv_l = invert_covmatrices(cov, lmax, freqfilter, specfilter)
        >>> weights = calculate_weights(cov_inv_l, lmax, freqfilter, specfilter)


"""

import logging
#%%
import os
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from logdecorator import log_on_end, log_on_error, log_on_start

import component_separation.MSC.MSC.pospace as ps
from component_separation.cs_util import Planckf, Plancks, Planckr

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]
PLANCKMAPAR = [p.value for p in list(Planckr)]
PLANCKMAPNSIDE = [1024, 2048]

""" Doctest:
The following constants be needed because functions are called with globals()
"""
path='test/data'
freqfilter = [Planckf.LFI_1.value, Planckf.LFI_2.value, Planckf.HFI_1.value, Planckf.HFI_5.value, Planckf.HFI_6.value]
specfilter = [Plancks.TE, Plancks.TB, Plancks.ET, Plancks.BT]
lmax = 20
lmax_mask = 80

def set_logger(loglevel=logging.INFO):
    logging.basicConfig(format='   %(levelname)s:      %(message)s', level=loglevel)


@log_on_start(INFO, "Starting to replace UNSEEN pixels for 100Ghz map")
@log_on_end(DEBUG, "UNSEEN pixels replaced successfully: '{result}' ")
def hphack(tqumap: List[Dict]) -> List[Dict]:
    """Replaces UNSEEN pixels in the polarisation maps (Q, U) with 0.0. This is a quickfix for healpy `_sphtools.pyx`,
    as it throws errors when processing maps with UNSEEN pixels. reason being, one statement is ambigious: `if np.array([True, True])`.

    Args:
        tqumap (Dict): Data as coming from `pw.get_data()`

    Returns:
        Dict: The corrected data
    """
    UNSEEN = -1.6375e30
    UNSEEN_tol = 1.e-7 * 1.6375e30
    def count_bad(m):
        i = 0
        nbad = 0
        size = m.size
        for i in range(m.size):
            if np.abs(m[i] - UNSEEN) < UNSEEN_tol:
                nbad += 1
        return nbad

    def mkmask(m):
        nbad = 0
        size = m.size
        i = 0
        # first, count number of bad pixels, to see if allocating a mask is needed
        nbad = count_bad(m)
        mask = np.ndarray(shape=(1,), dtype=np.int8)
        #cdef np.ndarray[double, ndim=1] m_
        if nbad == 0:
            return False
        else:
            mask = np.zeros(size, dtype = np.int8)
            #m_ = m
            for i in range(size):
                if np.abs(m[i] - UNSEEN) < UNSEEN_tol:
                    mask[i] = 1
        mask.dtype = bool
        return mask

    if '100' in tqumap[1].keys():
        maps = [tqumap[1]["100"]['map'], tqumap[2]["100"]['map']]
        maps_c = [np.ascontiguousarray(m, dtype=np.float64) for m in maps]
        masks = [np.array([False]) if count_bad(m) == 0 else mkmask(m) for m in maps_c]
        for idx, (m, mask) in enumerate(zip(maps_c, masks)):
            if mask.any():
                m[mask] = 0.0
            tqumap[idx+1]["100"]['map'] = m
    return tqumap

#%% Calculate spectrum
@log_on_start(INFO, "Starting to calculate powerspectra up to lmax={lmax} and lmax_mask={lmax_mask}")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def powerspectrum(tqumap: List[Dict[str, Dict]], lmax: int, lmax_mask: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, Dict]:
    """Calculate powerspectrum using MSC.pospace

    Args:
        tqumap List[Dict[str, Dict]]: Planck maps (data and masks) and some header information
        lmax (int): Maximum multipol of data to be considered
        lmax_mask (int): Maximum multipol of mask to be considered. Hint: take >3*lmax
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict[str, Dict]: Powerspectra as provided from MSC.pospace
    """

    spectrum = {
        FREQ+'-'+FREQ2: 
            {spec: ps.map2cls(
                    tqumap=[tqumap[0][FREQ]['map'], tqumap[1][FREQ]['map'], tqumap[2][FREQ]['map']],
                    tqumap2=[tqumap[0][FREQ2]['map'], tqumap[1][FREQ2]['map'], tqumap[2][FREQ2]['map']],
                    tmask=tqumap[0][FREQ]['mask'],
                    pmask=tqumap[1][FREQ]['mask'],
                    lmax=lmax,
                    lmax_mask=lmax_mask)[idx]
                for idx, spec in enumerate(PLANCKSPECTRUM) if spec not in specfilter}
            for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
            for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and int(FREQ2)>=int(FREQ)}
    return spectrum

#%% Create df for no apparent reason
@log_on_start(INFO, "Starting to create powerspectrum dataframe with {spectrum}")
@log_on_end(DEBUG, "Dataframe created successfully: '{result}' ")
def create_df(spectrum: Dict[str, Dict[str, List]], freqfilter: List[str], specfilter: List[str]) -> Dict:
    """For easier parallelisation, tracking, and plotting

    Args:
        spectrum (Dict[str, Dict[str, List]]): Powerspectra as provided from MSC.pospace
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict: A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
    """
    df = {
        spec: 
            pd.DataFrame(
                data={"{}-{}".format(FREQ,FREQ2): spectrum[FREQ+'-'+FREQ2][spec]
                    for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                    for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))
            }) 
        for spec in PLANCKSPECTRUM if spec not in specfilter
    }

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            for fkey, _ in df[spec].items():
                df[spec][fkey].index.name = 'multipole'
    
    return df

#%% Apply 1e12*l(l+1)/2pi
@log_on_start(INFO, "Starting to apply scaling onto data {df}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_scale(df: Dict, specfilter: List[str], llp1: bool = True) -> Dict:
    """Multiplies powerspectra by :math:`l(l+1)/(2\pi)1e12`

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        df (llp1, Optional): Set to False, if :math:`l(l+1)/(2\pi)` should not be applied. Default: True

    Returns:
        Dict: A "2D"-DataFrame of scaled powerspectra with spectrum and frequency-combinations in the columns
    """
    df_scaled = df.copy()
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            for fkey, fval in df_scaled[spec].items():
                if llp1:
                    lnorm = pd.Series(fval.index.to_numpy()*(fval.index.to_numpy()+1)/(2*np.pi))
                    df_scaled[spec][fkey] = df[spec][fkey].multiply(lnorm*1e12, axis='index')
                else:
                    df_scaled[spec][fkey] = df[spec][fkey]*1e12
    return df_scaled

#%% Apply Beamfunction
@log_on_start(INFO, "Starting to apply Beamfunnction on dataframe with {df}")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(df: Dict,  beamf: Dict, lmax: int, specfilter: List[str]) -> Dict:
    """divides the spectrum derivded from channel `ij` and provided via `df_(ij)`,
    by `beamf_i beamf_j` as described by the planck beamf .fits-file header.

    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns

        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        DataFrame: A "2D"-DataFrame of powerspectra including effect of Beam, with spectrum and frequency-combinations in the columns

    """
    df_bf = df.copy()
    TEB_dict = {
        "T": 0,
        "E": 1,
        "B": 2
    }

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            for fkey in df[spec]:
                hdul = beamf[fkey]
                df_bf[spec][fkey] = df[spec][fkey] \
                    .divide(hdul[1].data.field(TEB_dict[spec[0]])[:lmax+1], axis='index') \
                    .divide(hdul[1].data.field(TEB_dict[spec[1]])[:lmax+1], axis='index')
    return df_bf

#%% Build covariance matrices
@log_on_start(INFO, "Starting to build convariance matrices with {df}")
@log_on_end(DEBUG, "Covariance matrix built successfully: '{result}' ")
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
                        for spec in PLANCKSPECTRUM:
                            if spec not in specfilter:
                                ispec+=1
                                cov[spec][ifreq][ifreq2] = df[spec][FREQ+'-'+FREQ2]
                                cov[spec][ifreq2][ifreq] = df[spec][FREQ+'-'+FREQ2]
    print('\n\nCovariance matrix EE:', cov['EE'].shape)
    # print(cov['EE'])
    return cov

#%% slice along l (3rd axis) and invert
@log_on_start(INFO, "Starting to invert convariance matrix {cov}")
@log_on_end(DEBUG, "Inversion successful: '{result}' ")
def invert_covmatrices(cov: Dict[str, np.ndarray], lmax: int, freqfilter: List[str], specfilter: List[str]):
    """Inverts a covariance matrix

    Args:
        cov (Dict[str, np.ndarray]): The covariance matrices of Dimension [Nspec,Nspec,lmax]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
    """
    def is_invertible(a, l):
        truth = a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
        if not truth:
            print('{} not invertible: {}'.format(l, a) )
        return truth
    cov_inv_l = {
        spec: {
            l: np.linalg.inv(cov[spec][:,:,l]) if is_invertible(cov[spec][:,:,l], l) else None
                for l in range(lmax)
            }for spec in PLANCKSPECTRUM 
                if spec not in specfilter
        }
    # print(cov_inv_l)
    return cov_inv_l

# %% Calculate weightings and store in df
@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: Dict, lmax: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, DataFrame]:
    """Calculates weightings of the respective Frequency channels

    Args:
        cov (Dict): The inverted covariance matrices of Dimension [Nspec,Nspec,lmax]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict[str, DataFrame]: The weightings of the respective Frequency channels
    """

    # TODO add dummy value for spec: np.array(..) when cov is not invertible
    elaw = np.ones(len([dum for dum in PLANCKMAPFREQ if dum not in freqfilter]))
    weighting = {spec: np.array([(cov[spec][l] @ elaw) / (elaw.T @ cov[spec][l] @ elaw)
                     if cov[spec][l] is not None else np.array([0.0 for n in range(len(elaw))])
                        for l in range(lmax) if l in cov[spec].keys()])
                    for spec in PLANCKSPECTRUM if spec not in specfilter}
    
    # print(cov)
    weights = {spec:
                pd.DataFrame(
                    data=weighting[spec],
                    columns = ["channel @{}GHz".format(FREQ)
                                for FREQ in PLANCKMAPFREQ
                                if FREQ not in freqfilter])
                    for spec in PLANCKSPECTRUM
                    if spec not in specfilter}
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            weights[spec].index.name = 'multipole'

    # print(weights)
    return weights

if __name__ == '__main__':
    import doctest
    # doctest.run_docstring_examples(get_data, globals(), verbose=True)


def general_pipeline():
    pass

# %%
