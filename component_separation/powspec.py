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
import functools
import healpy as hp
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

def _crosscomb(option, f1, f2):
        if option:
            return int(f1) <= int(f2)
        else:
            return int(f1) == int(f2)

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
    
    if '100' in tqumap[0].keys():
        # only fixing q and u maps
        if len(tqumap)==2:
            maps = [tqumap[0]["100"]['map'], tqumap[1]["100"]['map']]
        else:
            maps = [tqumap[1]["100"]['map'], tqumap[2]["100"]['map']]
        maps_c = [np.ascontiguousarray(m, dtype=np.float64) for m in maps]

        masks = [np.array([False]) if count_bad(m) == 0 else mkmask(m) for m in maps_c]
        for idx, (m, mask) in enumerate(zip(maps_c, masks)):
            if mask.any():
                m[mask] = 0.0
            if len(tqumap)==2:
                tqumap[idx]["100"]['map'] = m
            else:
                tqumap[idx+1]["100"]['map'] = m
    return tqumap

#%% Calculate spectrum
@log_on_start(INFO, "Starting to calculate powerspectra up to lmax={lmax} and lmax_mask={lmax_mask}")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def tqupowerspec(tqumap: List[Dict[str, Dict]], lmax: int, lmax_mask: int, freqcomb: List[str], specfilter: List[str]) -> Dict[str, Dict]:
    """Calculate powerspectrum using MSC.pospace and TQUmaps

    Args:
        tqumap List[Dict[str, Dict]]: Planck maps (data and masks) and some header information
        lmax (int): Maximum multipol of data to be considered
        lmax_mask (int): Maximum multipol of mask to be considered. Hint: take >3*lmax
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict[str, Dict]: Powerspectra as provided from MSC.pospace
    """
    #TODO once LFI x HFI is added to the calculation, need to pass tmask2 and pmask2

    buff = {
        FREQC:
            ps.map2cls(
                tqumap=[tqumap[0][FREQC.split("-")[0]]['map'], tqumap[1][FREQC.split("-")[0]]['map'], tqumap[2][FREQC.split("-")[0]]['map']],
                tmask=tqumap[0][FREQC.split("-")[0]]['mask'],
                pmask=tqumap[1][FREQC.split("-")[0]]['mask'],
                lmax=lmax,
                lmax_mask=lmax_mask,
                tqumap2=[tqumap[0][FREQC.split("-")[1]]['map'], tqumap[1][FREQC.split("-")[1]]['map'], tqumap[2][FREQC.split("-")[1]]['map']],
                tmask2=tqumap[0][FREQC.split("-")[1]]['mask'],
                pmask2=tqumap[1][FREQC.split("-")[1]]['mask'] #this needs to be fixed, once LFI x HFI is passed
                )
            for FREQC in freqcomb}
    spectrum = dict()
    for FREQC, _ in buff.items():
        spectrum.update({
            FREQC: {
                spec: buff[FREQC][idx]
                    for idx, spec in enumerate([p for p in PLANCKSPECTRUM if p not in specfilter])
                    }
           })
    return spectrum


#%% Calculate spectrum
@log_on_start(INFO, "Starting to calculate powerspectra up to lmax={lmax} and lmax_mask={lmax_mask}")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def qupowerspec(qumap: List[Dict[str, Dict]], lmax: int, lmax_mask: int, freqcomb: List[str], specfilter: List[str]) -> Dict[str, Dict]:
    """Calculate powerspectrum using MSC.pospace and TQUmaps

    Args:
        qumap List[Dict[str, Dict]]: Planck maps (data and masks) and some header information
        lmax (int): Maximum multipol of data to be considered
        lmax_mask (int): Maximum multipol of mask to be considered. Hint: take >3*lmax
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["EB"]

    Returns:
        Dict[str, Dict]: Powerspectra as provided from MSC.pospace
    """
    buff = {
        FREQC: 
            ps.map2cl_spin(
                qumap = [qumap[0][FREQC.split("-")[0]]['map'], qumap[1][FREQC.split("-")[0]]['map']],
                lmax = lmax,
                lmax_mask = lmax_mask,
                spin = 2,
                mask = qumap[0][FREQC.split("-")[0]]['mask'],
                qumap2 = [qumap[0][FREQC.split("-")[1]]['map'], qumap[1][FREQC.split("-")[1]]['map']],
                spin2 = 2,
                mask2 = qumap[0][FREQC.split("-")[1]]['mask'],
                ret_eb_be = True
                )
            for FREQC in freqcomb}

    spectrum = dict()
    for FREQC, _ in buff.items():
        spectrum.update({
            FREQC: {
                spec: buff[FREQC][idx]
                    for idx, spec in enumerate([p for p in PLANCKSPECTRUM if p not in specfilter])
                    }
           })
    return spectrum


def create_synmap(spectrum: Dict[str, Dict], cf: Dict, mch: str, freqcomb: List[str], specfilter: List[str]) -> List[Dict[str, Dict]]:
    indir_path = cf[mch]['indir']
    mskset = cf['pa']['mskset'] # smica or lens
    freqfilter = cf['pa']["freqfilter"]
    specfilter = cf['pa']["specfilter"]

    tmask_path = cf[mch][mskset]['tmask']["path"]
    pmask_path = cf[mch][mskset]['pmask']["path"]

    tmask_filename = cf[mch][mskset]['tmask']['filename']
    pmask_filename = cf[mch][mskset]['pmask']['filename']

    nside = cf['pa']["nside"]

    synmap = dict()
    for freqc in freqcomb:
        synmap.update({
            freqc: hp.synfast(
                cls = [
                    spectrum[freqc]["TT"],
                    spectrum[freqc]["EE"],
                    spectrum[freqc]["BB"],
                    spectrum[freqc]["TE"]],
                nside = nside[0] if int(freqc.split("-")[0])<100 else nside[1],
                new=True)})

    if tmask_filename is None:
        tmask = None
        tmask_d = None
    else:
        tmask = hp.read_map(
            '{path}{tmask_path}{tmask_filename}'
            .format(
                path = indir_path,
                tmask_path = tmask_path,
                tmask_filename = tmask_filename), field=0, dtype=np.bool)
        # tmask = np.array([True for t in tmask])
        tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=nside[0])

    def multi(a,b):
        return a*b
    pmasks = [hp.read_map(
        '{path}{pmask_path}{pmask_filename}'
        .format(
            path = indir_path,
            pmask_path = pmask_path,
            pmask_filename = a), dtype=np.bool) for a in pmask_filename]
    pmask = functools.reduce(multi, pmasks)
    # pmask = np.array([True for p in pmask])
    pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=nside[0])


    flag = False
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            if "T" in spec:
                flag = True
                break
    if flag and (tmask is not None):
        tmap = {
            FREQ: {
                "map": synmap["-".join([FREQ,FREQ])][0],
                "mask": tmask_d if int(FREQ)<100 else tmask
                }for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
        }
    elif flag and (tmask is None):
        print("Temperature spectra requested, but no Temperature masking provided. Spectra including temperature are skipped")
        tmap = None
    elif flag == False:
        tmap = None

    qmap = {
        FREQ: {
            "map": synmap["-".join([FREQ,FREQ])][1],
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    umap = {
        FREQ: {
            "map": synmap["-".join([FREQ,FREQ])][2],
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]


#%% Create df for no apparent reason
@log_on_start(INFO, "Starting to create powerspectrum dataframe with {spectrum}")
@log_on_end(DEBUG, "Dataframe created successfully: '{result}' ")
def create_df(spectrum: Dict[str, Dict[str, List]], offdiag: bool, freqfilter: List[str], specfilter: List[str]) -> Dict:
    """For easier parallelisation, tracking, and plotting

    Args:
        spectrum (Dict[str, Dict[str, List]]): Powerspectra as provided from MSC.pospace
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict: powerspectra with spectrum and frequency-combinations in the columns
    """
    df = {
        spec: 
            pd.DataFrame(
                data={"{}-{}".format(FREQ,FREQ2): spectrum[FREQ+'-'+FREQ2][spec]
                    for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter
                    for FREQ2 in PLANCKMAPFREQ if (FREQ2 not in freqfilter) and _crosscomb(offdiag, FREQ, FREQ2)
            }) 
        for spec in PLANCKSPECTRUM if spec not in specfilter
    }

    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            for fkey, _ in df[spec].items():
                df[spec][fkey].index.name = 'multipole'

    return df

#%% Apply 1e12*l(l+1)/2pi
@log_on_start(INFO, "Starting to apply scaling onto data {data}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_scale(data: Dict, llp1: bool = True) -> Dict:
    """Multiplies powerspectra by :math:`l(l+1)/(2\pi)1e12`

    Args:
        df (Dict): powerspectra with spectrum and frequency-combinations in the columns
        df (llp1, Optional): Set to False, if :math:`l(l+1)/(2\pi)` should not be applied. Default: True

    Returns:
        Dict: scaled powerspectra with spectrum and frequency-combinations in the columns
    """
    for freqc, spec in data.items():
        for specID, val in spec.items():
            if llp1:
                lmax = len(next(iter((next(iter(data.values()))).values())))
                ll = lambda x: x*(x+1)*1e12/(2*np.pi)
                sc = np.array([ll(idx) for idx in range(lmax)])
                data[freqc][specID] *= sc
            else:
                print("Nothing has been scaled.")
    return data

#%% Apply Beamfunction
@log_on_start(INFO, "Starting to apply Beamfunnction on dataframe with {data}")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(data: Dict,  beamf: Dict, lmax: int, specfilter: List[str]) -> Dict:
    """divides the spectrum derived from channel `ij` and provided via `df_(ij)`,
    by `beamf_i beamf_j` as described by the planck beamf .fits-file header.

    Args:
        df (Dict): powerspectra with spectrum and frequency-combinations in the columns

        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        DataFrame: powerspectra including effect of Beam, with spectrum and frequency-combinations in the columns

    """
    TEB_dict = {
        "T": 0,
        "E": 1,
        "B": 2
    }
    LFI_dict = {
        "030": 28,
        "044": 29,
        "070": 30
    }
    for freqc, spec in data.items():
        freqs = freqc.split('-')
        hdul = beamf[freqc]
        for specID, val in spec.items():
            if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[0]])[:lmax+1]
                data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmax+1]
            elif int(freqs[0]) < 100 and int(freqs[1]) < 100:
                data[freqc][specID] /= np.concatenate(
                    (np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0)),
                    np.array([0.12 for n in range(lmax+1-len(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0)))])))
                data[freqc][specID] /= np.concatenate((
                    np.sqrt(hdul["LFI"][LFI_dict[freqs[1]]].data.field(0)[:lmax+1]),
                    np.array([0.12 for n in range(lmax+1-len(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0)))])))
            else:
                data[freqc][specID] /= np.concatenate((
                    np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0)[:lmax+1]),
                    np.array([0.12 for n in range(lmax+1-len(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0)))])))
 
                data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmax+1]
    return data

#%% Build covariance matrices
@log_on_start(INFO, "Starting to build convariance matrices with {data}")
@log_on_end(DEBUG, "Covariance matrix built successfully: '{result}' ")
def build_covmatrices(data: Dict, offdiag: str, lmax: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, np.ndarray]:
    """Calculates the covariance matrices from the data

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
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
                    if _crosscomb(offdiag, FREQ, FREQ2):
                        ispec=-1
                        for spec in PLANCKSPECTRUM:
                            if spec not in specfilter:
                                ispec+=1
                                cov[spec][ifreq][ifreq2] = data[FREQ+'-'+FREQ2][spec]
                                cov[spec][ifreq2][ifreq] = data[FREQ+'-'+FREQ2][spec]
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
    return cov_inv_l

# %% Calculate weightings and store in df
@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: Dict, lmax: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, DataFrame]:
    """Calculates weightings of the respective Frequency channels

    Args:
        cov (Dict): The inverted covariance matrices of Dimension [lmax,Nspec,Nspec]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict[str, DataFrame]: The weightings of the respective Frequency channels
    """
    elaw = np.ones(len([dum for dum in PLANCKMAPFREQ if dum not in freqfilter]))
    weighting = {spec: np.array([(cov[spec][l] @ elaw) / (elaw.T @ cov[spec][l] @ elaw)
                     if cov[spec][l] is not None else np.array([0.0 for n in range(len(elaw))])
                        for l in range(lmax) if l in cov[spec].keys()])
                    for spec in PLANCKSPECTRUM if spec not in specfilter}
    
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

    return weights

if __name__ == '__main__':
    import doctest
    # doctest.run_docstring_examples(get_data, globals(), verbose=True)