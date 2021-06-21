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

sys.path.append('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/component_separation/spherelib/python/spherelib/')

import json
import os

import healpy as hp
import numpy as np
from logdecorator import log_on_end, log_on_error, log_on_start

import component_separation
import component_separation.MSC.MSC.pospace as ps
import component_separation.preprocess as prep
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Helperfunctions as hpf
from component_separation.cs_util import Planckf, Planckr, Plancks

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

PLANCKMAPAR = [p.value for p in list(Planckr)]
PLANCKMAPNSIDE = cf["pa"]['nside']

""" Doctest:
The following constants be needed because functions are called with globals()
"""
path='test/data'
freqfilter = [Planckf.LFI_1.value, Planckf.LFI_2.value, Planckf.HFI_1.value, Planckf.HFI_5.value, Planckf.HFI_6.value]
specfilter = [Plancks.TE, Plancks.TB, Plancks.ET, Plancks.BT]
lmax = 20
lmax_mask = 80


def _crosscomb(option, f1, f2):
        if option:
            return int(f1) <= int(f2)
        else:
            return int(f1) == int(f2)


@log_on_start(INFO, "Starting to calculate powerspectra up to lmax={lmax} and lmax_mask={lmax_mask}")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def tqupowerspec(tqumap, tmask: List, pmask: List, lmax: int, lmax_mask: int) -> Dict[str, Dict]:
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
    def _ud_grade(data, FREQ):
        if int(FREQ)<100:
            return hp.pixelfunc.ud_grade(data, nside_out=PLANCKMAPNSIDE[0])
        else:
            return hp.pixelfunc.ud_grade(data, nside_out=PLANCKMAPNSIDE[1])
    buff = {
        FREQC:
            ps.map2cls(
                tqumap=tqumap[FREQC.split("-")[0]],
                tmask=_ud_grade(tmask, FREQC.split("-")[0]),
                pmask=_ud_grade(pmask, FREQC.split("-")[0]),
                lmax=lmax,
                lmax_mask=lmax_mask,
                tqumap2=tqumap[FREQC.split("-")[1]],
                tmask2=_ud_grade(tmask, FREQC.split("-")[1]),
                pmask2=_ud_grade(pmask, FREQC.split("-")[1])
                )
            for FREQC in csu.freqcomb}
    spectrum = dict()
    for FREQC, _ in buff.items():
        spectrum.update({
            FREQC: {
                spec: buff[FREQC][idx]
                    for idx, spec in enumerate([p for p in csu.PLANCKSPECTRUM_f])
                    }
           })
    return spectrum


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
                    for idx, spec in enumerate([p for p in csu.PLANCKSPECTRUM_f])
                    }
           })
    return spectrum


def create_mapsyn(spectrum: Dict[str, Dict], cf: Dict) -> List[Dict[str, Dict]]:
    nside = cf['pa']["nside"]
    synmap = dict()
    for freqc in csu.freqcomb:
        synmap.update({
            freqc: hp.synfast(
                cls = [
                    spectrum[freqc]["TT"],
                    spectrum[freqc]["EE"],
                    spectrum[freqc]["BB"],
                    spectrum[freqc]["TE"]],
                nside = nside[0] if int(freqc.split("-")[0])<100 else nside[1],
                new=True)})
    maps = {
        FREQ: synmap["-".join([FREQ,FREQ])][0:3]
            for FREQ in csu.PLANCKMAPFREQ_f
    }
    return maps


@log_on_start(INFO, "Starting to apply scaling onto data {data}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_scale(data: Dict, scale: str = 'C_l') -> Dict:
    """Multiplies powerspectra by :math:`l(l+1)/(2\pi)1e12` and the pixwindowfunction

    Args:
        df (Dict): powerspectra with spectrum and frequency-combinations in the columns
        
    Returns:
        Dict: scaled powerspectra with spectrum and frequency-combinations in the columns
    """
    for freqc, spec in data.items():
        for specID, val in spec.items():
            lmax = len(next(iter((next(iter(data.values()))).values())))
            if scale == "D_l":
                sc = np.array([hpf.llp1e12(idx) for idx in range(lmax)])
            elif scale == "C_l":
                sc = np.array([1e12 for idx in range(lmax)])
            data[freqc][specID] *= sc
            if int(freqc.split("-")[0]) < 100:
                data[freqc][specID] /= hp.pixwin(1024)[:lmax]
            else:
                data[freqc][specID] /= hp.pixwin(2048)[:lmax]
            if int(freqc.split("-")[1]) < 100:
                data[freqc][specID] /= hp.pixwin(1024)[:lmax]
            else:
                data[freqc][specID] /= hp.pixwin(2048)[:lmax]
    return data


@log_on_start(INFO, "Starting to apply Beamfunction")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(data: Dict,  beamf: Dict, lmax: int, specfilter: List[str]) -> Dict:
    """divides the spectrum derived from channel `ij` and provided via `df_(ij)`,
    by `beamf_i beamf_j` as described by the planck beamf .fits-file header.

    Args:
        df (Dict): powerspectra with spectrum and frequency-combinations in the columns

        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        np.array: powerspectra including effect of Beam, with spectrum and frequency-combinations in the columns

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
                b = np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0))
                buff = np.concatenate((
                    b[:min(lmax+1, len(b))],
                    np.array([np.NaN for n in range(max(0, lmax+1-len(b)))])))
                data[freqc][specID] /= buff
                data[freqc][specID] /= buff
            else:
                b = np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0))
                buff = np.concatenate((
                    b[:min(lmax+1, len(b))],
                    np.array([np.NaN for n in range(max(0, lmax+1-len(b)))])))
                data[freqc][specID] /= buff
                data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmax+1]
    return data


@log_on_start(INFO, "Starting to build convariance matrices with {data}")
@log_on_end(DEBUG, "Covariance matrix built successfully: '{result}' ")
def build_covmatrices(data: Dict, lmax: int, freqfilter: List[str], specfilter: List[str], Tscale: str = "K_CMB") -> Dict[str, np.ndarray]:
    """Calculates the covariance matrices from the data

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        Dict[str, np.ndarray]: The covariance matrices of Dimension [Nspec,Nspec,lmax]
    """
    NFREQUENCIES = len([FREQ for FREQ in csu.PLANCKMAPFREQ_f])
    cov = {spec: np.zeros(shape=(NFREQUENCIES, NFREQUENCIES, lmax+1))
                for spec in csu.PLANCKSPECTRUM_f}
    def LFI_cutoff(fr):
        # KEEP. Cutting off LFI channels for ell=700 as they cause numerical problems
        return {
            '030': 2000,
            '044': 2000,
            '070': 2000,
            '100': lmax,
            '143': lmax,
            '217': lmax,
            '353': lmax
        }[fr]
    ifreq, ifreq2, ispec = -1, -1, 0
    for FREQ in csu.PLANCKMAPFREQ:
        ifreq2 = -1
        if FREQ not in freqfilter:
            ifreq+=1
            for FREQ2 in csu.PLANCKMAPFREQ:
                if FREQ2 not in freqfilter:
                    ifreq2+=1
                    if _crosscomb(True, FREQ, FREQ2):
                        ispec=-1
                        for spec in csu.PLANCKSPECTRUM:
                            if spec not in specfilter:
                                ispec+=1
                                if int(FREQ)<100 or int(FREQ2)<100:
                                    b  = np.array([np.nan for n in range(max(lmax+1-min(LFI_cutoff(FREQ),LFI_cutoff(FREQ2)), 0))])
                                    a = np.concatenate((data[FREQ+'-'+FREQ2][spec][:min(lmax+1, min(LFI_cutoff(FREQ),LFI_cutoff(FREQ2)))], b))
                                else:
                                    a = data[FREQ+'-'+FREQ2][spec]
                                if Tscale == "K_RJ":
                                    cov[spec][ifreq][ifreq2] = a * prep.tcmb2trj_sc(FREQ) * prep.tcmb2trj_sc(FREQ2)
                                    cov[spec][ifreq2][ifreq] = a * prep.tcmb2trj_sc(FREQ) * prep.tcmb2trj_sc(FREQ2)
                                else:
                                    cov[spec][ifreq][ifreq2] = a
                                    cov[spec][ifreq2][ifreq] = a
    return cov


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
    def maskit(a):
        """drops all row and columns with np.nan's.
        """
        masked = a[ ~np.isnan(a) ]
        mskd = masked.reshape(int(np.sqrt(len(masked))),int(np.sqrt(len(masked))))
        return mskd

    def is_invertible(a, l):
        truth = a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
        if not truth:
            print('{} not invertible: {}'.format(l, a) )
        return truth

    def shp2cov_nan(shp):
        a = np.empty(shp)
        a[:] = np.nan
        return a

    def pad_shape(a):
        return ((7-a.shape[0],0),(7-a.shape[0],0))

    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0.0)
        vector[:pad_width[0]] = pad_value
        if pad_width[1] != 0:                      # <-- (0 indicates no padding)
            vector[-pad_width[1]:] = pad_value
    
    cov_inv_l = {
        spec: {
            l: np.pad(np.linalg.inv(maskit(cov[spec][:,:,l])), pad_shape(maskit(cov[spec][:,:,l])), pad_with)
            if is_invertible(maskit(cov[spec][:,:,l]), l)
            else np.pad(shp2cov_nan(maskit(cov[spec][:,:,l]).shape), pad_shape(maskit(cov[spec][:,:,l])), pad_with)
                for l in range(lmax)
            }for spec in csu.PLANCKSPECTRUM_f
        }
    return cov_inv_l


@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: Dict, lmax: int, freqfilter: List[str], specfilter: List[str], Tscale: str = "K_CMB") -> np.array:
    """Calculates weightings of the respective Frequency channels

    Args:
        cov (Dict): The inverted covariance matrices of Dimension [lmax,Nspec,Nspec]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]

    Returns:
        np.array: The weightings of the respective Frequency channels
    """
    
    def _elaw(shp):
        if Tscale == "K_RJ":
            return np.array([prep.tcmb2trj_sc(FREQ) for FREQ in csu.PLANCKMAPFREQ_f])
        else:
            return np.ones((shp))
    weight_arr = np.array([
            [(cov[spec][l] @ _elaw(cov[spec][l].shape[0])) / (_elaw(cov[spec][l].shape[0]).T @ cov[spec][l] @ _elaw(cov[spec][l].shape[0]))
                for l in range(lmax)]
        for spec in csu.PLANCKSPECTRUM_f])
    return weight_arr


def smoothC_l(data, smoothing_window=5, max_polynom=2):
    from scipy.signal import savgol_filter
    for key, val in data.items():
        for k, v in val.items():
            data[key][k] = savgol_filter(v, smoothing_window, max_polynom)
    return data


def spec2alms(spectrum):
    return None


def alms2almsxweight(alms, weights):
    return None


def alms2cls(alms_w):
    return None


if __name__ == '__main__':
    import doctest

    # doctest.run_docstring_examples(get_data, globals(), verbose=True)