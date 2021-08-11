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

import os

import healpy as hp
import numpy as np
from logdecorator import log_on_end, log_on_error, log_on_start

import component_separation
import component_separation.MSC.MSC.pospace as ps
import component_separation.transform_map as trsf_m
from component_separation.cs_util import Helperfunctions as hpf
from component_separation.cs_util import Planckf, Planckr, Plancks


""" Doctest:
The following constants be needed because functions are called with globals()
"""
# path='test/data'
# freqfilter = [Planckf.LFI_1.value, Planckf.LFI_2.value, Planckf.HFI_1.value, Planckf.HFI_5.value, Planckf.HFI_6.value]
# specfilter = [Plancks.TE, Plancks.TB, Plancks.ET, Plancks.BT]
# lmax = 20
# lmax_mask = 80


@log_on_start(INFO, "Starting to calculate powerspectra up to lmax={lmax} and lmax_mask={lmax_mask}")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def tqupowerspec(tqumap, tmask: List, pmask: List, lmax: int, lmax_mask: int, nside_out, freqcomb) -> Dict[str, Dict]:
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
            return hp.pixelfunc.ud_grade(data, nside_out=nside_out[0])
        else:
            return hp.pixelfunc.ud_grade(data, nside_out=nside_out[1])

    retval = np.array([
        ps.map2cls(
            tqumap=tqumap[FREQC.split("-")[0]],
            tmask=_ud_grade(tmask[FREQC.split("-")[0]], FREQC.split("-")[0]),
            pmask=_ud_grade(pmask[FREQC.split("-")[0]], FREQC.split("-")[0]),
            lmax=lmax,
            lmax_mask=lmax_mask,
            tqumap2=tqumap[FREQC.split("-")[1]],
            tmask2=_ud_grade(tmask[FREQC.split("-")[0]], FREQC.split("-")[1]),
            pmask2=_ud_grade(pmask[FREQC.split("-")[0]], FREQC.split("-")[1])
        ) for FREQC in freqcomb 
    ])
    return retval


def create_mapsyn(spectrum: Dict[str, Dict], cf: Dict, freqcomb) -> List[Dict[str, Dict]]:
    synmap = dict()
    for freqc in freqcomb:
        synmap.update({
            freqc: hp.synfast(
                cls = [
                    spectrum[freqc]["TT"],
                    spectrum[freqc]["EE"],
                    spectrum[freqc]["BB"],
                    spectrum[freqc]["TE"]],
                nside = cf.nside[0] if int(freqc.split("-")[0])<100 else cf.nside[1],
                new=True)})

    #TODO return only numpy
    return np.array([])


@log_on_start(INFO, "Starting to build convariance matrices with {data}")
@log_on_end(DEBUG, "Covariance matrix built successfully: '{result}' ")
def cov2weight(data: np.array, freqs=np.array([F.value for F in list(Planckf)[:-2]]), Tscale='K_CMB'):
    """Calculates weights vector from cov matrix

    Args:
        data: powerspectra with spectrum and frequency-combinations in the columns
        
    Returns:
        Dict[str, np.ndarray]: The covariance matrices of Dimension [Nspec,Nspec,lmax]
    """
    nfreq = len(freqs) #KEEP. shape of weights shall have all frequencies
    def invert_cov(cov: np.array):
        """Inverts a covariance matrix
        Args:
            cov: The covariance matrices of Dimension [Nspec,Nspec,lmax]
        """
        def maskit(a):
            """drops all row and columns with np.nan's.
            """
            masked = a[ ~np.isnan(a) ]
            mskd = masked.reshape(int(np.sqrt(len(masked))),int(np.sqrt(len(masked))))
            return mskd

        def is_invertible(a):
            truth = a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
            if not truth:
                pass
                # print('not invertible: {}'.format(a) )
            return truth

        def shp2cov_nan(shp):
            a = np.empty(shp)
            a[:] = np.nan
            return a

        def pad_shape(a):
            return ((nfreq-a.shape[0],0),(nfreq-a.shape[0],0))

        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 0.0)
            vector[:pad_width[0]] = pad_value
            if pad_width[1] != 0:                      # <-- (0 indicates no padding)
                vector[-pad_width[1]:] = pad_value

        ret = np.zeros(shape=(cov.shape[0], nfreq, nfreq, cov.shape[-1]))
        if is_invertible(maskit(cov)):
            ret = np.pad(
                np.linalg.inv(maskit(cov)),
                pad_shape(maskit(cov)),
                pad_with)
        else:
            ret = np.pad(
                shp2cov_nan(maskit(cov).shape),
                pad_shape(maskit(cov)), pad_with)
        return ret

    def calc_w(cov: np.array) -> np.array:
        """Calculates weightings of the frequency channels
        Args:
            cov: The inverted covariance matrices of Dimension [lmax,Nspec,Nspec]
        Returns:
            np.array: The weightings of the respective Frequency channels
        """

        Tscale_mat = np.zeros(shape=(nfreq,nfreq)) #keep shape as if for all frequencies
        for idx1, FREQ1 in enumerate(freqs):
            for idx2, FREQ2 in enumerate(freqs):
                Tscale_mat[idx1,idx2] = trsf_m.tcmb2trj_sc(FREQ1, fr=r'K_CMB', to=Tscale) * trsf_m.tcmb2trj_sc(FREQ2, fr=r'K_CMB', to=Tscale)

        elaw = np.array([trsf_m.tcmb2trj_sc(FREQ, fr=r'K_CMB', to=Tscale) for FREQ in freqs])
        weight_arr = np.zeros(shape=(np.take(cov.shape, [0,1,-1])))
        for spec in range(weight_arr.shape[0]):
            for l in range(cov.shape[-1]):
                weight_arr[spec,:,l] = np.array([
                    (invert_cov(np.multiply(cov[spec,:,:,l],Tscale_mat)) @ elaw) / (elaw.T @ invert_cov(np.multiply(cov[spec,:,:,l],Tscale_mat)) @ elaw)])
        return weight_arr

    return calc_w(data)


@log_on_start(INFO, "Starting to build convariance matrices with {data}")
@log_on_end(DEBUG, "Covariance matrix built successfully: '{result}' ")
def build_covmatrices(data: np.array, Tscale, freqcomb, PLANCKMAPFREQ_f):
    """Calculates the covariance matrices from the data

    Args:
        data: powerspectra with spectrum and frequency-combinations in the columns
        
    Returns:
        Dict[str, np.ndarray]: The covariance matrices of Dimension [Nspec,Nspec,lmax]
    """
    def get_nfreq(nfreqcombs):
        _i = 0
        _j = 0
        while True:
            _i+=1
            _j +=_i
            if _j == nfreqcombs:
                break
        return _i
    NFREQUENCIES = get_nfreq(data.shape[0])
    lmaxp1 = data.shape[-1]
    def LFI_cutoff(fr):
        # KEEP. Cuts LFI channels for ell=700 as they cause numerical problems
        return {
            30: 1000,
            44: 1000,
            70: 1000,
            100: lmaxp1,
            143: lmaxp1,
            217: lmaxp1,
            353: lmaxp1
        }[fr]


    covn = np.zeros(shape=(data.shape[1], NFREQUENCIES, NFREQUENCIES, lmaxp1))
    for sidx in range(covn.shape[0]):
        for fcombidx, freqc in enumerate(freqcomb):
            FREQ1, FREQ2 = int(freqc.split('-')[0]), int(freqc.split('-')[1])
            freqf = np.array([int(n) for n in PLANCKMAPFREQ_f])
            fidx1 = np.where(freqf == FREQ1)[0][0]
            fidx2 = np.where(freqf == FREQ2)[0][0]

            if FREQ1<100 or FREQ2<100:
                b  = np.array([np.nan for n in range(max(lmaxp1-min(LFI_cutoff(FREQ1),LFI_cutoff(FREQ2)), 0))])
                a = np.concatenate((data[fcombidx,sidx][:min(lmaxp1, min(LFI_cutoff(FREQ1),LFI_cutoff(FREQ2)))], b))
            else:
                a = data[fcombidx,sidx]
            covn[sidx,fidx1,fidx2] = a * trsf_m.tcmb2trj_sc(FREQ1, fr=r'K_CMB', to=Tscale) * trsf_m.tcmb2trj_sc(FREQ2, fr=r'K_CMB', to=Tscale)
            covn[sidx,fidx2,fidx1] = a * trsf_m.tcmb2trj_sc(FREQ1, fr=r'K_CMB', to=Tscale) * trsf_m.tcmb2trj_sc(FREQ2, fr=r'K_CMB', to=Tscale)
    return covn


@log_on_start(INFO, "Starting to invert convariance matrix {cov}")
@log_on_end(DEBUG, "Inversion successful: '{result}' ")
def invert_covmatrices(cov: np.array):
    """Inverts a covariance matrix

    Args:
        cov: The covariance matrices of Dimension [Nspec,Nspec,lmax]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
    """

    nfreq = 7 #KEEP. shape of weights shall have all frequencies. cov['EE'].shape[0]
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
        return ((nfreq-a.shape[0],0),(nfreq-a.shape[0],0))

    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0.0)
        vector[:pad_width[0]] = pad_value
        if pad_width[1] != 0:                      # <-- (0 indicates no padding)
            vector[-pad_width[1]:] = pad_value

    ret = np.zeros(shape=(cov.shape[0], nfreq, nfreq, cov.shape[-1]))
    for spec in range(cov.shape[0]):
        for l in range(cov.shape[-1]):
            if is_invertible(maskit(cov[spec,:,:,l]), l):
                ret[spec,:,:,l] = np.pad(
                    np.linalg.inv(maskit(cov[spec,:,:,l])),
                    pad_shape(maskit(cov[spec,:,:,l])),
                    pad_with)
            else:
                ret[spec,:,:,l] = np.pad(
                    shp2cov_nan(maskit(cov[spec,:,:,l]).shape),
                    pad_shape(maskit(cov[spec,:,:,l])), pad_with)
    return ret


@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: np.array, freqs, Tscale: str = r"K_CMB") -> np.array:
    """Calculates weightings of the respective Frequency channels
    Args:
        cov (Dict): The inverted covariance matrices of Dimension [lmax,Nspec,Nspec]

    Returns:
        np.array: The weightings of the respective Frequency channels
    """

    elaw = np.array([trsf_m.tcmb2trj_sc(FREQ, fr=r'K_CMB', to=Tscale) for FREQ in freqs])
    weight_arr = np.zeros(shape=(np.take(cov.shape, [0,1,-1])))
    for spec in range(weight_arr.shape[0]):
        for l in range(cov.shape[-1]):
            weight_arr[spec,:,l] = np.array([
                    (cov[spec,:,:,l] @ elaw) / (elaw.T @ cov[spec,:,:,l] @ elaw)])
    return weight_arr


def map2alm_spin(maps, pmask, spin, lmax):
    return ps.map2alm_spin([maps[1] * pmask, maps[2] * pmask], spin,
                                lmax=lmax)


def map2alm(maps, tmask, lmax):
    return ps.map2alm([maps[0] * tmask, maps[2] * pmask],
                                lmax=lmax)


def alms2almsxweight(alms, weights):
    return None


def alms2cls(alms_w):
    return None


if __name__ == '__main__':
    import doctest