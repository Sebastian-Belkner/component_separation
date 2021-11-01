import healpy as hp
import numpy as np

from component_separation.cs_util import Config
from component_separation.io import IO
from component_separation.config_planck import Planckf, Planckr, Plancks
from component_separation.cs_util import Helperfunctions as hpf
import component_separation.map as mp

from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

@log_on_start(INFO, "Starting to build convariance matrices with {data}")
@log_on_end(DEBUG, "Covariance matrix built successfully: '{result}' ")
def build_covmatrices(data: np.array, Tscale, freqcomb, PLANCKMAPFREQ_f, cutoff=None):
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
    if cutoff is None:
        cutoff = lmaxp1
    def LFI_cutoff(fr):
        # KEEP. Cuts LFI channels for ell=700 as they cause numerical problems
        return {
            30: cutoff,
            44: cutoff,
            70: cutoff,
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
            covn[sidx,fidx1,fidx2] = a * mp.tcmb2trj_sc(FREQ1, fr=r'K_CMB', to=Tscale) * mp.tcmb2trj_sc(FREQ2, fr=r'K_CMB', to=Tscale)
            covn[sidx,fidx2,fidx1] = a * mp.tcmb2trj_sc(FREQ1, fr=r'K_CMB', to=Tscale) * mp.tcmb2trj_sc(FREQ2, fr=r'K_CMB', to=Tscale)
    return covn

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
                Tscale_mat[idx1,idx2] = mp.tcmb2trj_sc(FREQ1, fr=r'K_CMB', to=Tscale) * mp.tcmb2trj_sc(FREQ2, fr=r'K_CMB', to=Tscale)

        elaw = np.array([mp.tcmb2trj_sc(FREQ, fr=r'K_CMB', to=Tscale) for FREQ in freqs])
        weight_arr = np.zeros(shape=(np.take(cov.shape, [0,1,-1])))
        for spec in range(weight_arr.shape[0]):
            for l in range(cov.shape[-1]):
                weight_arr[spec,:,l] = np.array([
                    (invert_cov(np.multiply(cov[spec,:,:,l],Tscale_mat)) @ elaw) / (elaw.T @ invert_cov(np.multiply(cov[spec,:,:,l],Tscale_mat)) @ elaw)])
        return weight_arr

    return calc_w(data)


@log_on_start(INFO, "Starting to invert convariance matrix {cov}")
@log_on_end(DEBUG, "Inversion successful: '{result}' ")
def cov2icov(cov: np.array):
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

    
@hpf.deprecated
@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: np.array, freqs, Tscale: str = r"K_CMB") -> np.array:
    """Calculates weightings of the respective Frequency channels
    Args:
        cov (Dict): The inverted covariance matrices of Dimension [lmax,Nspec,Nspec]

    Returns:
        np.array: The weightings of the respective Frequency channels
    """

    elaw = np.array([mp.tcmb2trj_sc(FREQ, fr=r'K_CMB', to=Tscale) for FREQ in freqs])
    weight_arr = np.zeros(shape=(np.take(cov.shape, [0,1,-1])))
    for spec in range(weight_arr.shape[0]):
        for l in range(cov.shape[-1]):
            weight_arr[spec,:,l] = np.array([
                    (cov[spec,:,:,l] @ elaw) / (elaw.T @ cov[spec,:,:,l] @ elaw)])
    return weight_arr