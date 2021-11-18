from enum import Enum
import itertools
import platform
import numpy as np


class Frequency(Enum):
    LFI_1 = '021'
    LFI_2 = '025'
    LFI_3 = '030'
    LFI_4 = '036'
    LFI_5 = '043'
    LFI_6 = '052'
    LFI_7 = '062'
    LFI_8 = '075'
    LFI_9 = '090'
    HFI_1 = '108'
    HFI_2 = '129'
    HFI_3 = '155'
    HFI_4 = '186'
    HFI_5 = '223'
    HFI_6 = '268'
    HFI_7 = '321'
    HFI_8 = '385'
    HFI_9 = '462'
    HFI_10 = '555'
    HFI_11 = '666'
    HFI_12 = '799'

    
class Spectrum(Enum):
    # the order must be the same as the order of pospace function returns
    TT = "TT"#00
    EE = "EE"#11
    BB = "BB"#22
    TE = "TE"#01
    TB = "TB"#02
    EB = "EB"#12
    ET = "ET"#10
    BT = "BT"#20
    BE = "BE"#21


class Beamwidth:
    Bw = np.array([38.4, 32.0, 28.3, 23.6, 22.2, 18.4, 12.8, 10.7, 9.5, 7.9, 7.4, 6.2, 4.3, 3.6, 3.2, 2.6, 2.5, 2.1, 1.5, 1.3, 1.1])/60.


class Params:
    mskset = "lens"
    freqdset = "d90sim"
    spectrum_type = "JC"
    lmax = 1500
    lmax_mask = 2500
    freqdatsplit = ""
    num_sim = 5
    binname = "SMICA_highell_bins"
    overwrite_cache = True
    simdata = True
    simid = 0

    specfilter = [
        "TB",
        "EB",
        "ET",
        "BT",
        "BE"
    ]
    nside_out = [
        2048,
        2048
    ]
    nside_desc_map = [
        2048,
        2048
    ]
    outdir_ap= "/global/cscratch1/sd/sebibel/compsep/pico/"
    powspec_truthfile= "/global/homes/s/sebibel/git/component_separation/data/powspecplanck.txt"

    # Now, generate helper parameters from these
    uname = platform.uname()
    if uname.node == "DESKTOP-KMIGUPV":
        mch = "XPS"
    else:
        mch = "NERSC"

    FREQFILTER = [
        Frequency.LFI_3.value,
        Frequency.LFI_4.value,
        Frequency.LFI_5.value,
        Frequency.LFI_6.value,
        Frequency.LFI_7.value,
        Frequency.LFI_8.value,
        Frequency.HFI_2.value,
        Frequency.HFI_3.value,
        Frequency.HFI_4.value,
        Frequency.HFI_5.value,
        Frequency.HFI_6.value,
        Frequency.HFI_7.value,
        Frequency.HFI_8.value,
        Frequency.HFI_9.value,
        Frequency.HFI_10.value
        ]

    FREQ = [p.value for p in list(Frequency)]
    FREQ_f = [p.value for p in list(Frequency)
    if p.value not in [
        Frequency.LFI_3.value,
        Frequency.LFI_4.value,
        Frequency.LFI_5.value,
        Frequency.LFI_6.value,
        Frequency.LFI_7.value,
        Frequency.LFI_8.value,
        Frequency.HFI_2.value,
        Frequency.HFI_3.value,
        Frequency.HFI_4.value,
        Frequency.HFI_5.value,
        Frequency.HFI_6.value,
        Frequency.HFI_7.value,
        Frequency.HFI_8.value,
        Frequency.HFI_9.value,
        Frequency.HFI_10.value
    ]]

    SPECTRUM = [p.value for p in list(Spectrum)]

    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
        for FREQ, FREQ2  in itertools.product(FREQ,FREQ)
            if FREQ not in [
        Frequency.LFI_3.value,
        Frequency.LFI_4.value,
        Frequency.LFI_5.value,
        Frequency.LFI_6.value,
        Frequency.LFI_7.value,
        Frequency.LFI_8.value,
        Frequency.HFI_2.value,
        Frequency.HFI_3.value,
        Frequency.HFI_4.value,
        Frequency.HFI_5.value,
        Frequency.HFI_6.value,
        Frequency.HFI_7.value,
        Frequency.HFI_8.value,
        Frequency.HFI_9.value,
        Frequency.HFI_10.value
        ] and (FREQ2 not in [
        Frequency.LFI_3.value,
        Frequency.LFI_4.value,
        Frequency.LFI_5.value,
        Frequency.LFI_6.value,
        Frequency.LFI_7.value,
        Frequency.LFI_8.value,
        Frequency.HFI_2.value,
        Frequency.HFI_3.value,
        Frequency.HFI_4.value,
        Frequency.HFI_5.value,
        Frequency.HFI_6.value,
        Frequency.HFI_7.value,
        Frequency.HFI_8.value,
        Frequency.HFI_9.value,
        Frequency.HFI_10.value
        ]) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in SPECTRUM if spec not in [
        "TB",
        "EB",
        "ET",
        "BT",
        "BE"
    ]]


class d90:

    def _get_ddir():

        return "/project/projectdirs/pico/data_xx.yy/90.91/"


    def _get_dfn():

        return "pico_90p91_comb_f321_b03_ellmin00_map_2048_mc_0059.fits"


    def _get_noisedir():

        return "INTERNAL"


    def _get_noisefn():

        return "half_diff_npipe6v20{split}_{freq}_{nside}.fits"


    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy"


class d90sim:
    freq = [p.value for p in list(Frequency)]
    boloid = ['38', '32', '28', '24', '22', '18', '13', '11', '10', '08', '07', '06',
       '04', '04', '03', '03', '02', '02', '02', '01', '01']
    nside = '0512'
    simid_loc = np.arange(0,100)


    def _get_ddir(simid, freqdatsplit):

        return "/project/projectdirs/pico/data_xx.yy/90.91/"


    @classmethod
    def _get_dfn(cls, freqdatsplit, freq, simid):

        return "pico_90p91_comb_f{freq}_b{boloid}_ellmin00_map_{nside}_mc_{simid}.fits".format(
            freq = freq,
            boloid = cls.boloid[np.where(np.array(cls.freq)==freq)[0][0]],
            nside = cls.nside,
            simid = str(simid).zfill(4)
        )


    def _get_noisedir():

        return '/project/projectdirs/pico/data_xx.yy/90.00/'


    @classmethod
    def _get_noisefn(cls, freq, simid):

        return "cmbs4_90_noise_f{freq}_b{boloid}_ellmin00_map_{nside}_mc_{simid}.fits".format(
            freq = freq,
            boloid = cls.boloid[np.where(np.array(cls.freq)==freq)[0][0]],
            nside = cls.nside,
            simid = str(simid).zfill(4)
        )


    def _get_cmbdir():
        assert 0, "To be implemented"
    

    def _get_cmbfn():
        assert 0, "To be implemented"


    def _get_signalest():
        assert 0, "To be implemented"

        return "/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy"


class Lens_Mask:

    def get_dir():

        return "/global/homes/s/sebibel/data/mask/"


    def get_fn(TorP, apodized=False):

        if TorP == "T":
            if apodized:
                return [ 
                    "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz_apodized.npy"]
            else:
                return [
                    "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
                    ]
        else:
            if apodized:
                return [
                    "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz_apodized.npy"
                ]
            else:
                return [
                    "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
                    ]


class Smica_Mask:


    def get_dir():

        return "/global/homes/s/sebibel/data/mask/"


    def get_fn(TorP, apodized=False):

        if TorP == "T":
            if apodized:
                return [
                    "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz_apodized.npy"
                    ]
            else:
                return [
                    "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
                    ]
        else:
            if apodized:
                return [
                    "psmaskP_2048_gmaskP_apodized.npy",
                    ]
            else:
                return [
                    "psmaskP_2048.fits.gz",
                    "gmaskP_apodized_0_2048.fits.gz"
                    ]


class Beamfd90sim:
    freq = [p.value for p in list(Frequency)]
    arcmin = {
        '62': '12.8',
        '75': '10.7',
        '90': '9.5',
        '108': '7.9',
        '129': '7.4',
        '155': '6.2',
        '186': '4.3'
    }
    beamf = {
        "HFI": {     
            "ap": "/project/projectdirs/pico/reanalysis/powspecpars/Bl/",
            "filename": "bl_gauss_{arcmin}arcmin_{freq}_TP.dat "
        },
        "LFI": {     
            "ap": "/project/projectdirs/pico/reanalysis/powspecpars/Bl",
            "filename": "bl_gauss_{arcmin}arcmin_{freq}_TP.dat "
        },
        'info' : "d90sim"
    }


    @classmethod
    def get_beamf(cls, fits, freqcomb, lmax, freqdatsplit):
        import healpy as hp
        """Return Beams, but only from frequencies of interest

        Returns:
            np.array: beams
        """
        freqs = Params.FREQ_f
        allfreqs = np.array([f.value for f in list(Frequency)])
        lmax = Params.lmax if lmax is None else lmax
        ret = np.ones(shape=(3, len(freqs), len(freqs), lmax+1))
        for ifa, freqa in enumerate(freqs):
            for ifb, freqb in enumerate(freqs):
                if ifa<=ifb:
                    bifa = np.where(allfreqs==freqa)[0][0]
                    bifb = np.where(allfreqs==freqb)[0][0]
                    ret[:,ifa,ifb,:] *= np.sqrt(hp.gauss_beam(np.radians(Beamwidth.Bw[bifa]), lmax=lmax, pol=True)[:,0:3].T)
                    ret[:,ifa,ifb,:] *= np.sqrt(hp.gauss_beam(np.radians(Beamwidth.Bw[bifb]), lmax=lmax, pol=True)[:,0:3].T)
                else:
                    ret[:,ifb,ifa,:] = ret[:,ifa,ifb,:]
        return ret


class Beamfd90csim:
    freq = 'comb'
    arcmin = {
        'comb': '15.0'
    }
    beamf_info = {
        "HFI": {     
            "ap": "/project/projectdirs/pico/reanalysis/powspecpars/Bl/",
            "filename": "bl_gauss_{arcmin}arcmin_{freq}_TP.dat "
        },
        "LFI": {     
            "ap": "/project/projectdirs/pico/reanalysis/powspecpars/Bl",
            "filename": "bl_gauss_{arcmin}arcmin_{freq}_TP.dat "
        },
        'info' : "d90csim"
    }


    @classmethod
    def get_beamf(cls, freq_loc=None, lmax_loc=None):
        import healpy as hp
        """Return healpy gaussbeams as np.array

        Returns:
            np.array: beams
        """
        freq = [f.value for f in list(Frequency)] if freq_loc is None else freq_loc
        lmax = Params.lmax if lmax_loc is None else lmax_loc
        ret = np.array(shape=(3, len(freq), len(freq), lmax))
        for idf, freq in enumerate(freq):
            ret[:,freq,freq,:] = hp.gauss_beam(np.radian(Beamwidth.Bw[idf]), lmax=lmax, pol=True)[0:3]
        return ret

    
    @classmethod
    def get_beaminfo(cls):
        assert 0, "Files don't seem to be beams"

        return cls.beamf_info
    

class Asserter:
    info_component = ["N", "F", "S", "T"] #Noise, Foreground, Signal, Total
    info_combination = ["non-separated"]
    FREQ = [
        '021', '025', '030', '036', '043', '052', 
        '062', '075', '090', '108', '129', '155',
        '186', '223', '268', '321', '385', '462',
        '555', '666', '799']
    misc_type = ["w"]
    simid = np.arange(-1,100)


class Asserter_smica:
    info_component = ["N", "F", "S", "T"]
    info_combination = ["non-separated", "separated", "combined"]
    FREQ = [
        '021', '025', '030', '036', '043', '052', 
        '062', '075', '090', '108', '129', '155',
        '186', '223', '268', '321', '385', '462',
        '555', '666', '799']
    misc_type = ['cov', "cov4D", "CMB", "gal_mm", "gal", "w"]
    simid = np.arange(-1,100)