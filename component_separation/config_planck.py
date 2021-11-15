from enum import Enum
import itertools
import platform
import numpy as np


class Frequency(Enum):
    LFI_1 = '030'
    LFI_2 = '044'
    LFI_3 = '070'
    HFI_1 = '100'
    HFI_2 = '143'
    HFI_3 = '217'
    HFI_4 = '353'
    HFI_5 = '545'
    HFI_6 = '857'

    
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


class Frequencyclass(Enum):
    LFI = "LFI"
    HFI = "HFI"


class Params:
    mskset = "lens"
    freqdset = "DX12"
    spectrum_type = "JC"
    lmax = 4000
    lmax_mask = 6000
    freqdatsplit = ""
    num_sim = 5
    binname = "SMICA_highell_bins"
    overwrite_cache = True
    simdata = False

    specfilter = [
        "TB",
        "EB",
        "ET",
        "BT",
        "BE"
    ]
    nside_out = [
        1024,
        2048
    ]
    nside_desc_map = [
        1024,
        2048
    ]
    outdir_ap= "/global/cscratch1/sd/sebibel/compsep/"
    powspec_truthfile= "/global/homes/s/sebibel/git/component_separation/data/powspecplanck.txt"


    # Now, generate helper parameters from these
    uname = platform.uname()
    if uname.node == "DESKTOP-KMIGUPV":
        mch = "XPS"
    else:
        mch = "NERSC"

    FREQ = [p.value for p in list(Frequency)]
    FREQ_f = [p.value for p in list(Frequency) if p.value not in ["545",
        "857"]]
    SPECTRUM = [p.value for p in list(Spectrum)]

    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
        for FREQ, FREQ2  in itertools.product(FREQ,FREQ)
            if FREQ not in ["545", "857"] and (FREQ2 not in ["545", "857"]) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in SPECTRUM if spec not in [
        "TB",
        "EB",
        "ET",
        "BT",
        "BE"
    ]]

    FREQFILTER = [
        "545",
        "857"
    ] #careful, changes must be applied manually to freqcomb in here


class NPIPE:


    @classmethod
    def _get_ddir(cls, split_loc, simid_loc):

        return "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/".format(split=split_loc)


    @classmethod
    def _get_dfn(cls, split_loc, freq_loc):

        return "npipe6v20{split}_{freq}_map.fits".format(
            split = split_loc,
            freq = freq_loc
        )


    @classmethod
    def _get_noisedir(cls, split_loc, freq_loc):

        return "INTERNAL"


    @classmethod
    def _get_noisefn(cls, split_loc, freq_loc):

        return "half_diff_npipe6v20{split}_{freq}_{nside}.fits".format(
            split = split_loc,
            freq = freq_loc,
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside
        )


    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy"


class DX12:

    nside = ['1024', '2048']

    @classmethod
    def _get_ddir(cls, split_loc, simid_loc):

        return "/global/cfs/cdirs/cmb/data/planck2018/pr3/frequencymaps/"


    @classmethod
    def _get_dfn(cls, split_loc, freq_loc):

        return "{LorH}_SkyMap_{freq}_{nside}_R3.{num}_full.fits".format(
            freq= freq_loc,
            LorH = "LFI" if int(freq_loc)<100 else "HFI",
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside,
            num = "00" if int(freq_loc)<100 else "01"
        )


    def _get_noisedir():

        return "INTERNAL"


    @classmethod
    def _get_noisefn(cls, freq_loc):

        return "{LorH}_SkyMap_{freq}_{nside}_R3.{num}_full-eohd.fits".format(
            LorH = "LFI" if int(freq_loc)<100 else "HFI",
            freq = freq_loc,
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside,
            num = "00" if int(freq_loc)<100 else "01"
        )

    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy"


class NPIPEsim:
    simid = np.concatenate((np.array(['']), np.array([str(n).zfill(4) for n in range(200)])))
    split = ['','A','B']
    freq = [
        '030',
        '044',
        '070',
        '100',
        '143',
        '217',
        '353',
        '545',
        '857'
    ]
    nside = ['1024', '2048']
    data={
        "noisefix_filename": "noisefix/noisefix_{freq}{split}_{simid}.fits",
        "order": "NESTED",
    }


    @classmethod
    def _get_ddir(cls, split_loc, simid_loc):
        assert split_loc in cls.split, "{}".format(split_loc)
        assert simid_loc in cls.simid, "{}".format(simid_loc)

        return "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}_sim/{simid}/".format(simid=simid_loc, split=split_loc)


    @classmethod
    def _get_dfn(cls, split_loc, freq_loc):
        assert split_loc in cls.split, "{}".format(split_loc)
        assert freq_loc in cls.freq, "{}".format(freq_loc)

        return "npipe6v20{split}_{freq}_map.fits".format(freq=freq_loc, split=split_loc)

    def _get_noisedir():
        # NPIPEsimdiff:    
        #     "ap": "/global/cscratch1/sd/sebibel/map/frequency/",
        #     "filename": "{simid}_half_diff_npipe6v20{split}_{freq}_{nside}.fits",
        #     "order": "NESTED",
        assert 0, "To be implemented"
        return "INTERNAL"


    @classmethod
    def _get_noisefn(cls, split_loc, freq_loc):
        assert 0, "To be implemented"

        return "half_diff_npipe6v20{split}_{freq}_{nside}.fits".format(
            split = split_loc,
            freq = freq_loc,
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside[1])


    @classmethod
    def _get_cmbdir():
        # NPIPEsimcmb:
        #     "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}_sim/{simid}/input/",
        #     "filename": "ffp10_cmb_{freq}_alm_mc_{simid}_nside{nside}_quickpol.fits",
        #     "order": "NESTED",
        assert 0, "To be implemented"
    

    def _get_cmbfn():
        assert 0, "To be implemented"


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


class BeamfDX12:

    beamf = {
        "HFI": {     
            "ap": "/global/homes/s/sebibel/data/beamf/",
            "filename": "Bl_TEB_R3.01_fullsky_{freq1}x{freq2}.fits"
        },
        "LFI": {     
            "ap": "/global/homes/s/sebibel/data/beamf/",
            "filename": "LFI_RIMO_R3.31.fits"
        },
        'info' : "DX12"
    }

    @classmethod
    def get_beamf(cls):
        return cls.beamf


class BeamfNPIPE:

    beamf = { 
        "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/quickpol/",
        "filename": "Bl_TEB_npipe6v20_{freq1}GHzx{freq2}GHz.fits",
        'info' : "NPIPE"
    }
    
    @classmethod
    def get_beamf(cls):
        return cls.beamf


class ConfXPS:
    # TODO all of it
    powspec_truthfile= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/powspecplanck.txt",
    beamf={
        "HFI": {
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/beamf/BeamWf_HFI_R3.01/",
            "filename": "Bl_TEB_R3.01_fullsky_{freq1}x{freq2}.fits"
            },
        "LFI": {
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/beamf/BeamWF_LFI/",
        "filename": "LFI_RIMO_R3.31.fits"
            }
        },
    DX12={
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/map/frequency/",
            "filename": "{LorH}_SkyMap_{freq}-field_{nside}_R3.{00/1}_full.fits"
        },
    DX12_diff={
        "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/map/frequency/",
        "filename": "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full-eohd.fits"
    },
    NPIPE_sim_diff={
        "simid": "0200"
    },
    NPIPE_sim={
        "simid": "0200"
        }
    lens={
        "tmask":{
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename": "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            },
        "pmask":{
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename": [
                "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            ]
        }
    },
    smica={
        "tmask":{
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename": "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            },
        "pmask":{
            "ap": "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename": [
                "psmaskP_2048.fits.gz",
                "gmaskP_apodized_0_2048.fits.gz"
            ]
        }
    },
    outdir_ap= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/tmp/",
    outdir_misc_ap= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/tmp/misc/",
    outdir_spectrum_ap= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/tmp/spectrum/",
    outdir_weight_ap= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/tmp/weights/",
    outdir_map_ap= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/tmp/map/frequency/",
    outdir_mask_ap= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/tmp/mask/frequency/"


class Asserter:
    info_component = ["N", "F", "S", "T"]
    info_combination = ["non-separated"]
    FREQ = ['030', '044', '070', '100', '143', '217', '353', '545', '857']
    misc_type = ["w"]