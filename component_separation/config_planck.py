from enum import Enum
import itertools
import platform


class Planckf(Enum):
    LFI_1 = '030'
    LFI_2 = '044'
    LFI_3 = '070'
    HFI_1 = '100'
    HFI_2 = '143'
    HFI_3 = '217'
    HFI_4 = '353'
    HFI_5 = '545'
    HFI_6 = '857'

    
class Plancks(Enum):
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


class Planckr(Enum):
    LFI = "LFI"
    HFI = "HFI"


class Params:
    mskset = "lens"
    freqdset = "DX12"
    spectrum_type = "pseudo"
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

    PLANCKMAPFREQ = [p.value for p in list(Planckf)]
    PLANCKMAPFREQ_f = [p.value for p in list(Planckf) if p.value not in ["545",
        "857"]]
    PLANCKSPECTRUM = [p.value for p in list(Plancks)]

    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
        for FREQ, FREQ2  in itertools.product(PLANCKMAPFREQ,PLANCKMAPFREQ)
            if FREQ not in ["545", "857"] and (FREQ2 not in ["545", "857"]) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in [
        "TB",
        "EB",
        "ET",
        "BT",
        "BE"
    ]]

    freqfilter = [
        "545",
        "857"
    ] #careful, changes must be applied manually to freqcomb in here


class NPIPE:

    def _get_pladir():

        return "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/"


    def _get_plafn():

        return "npipe6v20{split}_{freq}_map.fits"


    def _get_noisedir():

        return "INTERNAL"


    def _get_noisefn():

        return "half_diff_npipe6v20{split}_{freq}_{nside}.fits"


    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy"



class DX12:

    def _get_pladir():

        return "/global/cfs/cdirs/cmb/data/planck2018/pr3/frequencymaps/"


    def _get_plafn():

        return "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full.fits"


    def _get_noisedir():

        return "INTERNAL"


    def _get_noisefn():

        return "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full-eohd.fits"


    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/Sest/ClS_NPIPEsim.npy"


class NPIPEsim:

    data={
        "noisefix_filename": "noisefix/noisefix_{freq}{split}_{sim_id}.fits",
        "order": "NESTED",
    }

    def _get_pladir():

        return "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}_sim/{sim_id}/"


    def _get_plafn():

        return "npipe6v20{split}_{freq}_map.fits"


    def _get_noisedir():
        # NPIPEsimdiff:    
        #     "ap": "/global/cscratch1/sd/sebibel/map/frequency/",
        #     "filename": "{sim_id}_half_diff_npipe6v20{split}_{freq}_{nside}.fits",
        #     "order": "NESTED",
        assert 0, "To be implemented"
        return "INTERNAL"


    def _get_noisefn():
        assert 0, "To be implemented"
        return "half_diff_npipe6v20{split}_{freq}_{nside}.fits"


    def _get_cmbdir():
        # NPIPEsimcmb:
        #     "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}_sim/{sim_id}/input/",
        #     "filename": "ffp10_cmb_{freq}_alm_mc_{sim_id}_nside{nside}_quickpol.fits",
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
        "sim_id": "0200"
    },
    NPIPE_sim={
        "sim_id": "0200"
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

