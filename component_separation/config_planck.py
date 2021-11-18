from enum import Enum
import itertools
import platform
import numpy as np
import os


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
    TT = "TT"#00 # the order must be the same as the order of pospace function returns
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
    mskset = "smica" # smica or lens
    freqdset = "DX12" # DX12 or NPIPE
    spectrum_type = "JC" # pseudo or JC
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
    outdir_ap= "/global/cscratch1/sd/sebibel/compsep/planck/"
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
    ] #careful, changes in here must be applied manually to freqcomb etc.


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
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside[1]
        )


    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/planck/Sest/ClS_NPIPEsim.npy"


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
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside[1],
            num = "00" if int(freq_loc)<100 else "01"
        )


    def _get_noisedir():

        return "INTERNAL"


    @classmethod
    def _get_noisefn(cls, freq_loc):

        return "{LorH}_SkyMap_{freq}_{nside}_R3.{num}_full-eohd.fits".format(
            LorH = "LFI" if int(freq_loc)<100 else "HFI",
            freq = freq_loc,
            nside = cls.nside[0] if int(freq_loc)<100 else cls.nside[1],
            num = "00" if int(freq_loc)<100 else "01"
        )

    def _get_signalest():

        return "/global/cscratch1/sd/sebibel/compsep/planck/Sest/ClS_NPIPEsim.npy"


class NPIPEsim:
    simid = np.concatenate((np.array(['']), np.array([str(n).zfill(4) for n in range(200)])))
    split = ['','A','B']
    freq = ['030','044', '070',  '100',  '143',  '217',  '353',  '545',  '857']
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


    @classmethod
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
    beamf_info = {
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
    freqdatsplit = ['','A','B']


    @classmethod
    def get_beaminfo(cls):

        return cls.beamf_info


    @classmethod
    def get_beamf(cls, fits, freqcomb, lmax, freqdatsplit_loc='') -> np.array:
        """Read files and return all auto- and cross-bl in np.array 

        Returns:
            np.array: [description]
        """
        indfreq = np.array([])
        beamf = dict()
        for freqc in freqcomb:
            freqs = freqc.split('-')
            indfreq = np.concatenate((indfreq, [freqs[0]]))
            if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                beamf.update({
                    freqc: {
                        "HFI": fits.open(
                            "{bf_path}{bf_filename}"
                            .format(
                                bf_path = cls.beamf_info["HFI"]['ap'].replace("{split}", freqdatsplit_loc),
                                bf_filename = cls.beamf_info["HFI"]['filename']
                                    .replace("{freq1}", freqs[0])
                                    .replace("{freq2}", freqs[1])
                            ))
                        }
                    })
            elif int(freqs[0]) < 100 and int(freqs[1]) >= 100:
                beamf.update({
                    freqc: {
                        "HFI": fits.open(
                            "{bf_path}{bf_filename}"
                            .format(
                                bf_path = cls.beamf_info["HFI"]['ap'].replace("{split}", freqdatsplit_loc),
                                bf_filename = cls.beamf_info["HFI"]['filename']
                                    .replace("{freq1}", freqs[1])
                                    .replace("{freq2}", freqs[1])
                        ))
                    }
                })
                beamf[freqc].update({
                    "LFI": fits.open(
                            "{bf_path}{bf_filename}"
                            .format(
                                bf_path = cls.beamf_info["LFI"]['ap'].replace("{split}", freqdatsplit_loc),
                                bf_filename = cls.beamf_info["LFI"]['filename']
                        ))
                })
            elif int(freqs[0]) < 100 and int(freqs[1]) < 100:
                beamf.update({
                    freqc: {
                        "LFI": fits.open(
                            "{bf_path}{bf_filename}"
                            .format(
                                bf_path = cls.beamf_info["LFI"]['ap'].replace("{split}", freqdatsplit_loc),
                                bf_filename = cls.beamf_info["LFI"]['filename']
                        ))
                    }})
        lmaxp1 = lmax+1
        indfreq_srt = sorted(set(indfreq))
        beamf_array = np.zeros(shape=(3, len(indfreq_srt), len(indfreq_srt), lmaxp1))
        for idspec, spec in enumerate(["T", "E", "B"]):
            for ida, freqa in enumerate(indfreq_srt):
                for idb, freqb in enumerate(indfreq_srt):
                    if ida < idb:
                        bf = beamf[freqa+'-'+freqb]
                    else:
                        bf = beamf[freqb+'-'+freqa]
                    
                    if int(freqa) >= 100 and int(freqb) >= 100:
                        beamf_array[idspec,ida,idb] = bf["HFI"][1].data.field(cls.TEB_dict[spec])[:lmaxp1]
                    elif int(freqa) < 100 and int(freqb) < 100:
                        b1 = np.sqrt(bf["LFI"][cls.LFI_dict[freqa]].data.field(0))
                        buff1 = np.concatenate((
                            b1[:min(lmaxp1, len(b1))],
                            np.array([np.NaN for n in range(max(0, lmaxp1-len(b1)))])))
                        b2 = np.sqrt(bf["LFI"][cls.LFI_dict[freqb]].data.field(0))
                        buff2 = np.concatenate((
                            b2[:min(lmaxp1, len(b2))],
                            np.array([np.NaN for n in range(max(0, lmaxp1-len(b2)))])))
                        beamf_array[idspec,ida,idb] = buff1*buff2
                    else:
                        if ida < idb:
                            freqc = freqa
                        else:
                            freqc = freqb
                        b = np.sqrt(bf["LFI"][cls.LFI_dict[freqc]].data.field(0))
                        buff = np.concatenate((
                            b[:min(lmaxp1, len(b))],
                            np.array([np.NaN for n in range(max(0, lmaxp1-len(b)))])))
                        beamf_array[idspec,ida,idb] = buff*np.sqrt(bf["HFI"][1].data.field(cls.TEB_dict[spec])[:lmaxp1])
        return beamf_array


class BeamfNPIPE:
    split = ['','A','B']
    beamf_info = { 
        "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/quickpol/",
        "filename": "Bl_TEB_npipe6v20_{freq1}GHzx{freq2}GHz.fits",
        'info' : "NPIPE"
    }
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
    

    @classmethod
    def get_beaminfo(cls):

        return cls.beamf_info


    @classmethod
    def get_beamf(cls, fits, freqcomb, lmax, freqdatsplit_loc):

        indfreq = np.array([])
        lmaxp1 = lmax+1
        beamf = dict()
        for freqc in freqcomb:
            freqs = freqc.split('-')
            indfreq = np.concatenate((indfreq, [freqs[0]]))
            beamf.update({
                freqc: fits.open(
                    os.path.join(
                        cls.beamf_info['ap'].format(split=freqdatsplit_loc), 
                        cls.beamf_info['filename'].format(freq1=freqs[0], freq2=freqs[1])
            ))})
        indfreq_srt = sorted(set(indfreq))
        beamf_array = np.zeros(shape=(3, len(indfreq_srt), len(indfreq_srt), lmaxp1))
        for idspec, spec in enumerate(["T", "E", "B"]):
            for ida, freqa in enumerate(indfreq_srt):
                for idb, freqb in enumerate(indfreq_srt):
                    if ida < idb:
                        bf = beamf[freqa+'-'+freqb]
                    else:
                        bf = beamf[freqb+'-'+freqa]
                    beamf_array[idspec,ida,idb] = bf["HFI"][1].data.field(cls.TEB_dict[spec])[:lmaxp1]
        return cls.beamf_info


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
    info_component = ["N", "F", "S", "T"] #Noise, Foreground, Signal, Total
    info_combination = ["non-separated"]
    FREQ = ['030', '044', '070', '100', '143', '217', '353', '545', '857']
    misc_type = ["w"]
    simid = np.arange(-1,500)


class Asserter_smica:
    info_component = ["N", "F", "S", "T"]
    info_combination = ["non-separated", "separated", "combined"]
    FREQ = ['030', '044', '070', '100', '143', '217', '353', '545', '857']
    misc_type = ['cov', "cov4D", "CMB", "gal_mm", "gal", "w"]
    simid = np.arange(-1,100)