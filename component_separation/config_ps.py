class Params:
    Tscale = "K_CMB"
    mskset = "smica"
    freqdset = "DX12"
    Spectrum_scale = "C_l"
    spectrum_type = "Chonetal"
    lmax = 4000
    lmax_mask = 6000
    freqdatsplit = ""
    smoothing_window = 0
    max_polynom = 0
    num_sim = 5
    binname = "SMICA_highell_bins"
    overwrite_cache = True
    split = ""
    nside_out = None
    splitvariation = "GHz"

    freqfilter = [
        "030",
        "044",
        "100",
        "217",
        "353",
        "545",
        "857"
    ]
    specfilter = [
        "TB",
        "EB",
        "ET",
        "BT",
        "BE"
    ]
    nside_desc_map = [
        1024,
        2048
    ]
    outdir_ap= "/global/cscratch1/sd/sebibel/"
    outdir_misc_ap= "/global/cscratch1/sd/sebibel/misc/"
    outdir_smica_ap= "/global/cscratch1/sd/sebibel/smica/"
    outdir_spectrum_ap= "/global/cscratch1/sd/sebibel/spectrum/"
    outdir_weight_ap= "/global/cscratch1/sd/sebibel/weights/"
    outdir_map_ap= "/global/cscratch1/sd/sebibel/map/frequency/"
    outdir_mask_ap= "/global/cscratch1/sd/sebibel/mask/frequency/"


class NPIPE:
    beamf={
        "HFI": {     
        "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/quickpol/",
        "filename": "Bl_TEB_npipe6v20_{freq1}{splitvariation}x{freq2}{splitvariation}.fits"
        }
    }
    powspec_truthfile= "/global/homes/s/sebibel/git/component_separation/data/powspecplanck.txt",
    data = {
            "ap"= "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}/",
            "filename"= "npipe6v20{split}_{freq}_map.fits",
            "split"= [
                "A",
                "B",
                ""
            ]
        }

class DX12:
    beamf = {
        "HFI": {     
            "ap": "/global/homes/s/sebibel/data/beamf/",
            "filename": "Bl_TEB_R3.01_fullsky_{freq1}x{freq2}.fits"
        },
        "LFI": {     
            "ap": "/global/homes/s/sebibel/data/beamf/",
            "filename": "LFI_RIMO_R3.31.fits"
        }
    }
    data = { 
            "ap": "/global/cfs/cdirs/cmb/data/planck2018/pr3/frequencymaps/",
            "filename": "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full.fits"
        }


class DX12_diff:
    data ={ 
            "ap": "/global/cscratch1/sd/sebibel/map/frequency/",
            "filename": "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full-eohd.fits"
        }
      

class NPIPE_diff:
    data = {
        "ap": "/global/cscratch1/sd/sebibel/map/frequency/",
        "filename": "half_diff_npipe6v20_{freq}_{nside}.fits",
        "order": "NESTED"
    }


class NPIPE_sim:
    data={
        "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}_sim/{sim_id}/",
        "filename": "npipe6v20{split}_{freq}_map.fits",
        "noisefix_filename": "noisefix/noisefix_{freq}{split}_{sim_id}.fits",
        "order": "NESTED",
        "sim_id": "0200",
        "split": [
            "A",
            "B",
            ""
        ]
    }


class NPIPE_sim_diff:    
    data={
        "ap": "/global/cscratch1/sd/sebibel/map/frequency/",
        "filename": "{sim_id}_half_diff_npipe6v20{split}_{freq}_{nside}.fits",
        "order": "NESTED",
        "sim_id": "0200",
        "split": [
            "A",
            "B",
            ""
        ]
    }
        
class NPIPE_sim_cmb:
    data = {
        "ap": "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20{split}_sim/{sim_id}/input/",
        "filename": "ffp10_cmb_{freq}_alm_mc_{sim_id}_nside{nside}_quickpol.fits",
        "order": "NESTED",
        "sim_id": "0200",
        "split": [
            "A",
            "B",
            ""
        ]
    }

class Mask:
    lens={
        "tmask": {
            "ap": "/global/homes/s/sebibel/data/mask/",
            "filename": "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            },
        "pmask": {
            "ap": "/global/homes/s/sebibel/data/mask/",
            "filename": [
                "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            ]
        }
    }
    smica={
        "tmask": {
            "ap"= "/global/homes/s/sebibel/data/mask/",
            "filename"= "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            },
        "pmask" :{
            "ap"= "/global/homes/s/sebibel/data/mask/",
            "filename"= [
                "psmaskP_2048.fits.gz",
                "gmaskP_apodized_0_2048.fits.gz"
                ]
            }
        }

class ConfXPS:
    powspec_truthfile= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/powspecplanck.txt",
    beamf={
        "HFI"= {
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/beamf/BeamWf_HFI_R3.01/",
            "filename"= "Bl_TEB_R3.01_fullsky_{freq1}x{freq2}.fits"
            },
        "LFI"= {
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/beamf/BeamWF_LFI/",
        "filename"= "LFI_RIMO_R3.31.fits"
            }
        },
    DX12"={
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/map/frequency/",
            "filename"= "{LorH}_SkyMap_{freq}-field_{nside}_R3.{00/1}_full.fits"
        },
    DX12_diff"={
        "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/map/frequency/",
        "filename"= "{LorH}_SkyMap_{freq}_{nside}_R3.{00/1}_full-eohd.fits"
    },
    NPIPE_sim_diff"={
        "sim_id"= "0200"
    },
    NPIPE_sim"={
        "sim_id"= "0200"
        },
    lens"={
        "tmask"={
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename"= "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            },
        "pmask"={
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename"= [
                "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            ]
        }
    },
    smica={
        "tmask"={
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename"= "PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
            },
        "pmask"={
            "ap"= "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/mask/",
            "filename"= [
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
