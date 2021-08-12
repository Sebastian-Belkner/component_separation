"""
io.py: Filehandling functions

"""


# '/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx

import functools
import json
import os
import copy
import platform
import sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start

import component_separation
from component_separation.cs_util import Config
from component_separation.cs_util import Planckf, Planckr, Plancks


class Filehandling:
    def __init__(self, cfmch, cfpa, nside_out, sim_id):

        self.out_misc_path = cfmch['outdir_misc_ap']
        iff_make_dir(self.out_misc_path)

        self.map_cmb_sc_path_name = self.out_misc_path + "map_cmb_in_nside_{}_sim_id_{}.npy".format(
            nside_out[1],
            sim_id)
        if "Spectrum_scale" in cfpa:
            self.total_filename = self.make_filenamestring(cfpa, sim_id)
            self.total_filename_raw = self.make_filenamestring(cfpa, sim_id, 'raw')

            self.out_map_path = cfmch['outdir_map_ap'] + cfpa["freqdset"] + "/"
            iff_make_dir(self.out_map_path)
            self.map_sc_filename = "MAP" + self.total_filename
            self.map_sc_path_name = self.out_map_path + self.map_sc_filename

            self.out_mapsyn_path = cfmch['outdir_map_ap'] + cfpa["freqdset"] + "/"
            iff_make_dir(self.out_mapsyn_path)
            self.mapsyn_sc_filename = "MAPSYN" + self.total_filename
            self.mapsyn_sc_path_name = self.out_mapsyn_path + self.mapsyn_sc_filename

            self.out_specsyn_path = cfmch['outdir_spectrum_ap'] + cfpa["freqdset"] + "/"
            iff_make_dir(self.out_specsyn_path)
            self.specsyn_sc_filename = "SPECSYN" + self.total_filename
            self.specsyn_sc_path_name = self.out_specsyn_path + self.specsyn_sc_filename

            self.out_spec_path = cfmch['outdir_spectrum_ap'] + cfpa["freqdset"] + "/"
            iff_make_dir(self.out_spec_path)

            self.spec_unsc_filename = "SPEC-RAW_" + self.total_filename_raw
            self.spec_unsc_path_name = self.out_spec_path + self.spec_unsc_filename

            self.cmb_unsc_filename = "CMB-RAW_" + self.total_filename_raw
            self.cmb_unsc_path_name = self.out_spec_path + self.cmb_unsc_filename

            self.spec_sc_filename = "SPEC" + self.total_filename
            self.spec_sc_path_name = self.out_spec_path + self.spec_sc_filename

            self.out_specsmica_path = cfmch['outdir_smica_ap'] + cfpa["freqdset"] + "/"
            iff_make_dir(self.out_specsmica_path)
            self.specsmica_sc_filename = "SPECSMICA" + "_" + cfpa['binname'] + self.total_filename
            self.specsmica_sc_path_name = self.out_specsmica_path + self.specsmica_sc_filename
            self.weight_smica_path_name = cfmch['outdir_smica_ap'] + "SMICAWEIG_" + cfpa["Tscale"] + "_" + cfpa['binname'] + self.total_filename

            self.cmbmap_smica_path_name = cfmch['outdir_smica_ap'] + "smicaminvarmap_{}".format(cfpa['binname']) + "_" + self.total_filename
            self.clmin_smica_path_name = cfmch['outdir_smica_ap'] + "smicaclmin_{}".format(cfpa['binname']) + "_" + self.total_filename
            self.cmb_specsmica_sc_path_name = self.out_specsmica_path + "CMB_" + self.specsmica_sc_filename
            self.gal_specsmica_sc_path_name = self.out_specsmica_path + "GAL_" + self.specsmica_sc_filename

            self.weight_path = cfmch['outdir_weight_ap'] + cfpa["freqdset"] + "/"
            iff_make_dir(self.weight_path)
            self.weight_path_name = self.weight_path + "WEIG_" + cfpa["Tscale"] + "_" + self.total_filename

            self.signal_sc_filename = "C_lS_in_sim_id_{}.npy".format(sim_id)
            self.signal_sc_path_name = self.out_misc_path + self.signal_sc_filename
            #TODO the following part needs reviewin

            cfmch_copy = copy.deepcopy(cfmch)
            cfpa_copy = copy.deepcopy(cfpa)

            ### the following lines are only needed for run_smica part of the code
            buff = cfpa['freqdset']
            if "diff" in buff or 'cmb' in buff:
                pass
            else:
                cfpa_copy['freqdset'] = buff+'_diff'
            self.noise_filename = self.make_filenamestring(cfpa_copy, sim_id)
            self.noise_filename_raw = self.make_filenamestring(cfpa_copy, sim_id, 'raw')
            self.noise_path = cfmch_copy['outdir_spectrum_ap'] + cfpa_copy["freqdset"] + "/"

            iff_make_dir(self.noise_path)
            self.noise_unsc_path_name = self.noise_path + 'SPEC-RAW_' + self.noise_filename_raw
            self.noise_sc_path_name = self.noise_path + "SPEC" + self.noise_filename

            cfpa_copy['freqdset'] = buff
    

        
    def make_filenamestring(self, cfpa_local, sim_id, desc='scaled'):
        """Helper function for generating unique filenames given te current configuration

        Args:
            cf (Dict): Configuration file - in general conf.json from root directory

        Returns:
            str: unique filename which may be used for spectra, weights, maps, etc..
        """

        spectrum_scale = cfpa_local["Spectrum_scale"]
        mskset = cfpa_local['mskset'] # smica or lens
        freqdset = cfpa_local['freqdset'] # DX12 or NERSC
        lmax = cfpa_local["lmax"]
        lmax_mask = cfpa_local["lmax_mask"]

        smoothing_window = cfpa_local["smoothing_window"]
        max_polynom = cfpa_local["max_polynom"]

        if desc == 'raw':
            return '{sim_id}_{spectrum_scale}_{freqdset}_{mskset}_{lmax}_{lmax_mask}_{split}.npy'.format(
                sim_id = sim_id,
                spectrum_scale = spectrum_scale,
                freqdset = freqdset,
                mskset = mskset,
                lmax = lmax,
                lmax_mask = lmax_mask,
                split = "Full" if cfpa_local["freqdatsplit"] == "" else cfpa_local["freqdatsplit"])
        else:
            return '{sim_id}_{spectrum_scale}_{freqdset}_{mskset}_{lmax}_{lmax_mask}_{smoothing_window}_{max_polynom}_{split}.npy'.format(
                sim_id = sim_id,
                spectrum_scale = spectrum_scale,
                freqdset = freqdset,
                mskset = mskset,
                lmax = lmax,
                lmax_mask = lmax_mask,
                split = "Full" if cfpa_local["freqdatsplit"] == "" else cfpa_local["freqdatsplit"],
                smoothing_window = smoothing_window,
                max_polynom = max_polynom)


class IO:
    def __init__(self, csu):
        ##TODO improve path management
        self.fh = Filehandling(csu.cf[csu.mch], csu.cf['pa'], csu.nside_out, csu.sim_id)
        self.csu = csu


    def load_powerspectra(self, dset, processed = True):
        if processed:
            if dset == 'noise':
                path_name = self.fh.noise_sc_path_name
            elif dset == 'full':
                path_name = self.fh.spec_sc_path_name
            elif dset == 'signal':
                path_name = self.fh.signal_sc_path_name
        else:
            if dset == 'noise':
                path_name = self.fh.noise_unsc_path_name
            elif dset == 'full':
                path_name = self.fh.spec_unsc_path_name
            elif dset == 'signal':
                path_name = self.fh.signal_unsc_path_name
        C_l = self.load_data(path_name=path_name)
        if C_l is None:
            print("couldn't find spectrum with given specifications at {}.".format(path_name))
            sys.exit()
        return C_l


    def read_pf(mask_path, mask_filename):
        return {FREQ: hp.read_map(
            '{mask_path}{mask_filename}'
            .format(
                mask_path = mask_path,
                mask_filename = mask_filename
                    .replace("{freqdset}", freqdset)
                    .replace("{freq}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                ), dtype=np.bool)
                for FREQ in PLANCKMAPFREQ_f
            }


    
    def load_data(self, path_name: str) -> Dict[str, Dict]:
        if os.path.isfile(path_name):
            data = np.load(path_name, allow_pickle=True)
            print('loaded {}'.format(path_name))
            if data.shape == ():
                return data.item()
            else:
                return data
        else:
            print("no existing data at {}".format(path_name))
            return None


    def load_plamap(self, cf_local, field, nside_out=None):
        """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
        Map data in `PATH/map/`.
        Args:
            cf (Dict): Configuration as coming from conf.json
            mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
            
        Returns:
            List[Dict]: Planck maps (data and masks) and some header information

        Doctest:
        >>> get_data(cf: Dict, mch: str) 
        NotSureWhatToExpect
        """

        freqdset = cf_local["pa"]['freqdset'] # NPIPE or DX12
        freqfilter = cf_local["pa"]["freqfilter"]
        nside_desc = cf_local["pa"]["nside_desc_map"]
        mch = self.csu.mch

        abs_path = cf_local[self.csu.mch][freqdset]['ap']
        freq_filename = cf_local[self.csu.mch][freqdset]['filename']
        mappath = {
            FREQ:'{abs_path}{freq_filename}'
                .format(
                    abs_path = abs_path\
                        .replace("{sim_id}", self.csu.sim_id)\
                        .replace("{split}", cf_local['pa']['freqdatsplit'] if "split" in cf_local[mch][freqdset] else ""),
                    freq_filename = freq_filename
                        .replace("{freq}", FREQ)
                        .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                        .replace("{nside}", str(nside_desc[0]) if int(FREQ)<100 else str(nside_desc[1]))
                        .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                        .replace("{even/half1}", "evenring" if int(FREQ)>=100 else "ringhalf-1")
                        .replace("{odd/half2}", "oddring" if int(FREQ)>=100 else "ringhalf-2")
                        .replace("{sim_id}", self.csu.sim_id)\
                        .replace("{split}", cf_local['pa']['freqdatsplit'] if "split" in cf_local[mch][freqdset] else "")
                    )
                for FREQ in self.csu.PLANCKMAPFREQ
                if FREQ not in freqfilter}

        maps = {
            FREQ: hp.ud_grade(hp.read_map(mappath[FREQ], field=field, dtype=np.float64, nest=False), nside_out=nside_out[0] if int(FREQ)<100 else nside_out[1])
                for FREQ in self.csu.PLANCKMAPFREQ
                if FREQ not in freqfilter
        }
        print("loaded {}".format(mappath))
        return maps


    def load_mask_per_freq(dg_to=1024):
        def _multi(a, b):
            return a*b
        maskset = cf['pa']['mskset']
        freqfilter = cf['pa']["freqfilter"]

        pmask_path = cf[mch][maskset]['pmask']["ap"]
        pmask_filename = cf[mch][maskset]['pmask']['filename']
        pmasks = [
            read_pf(pmask_path, a)
            for a in pmask_filename]
        pmask = {
            FREQ: functools.reduce(
                _multi,
                [a[FREQ] for a in pmasks])
                    for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter}

        pmask = {FREQ: hp.pixelfunc.ud_grade(pmask[FREQ], nside_out=dg_to if int(FREQ)<100 else 2048)
                for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
            }
        
        tmask_path = cf[mch][maskset]['tmask']["ap"]
        tmask_filename = cf[mch][maskset]['tmask']['filename']
        tmask = read_pf(tmask_path, tmask_filename)
        tmask = {FREQ: hp.pixelfunc.ud_grade(tmask[FREQ], nside_out=dg_to)
                        for FREQ in PLANCKMAPFREQ
                        if FREQ not in freqfilter
                    }
        return tmask, pmask, pmask


    def load_one_mask_forallfreq(self, nside_out=None):
        def _multi(a, b):
            return a*b
        def read_single(mask_path, mask_filename):
            return hp.read_map(
                '{mask_path}{mask_filename}'
                .format(
                    mask_path = mask_path,
                    mask_filename = mask_filename), dtype=np.bool)
        if nside_out == None:
            nside_out = self.csu.nside_out
        maskset = self.csu.cf['pa']['mskset']
        if maskset == None:
            tmask = np.ones(shape=hp.nside2npix(nside_out[1]))
            pmask = np.ones(shape=hp.nside2npix(nside_out[1]))
        else:    
            mch = self.csu.mch
            pmask_path = self.csu.cf[mch][maskset]['pmask']["ap"]
            pmask_filename = self.csu.cf[mch][maskset]['pmask']['filename']
            print('loading mask {}'.format(pmask_filename))
            pmasks = [
                read_single(pmask_path, a)
                for a in pmask_filename]
            pmask = functools.reduce(
                _multi,
                [a for a in pmasks])

            tmask_path = self.csu.cf[mch][maskset]['tmask']["ap"]
            tmask_filename = self.csu.cf[mch][maskset]['tmask']['filename']
            tmask = read_single(tmask_path, tmask_filename)

            if nside_out is not None:
                tmask = hp.ud_grade(tmask, nside_out=nside_out[1])
                pmask = hp.ud_grade(pmask, nside_out=nside_out[1])
        tdict = {FREQ: tmask
                    for FREQ in self.csu.PLANCKMAPFREQ_f
                    }
        pdict = {FREQ: pmask
                for FREQ in self.csu.PLANCKMAPFREQ_f
                }
        return tdict, pdict, pdict


    def load_truthspectrum(abs_path=""):
        return pd.read_csv(
            abs_path+cf[mch]['powspec_truthfile'],
            header=0,
            sep='    ',
            index_col=0)


    @log_on_start(INFO, "Starting to load weights from {path_name}")
    @log_on_end(DEBUG, "Weights loaded successfully")
    def load_weights(path_name: str, indir_root: str = None, indir_rel: str = None, in_desc: str = None, fname: str = None) -> Dict[str, Dict]:
        if path_name == None:
            fending = ".npy"
            path_name = indir_root+indir_rel+in_desc+fname+fending
        if os.path.isfile(path_name):
            data = np.load(path_name, allow_pickle=True)
            print( "loaded {}".format(path_name))
            return data
        else:
            print("no existing weights at {}".format(path_name))
            return None


    @log_on_start(INFO, "Starting to load beamf functions from frequency channels {freqcomb}")
    @log_on_end(DEBUG, "Beamfunction(s) loaded successfully")
    def load_beamf(self, freqcomb: List) -> Dict:
        """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

        Args:
            cf (Dict): Configuration as coming from conf.json
            mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
            freqcomb (List): Frequency channels which are to be ignored

        Returns:
            np.array: Planck beamfunctions with dimension (Nspec, nfreq, nfreq, lmaxp1)
        """
        beamf = dict()
        cf = self.csu.cf
        mch = self.csu.mch
        split = self.csu.cf['pa']['split']
        splitvariation = self.csu.cf['pa']['splitvariation']
        if self.csu.cf['pa']['freqdset'].startswith('DX12'):
            dset = 'DX12'
        elif self.csu.cf['pa']['freqdset'].startswith('NPIPE'):
            dset = 'NPIPE'

        if self.csu.cf['pa']['freqdset'].startswith('DX12'):
            for freqc in freqcomb:
                freqs = freqc.split('-')
                if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                    beamf.update({
                        freqc: {
                            "HFI": fits.open(
                                "{bf_path}{bf_filename}"
                                .format(
                                    bf_path = cf[mch]["beamf"][dset]["HFI"]['ap'].replace("{split}", split),
                                    bf_filename = cf[mch]["beamf"][dset]["HFI"]['filename']
                                        .replace("{freq1}", freqs[0])
                                        .replace("{freq2}", freqs[1])
                                        .replace("{splitvariation}", splitvariation)
                                ))
                            }
                        })
                elif int(freqs[0]) < 100 and int(freqs[1]) >= 100:
                    beamf.update({
                        freqc: {
                            "HFI": fits.open(
                                "{bf_path}{bf_filename}"
                                .format(
                                    bf_path = cf[mch]["beamf"][dset]["HFI"]['ap'].replace("{split}", split),
                                    bf_filename = cf[mch]["beamf"][dset]["HFI"]['filename']
                                        .replace("{freq1}", freqs[1])
                                        .replace("{freq2}", freqs[1])
                                        .replace("{splitvariation}", splitvariation)
                            ))
                        }
                    })
                    beamf[freqc].update({
                        "LFI": fits.open(
                                "{bf_path}{bf_filename}"
                                .format(
                                    bf_path = cf[mch]["beamf"][dset]["LFI"]['ap'].replace("{split}", split),
                                    bf_filename = cf[mch]["beamf"][dset]["LFI"]['filename']
                                        .replace("{splitvariation}", splitvariation)
                            ))
                    })
                if int(freqs[0]) < 100 and int(freqs[1]) < 100:
                    beamf.update({
                        freqc: {
                            "LFI": fits.open(
                                "{bf_path}{bf_filename}"
                                .format(
                                    bf_path = cf[mch]["beamf"][dset]["LFI"]['ap'].replace("{split}", split),
                                    bf_filename = cf[mch]["beamf"][dset]["LFI"]['filename']
                                        .replace("{splitvariation}", splitvariation)
                            ))
                        }})
        elif cf['pa']['freqdset'].startswith('NPIPE'):
            for freqc in freqcomb:
                freqs = freqc.split('-')
                beamf.update({
                    freqc: {
                        "HFI": fits.open(
                            "{bf_path}{bf_filename}"
                            .format(
                                bf_path = cf[mch]["beamf"][dset]["HFI"]['ap'].replace("{split}", split),
                                bf_filename = cf[mch]["beamf"][dset]["HFI"]['filename']
                                    .replace("{freq1}", freqs[0])
                                    .replace("{freq2}", freqs[1])
                                    .replace("{splitvariation}", splitvariation)
                            ))
                        }
                    })
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
        freqs = self.csu.PLANCKMAPFREQ_f
        lmaxp1 = self.csu.lmax+1
        beamf_array = np.zeros(shape=(3, len(freqs), len(freqs), lmaxp1))
        
        for idspec, spec in enumerate(["T", "E", "B"]):
            for ida, freqa in enumerate(freqs):
                for idb, freqb in enumerate(freqs):
                    if ida < idb:
                        bf = beamf[freqa+'-'+freqb]
                    else:
                        bf = beamf[freqb+'-'+freqa]
                    if self.csu.cf['pa']['freqdset'].startswith('DX12'):
                        if int(freqa) >= 100 and int(freqb) >= 100:
                            beamf_array[idspec,ida,idb] = bf["HFI"][1].data.field(TEB_dict[spec])[:lmaxp1]
                        elif int(freqa) < 100 and int(freqb) < 100:
                            b1 = np.sqrt(bf["LFI"][LFI_dict[freqa]].data.field(0))
                            buff1 = np.concatenate((
                                b1[:min(lmaxp1, len(b1))],
                                np.array([np.NaN for n in range(max(0, lmaxp1-len(b1)))])))
                            b2 = np.sqrt(bf["LFI"][LFI_dict[freqb]].data.field(0))
                            buff2 = np.concatenate((
                                b2[:min(lmaxp1, len(b2))],
                                np.array([np.NaN for n in range(max(0, lmaxp1-len(b2)))])))
                            beamf_array[idspec,ida,idb] = buff1*buff2
                        else:
                            if ida < idb:
                                freqc = freqa
                            else:
                                freqc = freqb
                            b = np.sqrt(bf["LFI"][LFI_dict[freqc]].data.field(0))
                            buff = np.concatenate((
                                b[:min(lmaxp1, len(b))],
                                np.array([np.NaN for n in range(max(0, lmaxp1-len(b)))])))
                            beamf_array[idspec,ida,idb] = buff*np.sqrt(bf["HFI"][1].data.field(TEB_dict[spec])[:lmaxp1])
                    elif cf['pa']['freqdset'].startswith('NPIPE'):
                        ### now that all cross beamfunctions exist, and beamf
                        ### files have the same structure, no difference between applying lfis and hfis anymore
                        beamf_array[idspec,ida,idb] = bf["HFI"][1].data.field(TEB_dict[spec])[:lmaxp1]
        return beamf_array
        


    @log_on_start(INFO, "Saving to {path_name}")
    @log_on_end(DEBUG, "Data saved successfully to {path_name}")
    def save_data(self, data, path_name: str, filename: str = 'default'):
        if os.path.exists(path_name):
            os.remove(path_name)
        np.save(path_name, data)
        print('Data saved to {}'.format(path_name))


    @log_on_start(INFO, "Saving to {path_name}")
    @log_on_end(DEBUG, "Data saved successfully to {path_name}")
    def save_figure(mp, path_name: str, outdir_root: str = None, outdir_rel: str = None, out_desc: str = None, fname: str = None):
        if path_name == None:
            fending = ".jpg"
            path_name = outdir_root+outdir_rel+out_desc+fname+fending
        mp.savefig(path_name, dpi=144)
        mp.close()


    @log_on_start(INFO, "Saving to {path_name}")
    @log_on_end(DEBUG, "Data saved successfully to {path_name}")
    def save_map(self, data, path_name: str):
        hp.write_map(path_name, data, overwrite=True)
        print("saved map to {}".format(path_name))
    
    
def iff_make_dir(outpath_name):
    if os.path.exists(outpath_name):
        pass
    else:
        os.makedirs(outpath_name)

def alert_cached(func):
    def wrapped(*args):
        if os.path.isfile(args[0]):
            print('Output file {} already exists. Overwrite settings are set to {}'.format(args[0], args[1]))
            if args[1]:
                print('Overwriting cache')
            else:
                print('Exiting..')
                sys.exit()
        return func(*args)
    return wrapped


