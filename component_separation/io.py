"""
io.py: Filehandling functions

"""

import functools
import os, sys
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start

class IO:
    def __init__(self, csu, fn=None):
        self.csu = csu
        self.fn = fn


    def load_powerspectra(self, dset, processed = True):
        if self.csu.spectrum_type == 'JC':
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
        elif self.csu.spectrum_type == 'pseudo':
            if processed:
                if dset == 'noise':
                    path_name = self.fh.psnoise_sc_path_name
                elif dset == 'full':
                    path_name = self.fh.psspec_sc_path_name
                elif dset == 'signal':
                    path_name = self.fh.signal_sc_path_name
            else:
                if dset == 'noise':
                    path_name = self.fh.psnoise_unsc_path_name
                elif dset == 'full':
                    path_name = self.fh.psspec_unsc_path_name
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


    def load_plamap(self, cf_loc, field, nside_out=None):
        """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
        Map data in `PATH/map/`.
        Args:
            cf (Dict): Configuration as coming from conf.json
            mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
            
        Returns:
            List[Dict]: Planck maps (data and masks) and some header information

        """

        freqdset = cf_loc["pa"]['freqdset'] # NPIPE or DX12
        freqfilter = cf_loc["pa"]["freqfilter"]
        nside_desc = cf_loc["pa"]["nside_desc_map"]
        mch = self.csu.mch

        abs_path = cf_loc[mch][freqdset]['ap']
        freq_filename = cf_loc[mch][freqdset]['filename']
        mappath = {
            FREQ:'{abs_path}{freq_filename}'
                .format(
                    abs_path = abs_path\
                        .replace("{sim_id}", self.csu.sim_id)\
                        .replace("{split}", cf_loc['pa']['freqdatsplit'] if "split" in cf_loc[mch][freqdset] else ""),
                    freq_filename = freq_filename
                        .replace("{freq}", FREQ)
                        .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                        .replace("{nside}", str(nside_desc[0]) if int(FREQ)<100 else str(nside_desc[1]))
                        .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                        .replace("{even/half1}", "evenring" if int(FREQ)>=100 else "ringhalf-1")
                        .replace("{odd/half2}", "oddring" if int(FREQ)>=100 else "ringhalf-2")
                        .replace("{sim_id}", self.csu.sim_id)\
                        .replace("{split}", cf_loc['pa']['freqdatsplit'] if "split" in cf_loc[mch][freqdset] else "")
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


    def load_pla(self, pathname, field, ud_grade=(None, None)):
        """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
        Map data in `PATH/map/`.
        Args:
            pathname (str): complete name to the file
            field (tuple): data fields to be read
            ud_grade (tuple): First element is None or True, second element is the freq-split where either ud_grade nside is applied
            
        Returns:
            List[Dict]: Planck maps (data and masks) and some header information

        """
        if ud_grade[0] is None:
            return hp.read_map(
                pathname,
                field=field,
                dtype=np.float64,
                nest=False)
        else:
            return hp.ud_grade(
                hp.read_map(
                    pathname,
                    field=field,
                    dtype=np.float64,
                    nest=False),
                nside_out=self.csu.nside_out[0] if int(ud_grade[1])<100 else self.csu.nside_out[1])


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
        maskset = self.csu.mskset
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


    def load_mask(self, mask_fn, stack=True):

        if stack != True:
            assert 0, 'Not yet implemented'

        def _multi(a, b):
            return a*b

        masks_file = [
            np.load(fn) 
            if fn.endswith('.npy') else hp.read_map(fn, dtype=np.bool)
                for fn in mask_fn]

        masks_file = functools.reduce(
            _multi,
            [a for a in masks_file])

        return masks_file


    def load_alms(self, component, id):
        if component == 'cmb':
            cmb_tlm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(id), hdu=1)
            cmb_elm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(id), hdu=2)
            cmb_blm = hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%int(id), hdu=3)
            
            return cmb_tlm, cmb_elm, cmb_blm
        else:
            print('unclear request for loading alms. Exiting..')
            sys.exit()


    def load_truthspectrum(self, abs_path=""):
        return pd.read_csv(
            abs_path+self.csu.powspec_truthfile,
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
    def load_beamf_old(self, freqcomb: List) -> Dict:
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


    @log_on_start(INFO, "Starting to load beamf functions from frequency channels {freqcomb}")
    @log_on_end(DEBUG, "Beamfunction(s) loaded successfully")
    def load_beamf(self, beamf_info, freqcomb: List) -> Dict:
        """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

        Args:
            cf (Dict): Configuration as coming from conf.json
            mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
            freqcomb (List): Frequency channels which are to be ignored

        Returns:
            np.array: Planck beamfunctions with dimension (Nspec, nfreq, nfreq, lmaxp1)
        """

        #TODO this belongs to fn_gen and should be passed to this function rather than being calculated here.
        # Then beamf_info can be removed from the function parameters
        def _get_beamffn():
            beamf = dict()
            if beamf_info['info'] == 'DX12':
                for freqc in freqcomb:
                    freqs = freqc.split('-')
                    if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                        beamf.update({
                            freqc: {
                                "HFI": fits.open(
                                    "{bf_path}{bf_filename}"
                                    .format(
                                        bf_path = beamf_info["HFI"]['ap'].replace("{split}", self.csu.freqdatsplit),
                                        bf_filename = beamf_info["HFI"]['filename']
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
                                        bf_path = beamf_info["HFI"]['ap'].replace("{split}", self.csu.freqdatsplit),
                                        bf_filename = beamf_info["HFI"]['filename']
                                            .replace("{freq1}", freqs[1])
                                            .replace("{freq2}", freqs[1])
                                ))
                            }
                        })
                        beamf[freqc].update({
                            "LFI": fits.open(
                                    "{bf_path}{bf_filename}"
                                    .format(
                                        bf_path = beamf_info["LFI"]['ap'].replace("{split}", self.csu.freqdatsplit),
                                        bf_filename = beamf_info["LFI"]['filename']
                                ))
                        })
                    elif int(freqs[0]) < 100 and int(freqs[1]) < 100:
                        beamf.update({
                            freqc: {
                                "LFI": fits.open(
                                    "{bf_path}{bf_filename}"
                                    .format(
                                        bf_path = beamf_info["LFI"]['ap'].replace("{split}", self.csu.freqdatsplit),
                                        bf_filename = beamf_info["LFI"]['filename']
                                ))
                            }})
            elif beamf_info['info'] == 'NPIPE':
                for freqc in freqcomb:
                    freqs = freqc.split('-')
                    beamf.update({
                        freqc: {
                            "HFI": fits.open(
                                "{bf_path}{bf_filename}"
                                .format(
                                    bf_path = beamf_info['ap'].replace("{split}", self.csu.freqdatsplit),
                                    bf_filename = beamf_info['filename']
                                        .replace("{freq1}", freqs[0])
                                        .replace("{freq2}", freqs[1])
                                ))
                            }
                        })

            return beamf

        beamf = _get_beamffn()

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
                    if beamf_info['info'] == 'DX12':
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
                    elif beamf_info['info'] == 'NPIPE':
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
        print('io.py: Data saved to {}'.format(path_name))


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
