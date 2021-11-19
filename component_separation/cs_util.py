
"""cs_util.py:
Configuration handler and Filenamegenerators.
Loads either of the config_XXX.py files and imports its data/beam/maskclasses, depending on the configuration
Filename class works depending on the configuration.
"""

import functools
import logging
import logging.handlers
import os
import os.path as path
import platform
import warnings

import numpy as np
from scipy import interpolate

import component_separation.cachechecker as cc

#TODO class Config could be further improved as follows:
# 1. collect all attributes which are supposed to be included as filenameattribute (in this way, its very transparent for future change)
# 2. test which of the attributes match the current request, remove if needed
# 3. loop over remaining attributes, link with '/'

# currently, 1-3 is done implicit in the following lines, but in a non-scalable way, and not transparent for my liking
class Config:
    def __init__(self, experiment='Planck', verbose = False, **kwargs):
        assert experiment in ['Planck', 'Pico']

        self.experiment = experiment
        if experiment == 'Planck':
            from component_separation.config_planck import (Params, 
            Frequency, Spectrum)
            from component_separation.config_planck import Cutoff as Pcut
        elif experiment == 'Pico':
            from component_separation.config_pico import (Params,
            Frequency, Spectrum)
            from component_separation.config_pico import Cutoff as Pcut

        self.cutoff_freq = Pcut().get_cutoff(Params.__dict__['cutoff'], Params.__dict__['lmax']+1)

        uname = platform.uname()
        if uname.node == "DESKTOP-KMIGUPV":
            self.mch = "XPS"
        else:
            self.mch = "NERSC"

        self.CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
        "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
        self.CB_color_cycle_lighter = ["#68ACCE", "#AC4657", "#ADAC57", "#005713", "#130268", "#8A2479", 
        "#248A79", "#797913", "#680235", "#460000", "#4679AC", "#686868"]

        """
        Overwrite all params which are directly passed to __init__
        """
        self.__dict__.update(Params.__dict__)
        self.__dict__.update(kwargs)

        self.bins = getattr(Smica_bins, self.binname)

        if experiment == 'Planck':
            if self.mskset == "smica":
                from component_separation.config_planck import BeamfDX12
                self.get_beamf = BeamfDX12.get_beamf
            elif self.mskset == "lens":
                from component_separation.config_planck import BeamfNPIPE
                self.get_beamf = BeamfNPIPE.get_beamf
        elif experiment == 'Pico':
            from component_separation.config_pico import Beamfd90sim
            self.get_beamf = Beamfd90sim.get_beamf

        if verbose == True:
            print(40*"$")
            print("Run with the following settings:")
            print(self.__dict__)
            print(40*"$")


class Filename_gen:
    """Generator to consistenly create filenames,
        1. from configuration file
        2. upon runtime
    One may want to differentiate between,
        simulation, realdata (NPIPE / DX12)
        powerspec, map, alms,
        powspectype (pseudo, chonetal)
        ...
    Create filename hierarchically,
        1. choose useful hierarical directory structure
        2. for each config file setting, generate unique level0 name
        3. for all files generated upon runtime, generate unique level1 name
    """

    def __init__(self, csu_loc, experiment_loc=None, dir_structure=None, fn_structure=None, simid=None):
        """To add a new attribute,
            1. add it to dir/fn_structure
            2a. add an instance test (here in the __init__)
            2b. or add a runtime test (perhaps in get_spectrum())
        """
        self.csu_loc = csu_loc
        experiment = csu_loc.experiment if experiment_loc is None else experiment_loc
        self.simid = simid

        if experiment == 'Planck':
            from component_separation.config_planck import Asserter as ass
            self.ass = ass
        elif experiment == 'Pico':
            from component_separation.config_pico import Asserter as ass
            self.ass = ass

        if csu_loc.mch == 'NERSC':
            if experiment == 'Planck':
                if csu_loc.freqdset == "NPIPE":
                    from component_separation.config_planck import NPIPE as dset_fn
                    self.dset_fn = dset_fn
                elif csu_loc.freqdset == "DX12":
                    from component_separation.config_planck import DX12 as dset_fn
                    self.dset_fn = dset_fn
                else:
                    assert 0, "to be implemented: {}".format(csu_loc.freqdset)
            elif experiment == 'Pico':
                if csu_loc.freqdset == "d90sim":
                    from component_separation.config_pico import d90sim as dset_fn
                    self.dset_fn = dset_fn
                else:
                    assert 0, "to be implemented: {}".format(csu_loc.freqdset)

        if experiment == 'Planck':
            if csu_loc.freqdset.startswith("NPIPE"):
                from component_separation.config_planck import BeamfNPIPE as beamf
                self.beamf = beamf
            elif csu_loc.freqdset.startswith("DX12"):
                from component_separation.config_planck import BeamfDX12 as beamf
                self.beamf = beamf
            else:
                assert 0, "beamf to be implemented: {}".format(csu_loc.freqdset)
        elif experiment == 'Pico':
            if csu_loc.freqdset.startswith("d90"):
                from component_separation.config_pico import Beamfd90sim as beamf
                self.beamf = beamf
            else:
                assert 0, "beamf to be implemented: {}".format(csu_loc.freqdset)

        if csu_loc.mskset == "smica":
            from component_separation.config_planck import Smica_Mask as mask
            self.mask = mask
        elif csu_loc.mskset == "lens":
            from component_separation.config_planck import Lens_Mask as mask
            self.mask = mask


    def get_misc(self, misc_type, simid=None, prefix=None):
        assert misc_type in self.ass.misc_type
        assert simid in self.ass.simid or simid is None, simid

        simid = self.simid if simid is None else simid

        dir_misc_loc = self._get_miscdir(simid)
        if prefix is None:
            pass
        else:
            prefix = dir_misc_loc
        filename_loc = self._get_miscfn(misc_type, simid, prefix=prefix)

        return path.join(dir_misc_loc, filename_loc)


    def get_spectrum(self, info_component, info_combination="non-separated", simid=-1, prefix=None):
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid


        if info_component == "S" and simid is -1:
            """This is a special case, as it needs to a fake signal datafile. Usually this is available for simulations, but not for real data.
            However, Smica needs a fake signal. The proper treatment may be found in Filname_gen_SMICA
            """
            assert 0, "Please use Filname_gen_SMICA for this"

        mapspecmiscdir = self._get_mapspecmiscdir(info_component, simid)
        if prefix is None:
            pass
        else:
            prefix = mapspecmiscdir
        filename_loc = self._get_specfn(info_component, info_combination, simid, prefix=prefix)

        return path.join(mapspecmiscdir, filename_loc)
    

    def get_d(self, freq, info_component, simid=None):
        assert freq in self.ass.FREQ, freq
        assert info_component in self.ass.info_component, info_component
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        if info_component == self.ass.info_component[0]:  # Noise
            dir_plamap_loc = self.dset_fn._get_noisedir(self.csu_loc.freqdatsplit, freq)
            if dir_plamap_loc == "INTERNAL":
                dir_plamap_loc = self._get_mapdir(info_component)
            fn_loc = self.dset_fn._get_noisefn(freq, simid)
        if info_component == self.ass.info_component[3]:  # data
            dir_plamap_loc = self.dset_fn._get_ddir(str(simid), self.csu_loc.freqdatsplit)
            fn_loc = self.dset_fn._get_dfn(self.csu_loc.freqdatsplit, freq, simid)
                    
        return path.join(dir_plamap_loc, fn_loc)
        

    def get_map(self, info_component, info_combination, simid=None, prefix=None):
        assert 0, 'To be implemented'
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        dir_map_loc = self._get_dir(info_component, simid)
        if prefix is None:
            pass
        else:
            prefix = dir_map_loc
        filename_loc = self._get_mapfn(info_component, info_combination, simid, prefix=prefix)

        return path.join(dir_map_loc, filename_loc)


    def get_mask(self, TorP, apodized):

        return [path.join(self.mask.get_dir(), fn) 
            for fn in self.mask.get_fn(TorP, apodized)]


    def get_beamfinfo(self, ):

        return self.beamf.get_beaminfo


    def _get_miscdir(self, simid=None):
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(simid)))
            cc.iff_make_dir(ap)

        return ap


    def _get_mapdir(self, info_component, simid=None):
        assert info_component in self.ass.info_component
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(simid)))
            cc.iff_make_dir(ap)

        ap = path.join(ap, info_component)
        cc.iff_make_dir(ap)

        ap_map = path.join(ap, 'map')
        cc.iff_make_dir(ap_map)

        return ap_map


    def _get_mapspecmiscdir(self, info_component, simid=None):
        assert info_component in self.ass.info_component
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(simid)))
            cc.iff_make_dir(ap)

        return ap


    def _get_specfn(self, info_component, info_combination, simid=None, prefix=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        if prefix is None:
            retval = 'Cl{}'.format(info_component)
        else:
            retval = prefix.replace('/', '_')

        retval = '_'.join([retval, info_combination])
        retval = '_'.join([retval, str(self.csu_loc.nside_out[1])])
        retval = '_'.join([retval, str(self.csu_loc.lmax)])

        if self.csu_loc.spectrum_type == 'JC':
            retval='_'.join([retval, str(self.csu_loc.lmax_mask)])
            retval='_'.join([retval, "JC"])
        elif self.csu_loc.spectrum_type == "pseudo":
            retval='_'.join([retval, "pseudo"])

        if self.csu_loc.simdata and simid is not None:
            retval = '_'.join([retval, str(simid)])

        return '.'.join([retval, "npy"])


    def _get_mapfn(self, info_component, info_combination, simid=None, prefix=None):
        assert 0, 'To be implemented'
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert info_combination in ["combined", "perfreq"]
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        if prefix is None:
            retval = 'Map'
        else:
            retval = prefix.replace('/', '_')

        retval = '_'.join([retval, info_combination])

        retval = '_'.join([retval, str(self.csu_loc.nside_out[1])])

        if self.csu_loc.spectrum_type == 'JC':
            retval='_'.join([retval, str(self.csu_loc.lmax_mask)])
            retval='_'.join([retval, "JC"])
        elif self.csu_loc.spectrum_type == "pseudo":
            retval='_'.join([retval, "pseudo"])

        if self.csu.simdata and simid is not None:
            retval = '_'.join([retval, str(simid)])

        return '.'.join([retval, "npy"])


    def _get_dfn(self, FREQ, simid=None):

        simid = self.simid if simid is None else simid
        freqdset = self.csu_loc.freqdset
        nside_desc = self.csu_loc.nside_desc_map
        mch = self.csu_loc.mch

        abs_path = self.csu_loc.cf[mch][freqdset]['ap']
        freq_filename = self.csu_loc.cf[mch][freqdset]['filename']

        return '{abs_path}{freq_filename}'\
            .format(
                abs_path = abs_path\
                    .replace("{simid}", simid)\
                    .replace("{split}", self.csu_loc.freqdatsplit),
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Frequencyclass.LFI.value if int(FREQ)<100 else Frequencyclass.HFI.value)
                    .replace("{nside}", str(nside_desc[0]) if int(FREQ)<100 else str(nside_desc[1]))
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                    .replace("{even/half1}", "evenring" if int(FREQ)>=100 else "ringhalf-1")
                    .replace("{odd/half2}", "oddring" if int(FREQ)>=100 else "ringhalf-2")
                    .replace("{simid}", simid)\
                    .replace("{split}", self.csu_loc.freqdatsplit)
            )


    def _get_miscfn(self, misc_type, simid, prefix):
        assert misc_type in self.ass.misc_type
        assert simid in self.ass.simid or simid is None

        if prefix is None:
            retval = "{}".format(misc_type)
        else:
            retval = prefix.replace('/', '_')


        retval = '_'.join([retval, self.csu_loc.spectrum_type])

        if simid != -1:
            retval = '_'.join([retval, str(simid).zfill(4)])

        return '.'.join([retval, "npy"])


class Filename_gen_SMICA:
    """Same as Filename_gen, but for SMICA runs
    """

    def __init__(self, csu_loc, dir_structure=None, fn_structure=None, simid=None):
        """To add a new attribute,
            1. add it to dir/fn_structure
            2a. add an instance test (here in the __init__)
            2b. or add a runtime test (perhaps in get_spectrum())
        """
        #TODO implement dir_structure and fn_structure

        self.csu_loc = csu_loc
        if csu_loc.experiment == 'Planck':
            from component_separation.config_planck import Asserter_smica as ass
        else:
             from component_separation.config_pico import Asserter_smica as ass
        self.ass = ass
        self.simid = simid

        if csu_loc.mch == 'NERSC':
            if csu_loc.experiment == 'Planck':
                if csu_loc.freqdset == "NPIPE":
                    from component_separation.config_planck import NPIPE as dset_fn
                    self.dset_fn = dset_fn
                elif csu_loc.freqdset == "DX12":
                    from component_separation.config_planck import DX12 as dset_fn
                    self.dset_fn = dset_fn
                else:
                    assert 0, "to be implemented: {}".format(csu_loc.freqdset)
            elif csu_loc.experiment == 'Pico':
                if csu_loc.freqdset == "d90sim":
                    from component_separation.config_pico import d90sim as dset_fn
                    self.dset_fn = dset_fn
                else:
                    assert 0, "to be implemented: {}".format(csu_loc.freqdset)
        
        dir_structure = "{dataset}/{mask}/{simXX}/{smicasepCOMP}/{spec}"
        fn_structure = "Cl_{binname}_{info_component}_{info_comb}_{lmax}_{lmax_mask}_{spectrum_type}_{nside}_{simXX}"


    def get_spectrum(self, info_component, info_combination, simid=None, prefix=None):
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination, info_combination
        assert simid in self.ass.simid or simid is None
        assert not(info_combination == "separated" and info_component == "T"), "{} and {} is not a valid request".format(info_component, info_combination)

        simid = self.simid if simid is None else simid

        if info_component == "S" and simid is -1:
            "Special case, as smica needs signal estimator"
            return self.dset_fn._get_signalest()

        dir_spec_loc = self._get_dir(info_component, info_combination, simid)
        if prefix is None:
            pass
        else:
            prefix = dir_spec_loc
        filename_loc = self._get_specfn(info_component, info_combination, simid, prefix=prefix)

        return path.join(dir_spec_loc, filename_loc)


    def get_map(self, info_component, info_combination, simid=None, prefix=None):
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination, info_combination
        assert simid in self.ass.simid or simid is None
        assert not(info_combination == "separated" and info_component == "T"), "{} and {} is not a valid request".format(info_component, info_combination)

        simid = self.simid if simid is None else simid


        dir_spec_loc = self._get_dir(info_component, info_combination, simid)
        if prefix is None:
            pass
        else:
            prefix = dir_spec_loc
        filename_loc = self._get_mapfn(info_component, info_combination, simid, prefix=prefix)

        return path.join(dir_spec_loc, filename_loc)


    def get_misc(self, desc, simid=None, prefix=None):
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        dir_misc_loc = self._get_miscdir(simid)
        if prefix is None:
            pass
        else:
            prefix = dir_misc_loc
        filename_loc = self._get_miscfn(desc, simid, prefix=prefix)

        return path.join(dir_misc_loc, filename_loc)


    def _get_miscdir(self, simid=None):
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(simid)))
            cc.iff_make_dir(ap)

        return ap


    def _get_dir(self, info_component, info_combination, simid=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(simid)))
            cc.iff_make_dir(ap)

        if info_combination == 'separated' or info_combination == 'combined':
            ap_spec = path.join(ap, 'smicasep')
            cc.iff_make_dir(ap_spec)

            return ap

        return ap


    def _get_specfn(self, info_component, info_combination, simid=None, prefix=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        if prefix is None:
            retval = 'Cl{}'.format(info_component)
        else:
            retval = prefix.replace('/', '_')

        retval = '_'.join([retval, info_combination])
        retval = '_'.join([retval, self.csu_loc.binname])
        retval = '_'.join([retval, str(self.csu_loc.nside_out[1])])
        retval = '_'.join([retval, str(self.csu_loc.lmax)])

        if self.csu_loc.spectrum_type == 'JC':
            retval='_'.join([retval, str(self.csu_loc.lmax_mask)])
            retval='_'.join([retval, "JC"])
        elif self.csu_loc.spectrum_type == "pseudo":
            retval='_'.join([retval, "pseudo"])

        if self.csu_loc.simdata and simid is not None:
            retval = '_'.join([retval, str(simid)])

        return '.'.join([retval, "npy"])


    def _get_mapfn(self, info_component, info_combination, simid=None, prefix=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert simid in self.ass.simid or simid is None

        simid = self.simid if simid is None else simid

        if prefix is None:
            retval = 'Map{}'.format(info_component)
        else:
            retval = prefix.replace('/', '_')

        retval = '_'.join([retval, info_combination])
        retval = '_'.join([retval, self.csu_loc.binname])
        retval = '_'.join([retval, str(self.csu_loc.nside_out[1])]) 
        retval = '_'.join([retval, str(self.csu_loc.lmax)])

        if self.csu_loc.spectrum_type == 'JC':
            retval='_'.join([retval, str(self.csu_loc.lmax_mask)])
            retval='_'.join([retval, "JC"])
        elif self.csu_loc.spectrum_type == "pseudo":
            retval='_'.join([retval, "pseudo"])

        if self.csu_loc.simdata and simid is not None:
            retval = '_'.join([retval, str(simid)])

        return '.'.join([retval, "npy"])


    def _get_signalestimator():
        #TODO do it properly
        return None


    def _get_miscfn(self, misc_type, simid, prefix):
        assert misc_type in self.ass.misc_type
        assert simid in self.ass.simid or simid is None

        if prefix is None:
            retval = "smica_{}".format(misc_type)
        else:
            retval = prefix.replace('/', '_')

        retval = '_'.join([retval, self.csu_loc.binname])
        retval = '_'.join([retval, self.csu_loc.spectrum_type])

        return '.'.join([retval, "npy"])


class Smica_bins:
    SMICA_lowell_bins = np.array([
    [5.00e+00, 9.00e+00], [1.00e+01, 1.90e+01],
    [2.00e+01, 2.90e+01], [3.00e+01, 3.90e+01],
    [4.00e+01, 5.90e+01], [6.00e+01, 7.90e+01],
    [8.00e+01, 9.90e+01], [1.00e+02, 1.19e+02],
    [1.20e+02, 1.39e+02], [1.40e+02, 1.59e+02]], dtype=float)

    SMICA_highell_bins = np.array([
    [2.00e+00, 9.00e+00], [1.00e+01, 1.90e+01],
    [2.00e+01, 2.90e+01], [3.00e+01, 3.90e+01],
    [4.00e+01, 5.90e+01], [6.00e+01, 7.90e+01],
    [8.00e+01, 9.90e+01], [1.00e+02, 1.19e+02],
    [1.20e+02, 1.39e+02], [1.40e+02, 1.59e+02],
    [1.60e+02, 1.79e+02], [1.80e+02, 1.99e+02],
    [2.00e+02, 2.19e+02], [2.20e+02, 2.39e+02],
    [2.40e+02, 2.59e+02], [2.60e+02, 2.79e+02],
    [2.80e+02, 2.99e+02], [3.00e+02, 3.19e+02],
    [3.20e+02, 3.39e+02], [3.40e+02, 3.59e+02],
    [3.60e+02, 3.79e+02], [3.80e+02, 3.99e+02],
    [4.00e+02, 4.19e+02], [4.20e+02, 4.39e+02],
    [4.40e+02, 4.59e+02], [4.60e+02, 4.79e+02],
    [4.80e+02, 4.99e+02], [5.00e+02, 5.49e+02],
    [5.50e+02, 5.99e+02], [6.00e+02, 6.49e+02],
    [6.50e+02, 6.99e+02], [7.00e+02, 7.49e+02],
    [7.50e+02, 7.99e+02], [8.00e+02, 8.49e+02],
    [8.50e+02, 8.99e+02], [9.00e+02, 9.49e+02],
    [9.50e+02, 9.99e+02]], dtype=float)

    linear_equisized_bins_100 = np.array([
       [   0,   99], [ 100,  199],
       [ 200,  299], [ 300,  399],
       [ 400,  499], [ 500,  599],
       [ 600,  699], [ 700,  799],
       [ 800,  899], [ 900,  999],
       [1000, 1099], [1100, 1199],
       [1200, 1299], [1300, 1399],
       [1400, 1499], [1500, 1599],
       [1600, 1699], [1700, 1799],
       [1800, 1899], [1900, 1999],
       [2000, 2099], [2100, 2199],
       [2200, 2299], [2300, 2399],
       [2400, 2499], [2500, 2599],
       [2600, 2699], [2700, 2799],
       [2800, 2899]], dtype=float)


class Helperfunctions:
    llp1e12 = lambda x: x*(x+1)*1e12/(2*np.pi)


    @staticmethod
    def bin_it(data, bins=Smica_bins.SMICA_lowell_bins):

        ret = np.ones((*data.shape[:-1], len(bins)))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(bins.shape[0]):
                    ret[i,j,k] = np.nanmean(data[i,j,int(bins[k][0]):int(bins[k][1])+1])
        ret = np.nan_to_num(ret)

        return ret


    @staticmethod
    def bin_it1D(data, bins):

        ret = np.ones(len(bins))
        for k in range(bins.shape[0]):
            ret[k] = np.nanmean(np.nan_to_num(data[int(bins[k][0]):int(bins[k][1])]))

        return np.nan_to_num(ret)


    @staticmethod   
    def interp_smica_mv_weights(W_smica, W_mv, bins, lmaxp1):
        
        ## TODO T currently not supported
        ## add smoothing for weights at high ell (especially important when crossing, e.g. npipe data with dx12 derived weights)
        W_total = np.zeros(shape=(*W_mv.shape[:-1], lmaxp1))
        xnew = np.arange(0,bins[-1][1]+1,1)
        for it in range(W_total.shape[1]): #weights do not depend on freqfilter, but almE/B do
            W_Einterp = interpolate.interp1d(np.mean(bins, axis=1), W_smica[0,it,:], bounds_error = False, fill_value='extrapolate')
            W_total[1,it] = np.concatenate((W_Einterp(xnew),W_mv[1,it,xnew.shape[0]:]))
            W_Binterp = interpolate.interp1d(np.mean(bins, axis=1), W_smica[1,it,:], bounds_error = False, fill_value='extrapolate')
            W_total[2,it] = np.concatenate((W_Binterp(xnew),W_mv[2,it,xnew.shape[0]:]))

        return W_total


    @staticmethod
    def multi(a,b):

        return a*b
    
    
    @staticmethod
    def reorder_spectrum_dict(spectrum):

        spec_data = dict()
        for f in spectrum.keys():
            for s in spectrum[f].keys():
                if s in spec_data:
                    spec_data[s].update({
                        f: spectrum[f][s]})
                else:
                    spec_data.update({s:{}})
                    spec_data[s].update({
                        f: spectrum[f][s]
                    })

        return spec_data


    @staticmethod
    def std_dev_binned(d, lmax=3000, binwidth=200, log=True):
        
        #TODO check if function is incorrect, errorbars sometimes vary with binsize?
        if log == False:
            bins = np.linspace(0, lmax+1, binwidth)
        else:
            bins = np.logspace(np.log10(1), np.log10(lmax+1), binwidth)

        bl = bins[:-1]
        br = bins[1:]
        if type(d) == np.ndarray:
            val = np.nan_to_num(d)
        elif type(d) == np.ma.core.MaskedArray:
            val = np.nan_to_num(d)
        else:
            val = np.nan_to_num(d.to_numpy())
        n, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins)
        sy, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val)
        sy2, _ = np.histogram(
            np.linspace(0,lmax,lmax),
            bins=bins,
            weights=val * val)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)

        return mean, std, _


    @staticmethod
    def negvalinterp(data, bins):
        """
        The following is a 'negative-value' to nearest neighbor interpolation,
        but it breaks the B-fit pipeline for SMICA.
        """
        ret = np.ones(len(bins))
        for i in range(ret.shape[0]):
            for j in range(ret.shape[1]):
                fill_left=False
                for k in range(ret.shape[2]):
                    if ret[i,j,k]<0:
                        if k==0:
                            fill_left = True
                        elif k>0:
                            ret[i,j,k] = ret[i,j,k-1]
                    if ret[i,j,k]>0 and fill_left==True:
                        fill_left = False
                        ret[i,j,:k] = [ret[i,j,k] for _ in range(k)]


    @staticmethod
    def deprecated(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Call to deprecated function {}.".format(func.__name__),
                        category=DeprecationWarning,
                        stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return wrapped
