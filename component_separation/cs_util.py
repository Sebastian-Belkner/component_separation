
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
from component_separation.config_planck import (Params, Planckf, Planckr,
                                                Plancks)

class Config(Params):
    def __init__(self, **kwargs):
        """
        Load Configuartion file and store as parameters, but do not expose config file.
        """
        # TODO configuration could depend on what is currently done
            # 1. map generation
            # 2. powerspectrum generation
            # 3. component separation

        uname = platform.uname()
        if uname.node == "DESKTOP-KMIGUPV":
            self.mch = "XPS"
        else:
            self.mch = "NERSC"

        self.bins = getattr(Smica_bins, self.binname)

        self.CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
        "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
        self.CB_color_cycle_lighter = ["#68ACCE", "#AC4657", "#ADAC57", "#005713", "#130268", "#8A2479", 
        "#248A79", "#797913", "#680235", "#460000", "#4679AC", "#686868"]


        """
        Overwrite all params which are directly passed to __init__
        """
        self.__dict__.update(Params.__dict__)
        self.__dict__.update(kwargs)

        print(40*"$")
        print("Run with the following settings:")
        print(self.__dict__)
        print(40*"$")


#TODO the algorithm could be further improved as follows:
# 1. collect all attributes which are supposed to be included as filenameattribute (in this way, its very transparent for future change)
# 2. test which of the attributes match the current request, remove if needed
# 3. loop over remaining attributes, link with '/'

# currently, 1-3 is done implicit in the following lines, but in a non-scalable way, and not transparent for my liking
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

    The goal is to,
     1. set up directories
     2. generate hierarch. filename
     2. pass info to cachechecker
    """

    def __init__(self, csu_loc, dir_structure=None, fn_structure=None, sim_id=None):
        """To add a new attribute,
            1. add it to dir/fn_structure
            2a. add an instance test (here in the __init__)
            2b. or add a runtime test (perhaps in get_spectrum())
        """
        #TODO implement dir_structure and fn_structure. Or don't

        self.csu_loc = csu_loc
        from component_separation.cachechecker import Asserter as ass
        self.ass = ass

        if csu_loc.mch == 'NERSC':
            if csu_loc.freqdset == "NPIPE":
                from component_separation.config_planck import NPIPE as dset_fn
                self.dset_fn = dset_fn
            elif csu_loc.freqdset == "DX12":
                from component_separation.config_planck import DX12 as dset_fn
                self.dset_fn = dset_fn
            else:
                assert 0, "to be implemented: {}".format(csu_loc.freqdset)
        
        if csu_loc.mskset == "smica":
            from component_separation.config_planck import Smica_Mask as mask
            self.mask = mask
        elif csu_loc.mskset == "lens":
            from component_separation.config_planck import Lens_Mask as mask
            self.mask = mask

        if csu_loc.freqdset.startswith("NPIPE"):
            from component_separation.config_planck import BeamfNPIPE as beamf
            self.beamf = beamf
        if csu_loc.freqdset.startswith("DX12"):
            from component_separation.config_planck import BeamfDX12 as beamf
            self.beamf = beamf

        # dir_structure = "{dataset}/{mask}/{simXX}/{spec}"
        # fn_structure = "Cl_{info_component}_{info_comb}_{lmax}_{lmax_mask}_{spectrum_type}_{nside}_{simXX}"
        #test if attributes should be in for this instance
        # if self.csu_loc['freqdset'] in ['DX12', 'NPIPE']:
        #     dir_structure.replace("{dataset}", self.csu_loc['freqdset'])

        # if self.csu_loc['mskset'] in ['smica', 'lens']:
        #     dir_structure.replace("{mask}", self.csu_loc['mskset'])
        
        # if self.csu_loc['simdata'] == True:
        #     assert type(sim_id), int
        #     dir_structure.replace("{simXX}", "sim{}".format(sim_id))
        # else:
        #     dir_structure.replace("_{simXX}", "")

        # there might be attributes left which can only be decided upon runtime
        # self.dir_name = dir_structure


    def get_misc(self, misc_type, sim_id=None, prefix=None):
        assert misc_type in self.ass.misc_type
        assert type(sim_id) == int or sim_id is None

        dir_misc_loc = self._get_miscdir(sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_misc_loc
        filename_loc = self._get_miscfn(misc_type, sim_id, prefix=prefix)

        return path.join(dir_misc_loc, filename_loc)


    def get_spectrum(self, info_component, info_combination="non-separated", sim_id=None, prefix=None):
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination
        assert type(sim_id) == int or sim_id is None


        if info_component == "S" and sim_id is None:
            """This is a special case, as it needs to a fake signal datafile. Usually this is available for simulations, but not for real data.
            However, Smica needs a fake signal. The proper treatment may be found in Filname_gen_SMICA
            """
            assert 0, "Please use Filname_gen_SMICA for this"


        mapspecmiscdir = self._get_mapspecmiscdir(info_component, sim_id)
        if prefix is None:
            pass
        else:
            prefix = mapspecmiscdir
        filename_loc = self._get_specfn(info_component, info_combination, sim_id, prefix=prefix)

        return path.join(mapspecmiscdir, filename_loc)
    

    def get_pla(self, freq, info_component, sim_id=None):
        assert freq in self.ass.PLANCKMAPFREQS, freq
        assert info_component in self.ass.info_component, info_component
        assert type(sim_id) == int or sim_id is None

        if info_component == self.ass.info_component[0]:  # Noise
            dir_plamap_loc = self.dset_fn._get_noisedir()
            if dir_plamap_loc == "INTERNAL":
                dir_plamap_loc = self._get_mapdir(info_component)
            fn_loc = self.dset_fn._get_noisefn()
        if info_component == self.ass.info_component[3]:  # PLA data
            dir_plamap_loc = self.dset_fn._get_pladir()
            fn_loc = self.dset_fn._get_plafn()
        dir_plamap_loc = dir_plamap_loc\
            .replace("{sim_id}", str(sim_id))\
            .replace("{split}", self.csu_loc.freqdatsplit)

        fn_loc = fn_loc\
            .replace("{sim_id}", str(sim_id))\
            .replace("{split}", self.csu_loc.freqdatsplit)\
            .replace("{freq}", freq)\
            .replace("{LorH}", Planckr.LFI.value if int(freq)<100 else Planckr.HFI.value)\
            .replace("{nside}", str(self.csu_loc.nside_desc_map[0]) if int(freq)<100 else str(self.csu_loc.nside_desc_map[1]))\
            .replace("{00/1}", "00" if int(freq)<100 else "01")
                    
        return path.join(dir_plamap_loc, fn_loc)
        

    def get_map(self, info_component, info_combination, sim_id=None, prefix=None):
        assert 0, 'To be implemented'
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination
        assert type(sim_id) == int or sim_id is None

        dir_map_loc = self._get_dir(info_component, sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_map_loc
        filename_loc = self._get_mapfn(info_component, info_combination, sim_id, prefix=prefix)

        return path.join(dir_map_loc, filename_loc)


    def get_mask(self, TorP):

        return [path.join(self.mask.get_dir(), fn) 
            for fn in self.mask.get_fn(TorP)]


    def get_beamf(self):
        # TODO this needs to access proper dir + fn, as with masks and data. It works, but it's ugly
        return self.beamf.beamf


    def _get_miscdir(self, sim_id=None):
        assert type(sim_id) == int or sim_id is None

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            cc.iff_make_dir(ap)

        return ap


    def _get_mapdir(self, info_component, sim_id=None):
        """dir structure:
            compsep/
                --syn/
                    sim_id/dataset/mask/
                        --map/
                        --spec/
                --dataset/mask/
                        --map/
                            dset_msk_map__ + filename
                        --spec/
        """
        assert info_component in self.ass.info_component
        assert type(sim_id) == int or sim_id is None

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            cc.iff_make_dir(ap)

        ap = path.join(ap, info_component)
        cc.iff_make_dir(ap)

        ap_map = path.join(ap, 'map')
        cc.iff_make_dir(ap_map)

        return ap_map


    def _get_mapspecmiscdir(self, info_component, sim_id=None):
        assert info_component in self.ass.info_component
        assert type(sim_id) == int or sim_id is None

        """dir structure:
            compsep/
                --syn/
                    sim_id/dataset/mask/
                        --map/
                        --spec/
                --dataset/mask/
                        --map/
                            dset_msk_map__ + filename
                        --spec/
        """

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            cc.iff_make_dir(ap)

        return ap


    def _get_specfn(self, info_component, info_combination, sim_id=None, prefix=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert type(sim_id) == int or sim_id is None

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

        if self.csu_loc.simdata and sim_id is not None:
            retval = '_'.join([retval, str(sim_id)])

        return '.'.join([retval, "npy"])


    def _get_mapfn(self, info_component, info_combination, sim_id=None, prefix=None):
        assert 0, 'To be implemented'
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert info_combination in ["combined", "perfreq"]
        assert type(sim_id) == int or sim_id is None

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

        if self.csu.simdata and sim_id is not None:
            retval = '_'.join([retval, str(sim_id)])

        return '.'.join([retval, "npy"])


    def _get_plafn(self, FREQ, sim_id=None):
        freqdset = self.csu_loc.freqdset
        nside_desc = self.csu_loc.nside_desc_map
        mch = self.csu_loc.mch

        abs_path = self.csu_loc.cf[mch][freqdset]['ap']
        freq_filename = self.csu_loc.cf[mch][freqdset]['filename']

        return '{abs_path}{freq_filename}'\
            .format(
                abs_path = abs_path\
                    .replace("{sim_id}", sim_id)\
                    .replace("{split}", self.csu_loc.freqdatsplit),
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside_desc[0]) if int(FREQ)<100 else str(nside_desc[1]))
                    .replace("{00/1}", "00" if int(FREQ)<100 else "01")
                    .replace("{even/half1}", "evenring" if int(FREQ)>=100 else "ringhalf-1")
                    .replace("{odd/half2}", "oddring" if int(FREQ)>=100 else "ringhalf-2")
                    .replace("{sim_id}", sim_id)\
                    .replace("{split}", self.csu_loc.freqdatsplit)
            )


    def _get_miscfn(self, misc_type, sim_id, prefix):
        assert misc_type in self.ass.misc_type
        assert type(sim_id) == int or sim_id is None

        if prefix is None:
            retval = "{}".format(misc_type)
        else:
            retval = prefix.replace('/', '_')


        retval = '_'.join([retval, self.csu_loc.spectrum_type])

        if sim_id is not None:
            retval = '_'.join([retval, sim_id])

        return '.'.join([retval, "npy"])


class Filename_gen_SMICA:
    """Same as Filename_gen, but for SMICA runs
    """

    def __init__(self, csu_loc, dir_structure=None, fn_structure=None, sim_id=None):
        """To add a new attribute,
            1. add it to dir/fn_structure
            2a. add an instance test (here in the __init__)
            2b. or add a runtime test (perhaps in get_spectrum())
        """
        #TODO implement dir_structure and fn_structure

        self.csu_loc = csu_loc
        from component_separation.cachechecker import Asserter_smica as ass
        self.ass = ass

        if csu_loc.mch == 'NERSC':
            if csu_loc.freqdset == "NPIPE":
                from component_separation.config_planck import NPIPE as dset_fn
                self.dset_fn = dset_fn
            elif csu_loc.freqdset == "DX12":
                from component_separation.config_planck import DX12 as dset_fn
                self.dset_fn = dset_fn
            else:
                assert 0, "to be implemented: {}".format(csu_loc.freqdset)
        
        dir_structure = "{dataset}/{mask}/{simXX}/{smicasepCOMP}/{spec}"
        fn_structure = "Cl_{binname}_{info_component}_{info_comb}_{lmax}_{lmax_mask}_{spectrum_type}_{nside}_{simXX}"


    def get_spectrum(self, info_component, info_combination, sim_id=None, prefix=None):
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination, info_combination
        assert type(sim_id) == int or sim_id is None
        assert not(info_combination == "separated" and info_component == "T"), "{} and {} is not a valid request".format(info_component, info_combination)

        if info_component == "S" and sim_id is None:
            "Special case, as smica needs signal estimator"
            return self.dset_fn._get_signalest()

        dir_spec_loc = self._get_dir(info_component, info_combination, sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_spec_loc
        filename_loc = self._get_specfn(info_component, info_combination, sim_id, prefix=prefix)

        return path.join(dir_spec_loc, filename_loc)


    def get_map(self, info_component, info_combination, sim_id=None, prefix=None):
        assert info_component in self.ass.info_component, info_component
        assert info_combination in self.ass.info_combination, info_combination
        assert type(sim_id) == int or sim_id is None
        assert not(info_combination == "separated" and info_component == "T"), "{} and {} is not a valid request".format(info_component, info_combination)


        dir_spec_loc = self._get_dir(info_component, info_combination, sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_spec_loc
        filename_loc = self._get_mapfn(info_component, info_combination, sim_id, prefix=prefix)

        return path.join(dir_spec_loc, filename_loc)


    def get_misc(self, desc, sim_id=None, prefix=None):
        assert type(sim_id) == int or sim_id is None

        dir_misc_loc = self._get_miscdir(sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_misc_loc
        filename_loc = self._get_miscfn(desc, sim_id, prefix=prefix)

        return path.join(dir_misc_loc, filename_loc)


    def _get_miscdir(self, sim_id=None):
        assert type(sim_id) == int or sim_id is None

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            cc.iff_make_dir(ap)

        return ap


    def _get_dir(self, info_component, info_combination, sim_id=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert type(sim_id) == int or sim_id is None

        ap = path.join(self.csu_loc.outdir_ap, self.csu_loc.freqdset)  
        cc.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        cc.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            cc.iff_make_dir(ap)

        if info_combination == 'separated' or info_combination == 'combined':
            ap_spec = path.join(ap, 'smicasep')
            cc.iff_make_dir(ap_spec)

            return ap

        return ap


    def _get_specfn(self, info_component, info_combination, sim_id=None, prefix=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert type(sim_id) == int or sim_id is None

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

        if self.csu_loc.simdata and sim_id is not None:
            retval = '_'.join([retval, str(sim_id)])

        return '.'.join([retval, "npy"])


    def _get_mapfn(self, info_component, info_combination, sim_id=None, prefix=None):
        assert info_component in self.ass.info_component
        assert info_combination in self.ass.info_combination
        assert type(sim_id) == int or sim_id is None

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

        if self.csu_loc.simdata and sim_id is not None:
            retval = '_'.join([retval, str(sim_id)])

        return '.'.join([retval, "npy"])


    def _get_signalestimator():
        #TODO do it properly
        return None


    def _get_miscfn(self, misc_type, sim_id, prefix):
        assert misc_type in self.ass.misc_type
        assert type(sim_id) == int or sim_id is None

        if prefix is None:
            retval = "smica_{}".format(misc_type)
        else:
            retval = prefix.replace('/', '_')

        retval = '_'.join([retval, self.csu_loc.binname])
        retval = '_'.join([retval, self.csu_loc.spectrum_type])

        return '.'.join([retval, "npy"])


class Smica_bins:

    SMICA_lowell_bins = np.array([
    [5.000000000000000000e+00, 9.000000000000000000e+00],
    [1.000000000000000000e+01, 1.900000000000000000e+01],
    [2.000000000000000000e+01, 2.900000000000000000e+01],
    [3.000000000000000000e+01, 3.900000000000000000e+01],
    [4.000000000000000000e+01, 5.900000000000000000e+01],
    [6.000000000000000000e+01, 7.900000000000000000e+01],
    [8.000000000000000000e+01, 9.900000000000000000e+01],
    [1.000000000000000000e+02, 1.190000000000000000e+02],
    [1.200000000000000000e+02, 1.390000000000000000e+02],
    [1.400000000000000000e+02, 1.590000000000000000e+02]])

    SMICA_highell_bins = np.array([
    [2.000000000000000000e+00, 9.000000000000000000e+00],
    [1.000000000000000000e+01, 1.900000000000000000e+01],
    [2.000000000000000000e+01, 2.900000000000000000e+01],
    [3.000000000000000000e+01, 3.900000000000000000e+01],
    [4.000000000000000000e+01, 5.900000000000000000e+01],
    [6.000000000000000000e+01, 7.900000000000000000e+01],
    [8.000000000000000000e+01, 9.900000000000000000e+01],
    [1.000000000000000000e+02, 1.190000000000000000e+02],
    [1.200000000000000000e+02, 1.390000000000000000e+02],
    [1.400000000000000000e+02, 1.590000000000000000e+02],
    [1.600000000000000000e+02, 1.790000000000000000e+02],
    [1.800000000000000000e+02, 1.990000000000000000e+02],
    [2.000000000000000000e+02, 2.190000000000000000e+02],
    [2.200000000000000000e+02, 2.390000000000000000e+02],
    [2.400000000000000000e+02, 2.590000000000000000e+02],
    [2.600000000000000000e+02, 2.790000000000000000e+02],
    [2.800000000000000000e+02, 2.990000000000000000e+02],
    [3.000000000000000000e+02, 3.190000000000000000e+02],
    [3.200000000000000000e+02, 3.390000000000000000e+02],
    [3.400000000000000000e+02, 3.590000000000000000e+02],
    [3.600000000000000000e+02, 3.790000000000000000e+02],
    [3.800000000000000000e+02, 3.990000000000000000e+02],
    [4.000000000000000000e+02, 4.190000000000000000e+02],
    [4.200000000000000000e+02, 4.390000000000000000e+02],
    [4.400000000000000000e+02, 4.590000000000000000e+02],
    [4.600000000000000000e+02, 4.790000000000000000e+02],
    [4.800000000000000000e+02, 4.990000000000000000e+02],
    [5.000000000000000000e+02, 5.490000000000000000e+02],
    [5.500000000000000000e+02, 5.990000000000000000e+02],
    [6.000000000000000000e+02, 6.490000000000000000e+02],
    [6.500000000000000000e+02, 6.990000000000000000e+02],
    [7.000000000000000000e+02, 7.490000000000000000e+02],
    [7.500000000000000000e+02, 7.990000000000000000e+02],
    [8.000000000000000000e+02, 8.490000000000000000e+02],
    [8.500000000000000000e+02, 8.990000000000000000e+02],
    [9.000000000000000000e+02, 9.490000000000000000e+02],
    [9.500000000000000000e+02, 9.990000000000000000e+02]
    ], dtype=int)

    linear_equisized_bins_100 = np.array([[   0,   99],
       [ 100,  199],
       [ 200,  299],
       [ 300,  399],
       [ 400,  499],
       [ 500,  599],
       [ 600,  699],
       [ 700,  799],
       [ 800,  899],
       [ 900,  999],
       [1000, 1099],
       [1100, 1199],
       [1200, 1299],
       [1300, 1399],
       [1400, 1499],
       [1500, 1599],
       [1600, 1699],
       [1700, 1799],
       [1800, 1899],
       [1900, 1999],
       [2000, 2099],
       [2100, 2199],
       [2200, 2299],
       [2300, 2399],
       [2400, 2499],
       [2500, 2599],
       [2600, 2699],
       [2700, 2799],
       [2800, 2899]])


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
        """
        The following is a 'negative-value' to nearest neighbor interpolation,
        but it breaks the B-fit pipeline for SMICA.
        """
        # for i in range(ret.shape[0]):
        #     for j in range(ret.shape[1]):
        #         fill_left=False
        #         for k in range(ret.shape[2]):
        #             if ret[i,j,k]<0:
        #                 if k==0:
        #                     fill_left = True
        #                 elif k>0:
        #                     ret[i,j,k] = ret[i,j,k-1]
        #             if ret[i,j,k]>0 and fill_left==True:
        #                 fill_left = False
        #                 ret[i,j,:k] = [ret[i,j,k] for _ in range(k)]   
        return ret


    @staticmethod
    def bin_it1D(data, bins):
        ret = np.ones(len(bins))
        for k in range(bins.shape[0]):
            ret[k] = np.nanmean(np.nan_to_num(data[int(bins[k][0]):int(bins[k][1])]))
        return np.nan_to_num(ret)


    ### TODO
    ## T currently not supported
    ## add smoothing for weights at high ell (especially important when crossing, e.g. npipe data with dx12 derived weights)
    @staticmethod   
    def interp_smica_mv_weights(W_smica, W_mv, bins, lmaxp1):
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


    #TODO this functions seems to be incorrect, errorbars vary with binsize..
    @staticmethod
    def std_dev_binned(d, lmax=3000, binwidth=200, log=True):
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
