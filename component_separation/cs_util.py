from enum import Enum
import json
import itertools
import numpy as np
import logging
import logging.handlers
import platform
import os.path as path
import os
import component_separation
import functools
import warnings
from scipy import interpolate
import component_separation.io as io

class Config():
    # LOGFILE = 'data/tmp/logging/messages.log'
    # logger = logging.getLogger("")
    # handler = logging.handlers.RotatingFileHandler(
    #         LOGFILE, maxBytes=(1048576*5), backupCount=0
    # )
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    def __init__(self, cf=None):
        if cf is None:
            with open(os.path.dirname(component_separation.__file__)+'/config_ps.json', "r") as f:
                self.cf = json.load(f)
        else:
            self.cf = cf
        self.PLANCKMAPFREQ = [p.value for p in list(Planckf)]
        self.PLANCKSPECTRUM = [p.value for p in list(Plancks)]
        self.freqcomb =  ["{}-{}".format(FREQ,FREQ2)
            for FREQ, FREQ2  in itertools.product(self.PLANCKMAPFREQ,self.PLANCKMAPFREQ)
                if FREQ not in self.cf['pa']["freqfilter"] and
                (FREQ2 not in self.cf['pa']["freqfilter"]) and (int(FREQ2)>=int(FREQ))]
        
        
        self.PLANCKMAPFREQ_f = [FREQ for FREQ in self.PLANCKMAPFREQ
            if FREQ not in self.cf['pa']["freqfilter"]]
        if 'specfilter' in self.cf['pa'].keys():
            self.PLANCKSPECTRUM_f = [SPEC for SPEC in self.PLANCKSPECTRUM
                if SPEC not in self.cf['pa']["specfilter"]]
            self.speccomb  = [spec for spec in self.PLANCKSPECTRUM if spec not in self.cf['pa']["specfilter"]]
            self.specfilter = self.cf['pa']["specfilter"]
            self.Tscale = self.cf['pa']["Tscale"]
            self.binname = self.cf['pa']['binname']
            self.bins = getattr(Constants, self.binname)

        self.CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                             "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
        self.CB_color_cycle_lighter = ["#68ACCE", "#AC4657", "#ADAC57", "#005713", "#130268", "#8A2479", 
                        "#248A79", "#797913", "#680235", "#460000", "#4679AC", "#686868"]
        self.freqdset = self.cf['pa']["freqdset"]
        self.freqfilter = self.cf['pa']["freqfilter"]
        self.spectrum_type = self.cf['pa']['spectrum_type']
        if 'lmax' in self.cf['pa'].keys():
            self.lmax = self.cf['pa']["lmax"]
        self.overwrite_cache = self.cf['pa']['overwrite_cache']
        self.mskset = self.cf['pa']['mskset']
        self.simdata = self.cf['pa']['simdata']      
        

        uname = platform.uname()
        if uname.node == "DESKTOP-KMIGUPV":
            self.mch = "XPS"
        else:
            self.mch = "NERSC"

        if "sim_id" in self.cf[self.mch][self.freqdset]:
            self.sim_id = self.cf[self.mch][self.freqdset]["sim_id"]
        else:
            self.sim_id = "0200"
        self.nside_out = self.cf['pa']['nside_out'] if self.cf['pa']['nside_out'] is not None else self.cf['pa']['nside_desc_map']


        print(40*"$")
        print("Run with the following settings:")
        print(self.cf['pa'])
        print(40*"$")

    def set_logger(loglevel=logging.INFO):
        logger.setLevel(loglevel)
        logging.StreamHandler(sys.stdout)


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
    def __init__(self, csu_loc):
        self.csu_loc = csu_loc


    def _get_miscdir(self, sim_id=None):
        assert type(sim_id) == int or sim_id is None

        ap = path.join(self.cf_loc[self.csu_loc.mch]['outdir_ap'], 'compsep')
        io.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.freqdset)  
        io.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        io.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            io.iff_make_dir(ap)

        return ap


    def _get_dir(self, info_component, sim_id=None):
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert type(sim_id) == int or sim_id is None

        """dir structure:
            compsep/
                --syn/
                    sim_id/dataset/mask/
                        --map/
                        --spec/
                        --smicaseparated/
                            --map/
                                syn_synid_dset_msk_smica_map__ + filename
                            --spec/
                                syn_synid_dset_msk_smica_spec__ + filename
                --dataset/mask/
                        --map/
                            dset_msk_map__ + filename
                        --spec/
                    --smicaseparated/
                        --map/
                        --spec/
        """
        ap = path.join(self.cf_loc[self.csu_loc.mch]['outdir_ap'], 'compsep')
        io.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.freqdset)  
        io.iff_make_dir(ap)

        ap = path.join(ap, self.csu_loc.mskset+'mask')  
        io.iff_make_dir(ap)

        if self.csu_loc.simdata:
            ap = path.join(ap, 'sim{}'.format(str(sim_id)))
            io.iff_make_dir(ap)

        if info_component != 'non-sep':
            ap = path.join(ap, 'smicasep-{}'.format(info_component))
            io.iff_make_dir(ap)

        ap_map = path.join(ap, 'map')
        io.iff_make_dir(ap_map)
        ap_spec = path.join(ap, 'spec')
        io.iff_make_dir(ap_spec)

        ap_misc = path.join(ap, 'misc')
        io.iff_make_dir(ap_misc)
        return ap_map, ap_spec, ap_misc


    def _get_specfilename(self, info_component, info_combination, represent, sim_id=None, prefix=None):
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert info_combination in ["combined", "perfreq"]
        assert represent in ["Cl", "Dl"]
        assert type(sim_id) == int or sim_id is None
        """
        Filename: 
            <lmax>
                <lmaxmask_chonetal>
                pseudo
            <smicacom>_<binname>_<lmax>
                    <lmaxmask_chonetal>
                    pseudo
        """
        if prefix is None:
            retval = ''
            retval = ''.join([retval, represent])
        else:
            retval = prefix.replace('/', '_')
            retval = '_'.join([retval, represent])
        
        if info_component != 'non-sep':
            retval = '_'.join([retval, self.csu_loc.binname])

        retval = '_'.join([retval, info_combination])

        retval = '_'.join([retval, str(self.csu_loc.nside_out[1])])

        retval = '_'.join([retval, str(self.csu_loc.lmax)])

        if self.csu_loc.spectrum_type == 'Chonetal':
            retval='_'.join([retval, str(self.csu_loc.lmax_mask)])
            retval='_'.join([retval, "Chonetal"])
        elif self.csu_loc.spectrum_type == "pseudo":
            retval='_'.join([retval, "pseudo"])

        if self.csu_loc.simdata and sim_id is not None:
            retval = '_'.join([retval, str(sim_id)])

        return '.'.join([retval, "npy"])


    def _get_mapfilename(self, info_component, info_combination, represent, sim_id=None, prefix=None):
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert info_combination in ["combined", "perfreq"]
        assert represent in ["Cl", "Dl"]
        assert type(sim_id) == int or sim_id is None
        """
        Filename: 
            <lmax>
            <smicacom>_<binname>_<lmax>
                    <lmaxmask_chonetal>
        """
        assert 0, 'To be implemented'
        if prefix is None:
            retval = ''
            retval = ''.join([retval, represent])
        else:
            retval = prefix.replace('/', '_')
            retval = '_'.join([retval, represent])
        
        if info_component != 'non-sep':
            retval = '_'.join([retval, self.csu_loc.binname])

        retval = '_'.join([retval, info_combination])

        retval = '_'.join([retval, str(self.csu_loc.nside_out[1])])

        if self.csu_loc.spectrum_type == 'Chonetal':
            retval='_'.join([retval, str(self.csu_loc.lmax_mask)])
            retval='_'.join([retval, "Chonetal"])
        elif self.csu_loc.spectrum_type == "pseudo":
            retval='_'.join([retval, "pseudo"])

        if self.csu.simdata and sim_id is not None:
            retval = '_'.join([retval, str(sim_id)])

        return '.'.join([retval, "npy"])


    def get_spectrum(self, info_component, info_combination, represent, sim_id=None, prefix=None):
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert info_combination in ["combined", "perfreq"]
        assert represent in ["Cl", "Dl"]
        assert type(sim_id) == int or sim_id is None

        dir_map_loc, dir_spec_loc, dir_misc_loc = self._get_dir(info_component, sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_spec_loc
        filename_loc = self._get_specfilename(info_component, info_combination, represent, sim_id, prefix=prefix)

        return path.join(dir_spec_loc, filename_loc)


    def get_map(self, info_component, info_combination, represent, sim_id=None, prefix=None):
        assert 0, 'To be implemented'
        assert info_component in ["noise", "foreground", "signal", "non-sep"]
        assert info_combination in ["combined", "perfreq"]
        assert represent in ["Cl", "Dl"]
        assert type(sim_id) == int or sim_id is None

        dir_map_loc, dir_spec_loc, dir_misc_loc = self._get_dir(info_component, sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_map_loc
        filename_loc = self._get_mapfilename(info_component, info_combination, represent, sim_id, prefix=prefix)

        return path.join(dir_map_loc, filename_loc)


    def get_misc(self, desc, sim_id=None, prefix=None):
        assert 0, 'To be implemented'
        assert type(sim_id) == int or sim_id is None

        dir_map_loc, dir_spec_loc, dir_misc_loc = self._get_miscdir(sim_id)
        if prefix is None:
            pass
        else:
            prefix = dir_misc_loc
        filename_loc = self._get_miscfilename(desc, sim_id, prefix=prefix)

        return path.join(dir_misc_loc, filename_loc)

class Constants:

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

class Helperfunctions:

    llp1e12 = lambda x: x*(x+1)*1e12/(2*np.pi)

    @staticmethod
    def bin_it(data, bins=Constants.SMICA_lowell_bins):
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
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Call to deprecated function {}.".format(func.__name__),
                        category=DeprecationWarning,
                        stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return new_func