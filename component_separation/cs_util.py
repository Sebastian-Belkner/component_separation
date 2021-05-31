#provides some constants and file formatters

from enum import Enum
import json
import itertools
import numpy as np
import os
import component_separation

with open(os.path.dirname(component_separation.__file__)+'/config.json', "r") as f:
    cf = json.load(f)


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
    TT = "TT"
    EE = "EE"
    BB = "BB"
    TE = "TE"
    TB = "TB"
    EB = "EB"
    ET = "ET"
    BT = "BT"
    BE = "BE"


class Planckr(Enum):
    LFI = "LFI"
    HFI = "HFI"


class Config(object):
    PLANCKMAPFREQ = [p.value for p in list(Planckf)]
    PLANCKSPECTRUM = [p.value for p in list(Plancks)]
    freqcomb =  ["{}-{}".format(FREQ,FREQ2)
        for FREQ, FREQ2  in itertools.product(PLANCKMAPFREQ,PLANCKMAPFREQ)
            if FREQ not in cf['pa']["freqfilter"] and
            (FREQ2 not in cf['pa']["freqfilter"]) and (int(FREQ2)>=int(FREQ))]
    speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in cf['pa']["specfilter"]]
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
])


class Helperfunctions:

    llp1e12 = lambda x: x*(x+1)*1e12/(2*np.pi)

    @staticmethod
    def bin_it(data, bins=Constants.SMICA_lowell_bins, offset=0):
        ret = np.ones((*data.shape[:-1], len(bins)))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(bins.shape[0]):
                    ret[i,j,k] = np.mean(np.nan_to_num(data[i,j,offset+int(bins[k][0]):offset+int(bins[k][1])]))
        return np.nan_to_num(ret)

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

# K_CMB to K_RJ conversion factors

# Instrument  Factor
# ------------------------------
# 030         0.9770745747972885
# 044         0.951460364644704
# 070         0.8824988010513074
# 100         0.7773000112639914
# 143         0.6048402898008157
# 217         0.3344250134127003
# 353         0.07754853977491984
# 545         0.006267933567707104
# 857         6.378414208594617e-05