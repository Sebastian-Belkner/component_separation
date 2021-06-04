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

    PLANCKMAPFREQ_f = [FREQ for FREQ in PLANCKMAPFREQ
        if FREQ not in cf['pa']["freqfilter"]]
    PLANCKSPECTRUM_f = [SPEC for SPEC in PLANCKSPECTRUM
        if SPEC not in cf['pa']["specfilter"]]


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

    linear_equisized_bins_10 = np.array([
        [0, 9.999], [10, 19.999], [20, 29.999],  [30, 39.999], [40, 49.999], [50, 59.999], [60, 69.999], [70, 79.999], [80, 89.999], [90, 99.999], [100, 109.999], [110, 119.999], [120, 129.999], [130, 139.999],
        [140, 149.999], [150, 159.999], [160, 169.999], [170, 179.999], [180, 189.999], [190, 199.999], [200, 209.999], [210, 219.999], [220, 229.999], [230, 239.999], [240, 249.999], [250, 259.999],
        [260, 269.999], [270, 279.999], [280, 289.999], [290, 299.999], [300, 309.999], [310, 319.999], [320, 329.999], [330, 339.999], [340, 349.999], [350, 359.999], [360, 369.999], [370, 379.999],
        [380, 389.999], [390, 399.999], [400, 409.999], [410, 419.999], [420, 429.999], [430, 439.999], [440, 449.999], [450, 459.999], [460, 469.999], [470, 479.999], [480, 489.999], [490, 499.999],
        [500, 509.999], [510, 519.999], [520, 529.999], [530, 539.999], [540, 549.999], [550, 559.999], [560, 569.999], [570, 579.999], [580, 589.999], [590, 599.999], [600, 609.999], [610, 619.999],
        [620, 629.999], [630, 639.999], [640, 649.999], [650, 659.999], [660, 669.999], [670, 679.999], [680, 689.999], [690, 699.999], [700, 709.999], [710, 719.999], [720, 729.999], [730, 739.999],
        [740, 749.999], [750, 759.999], [760, 769.999], [770, 779.999], [780, 789.999], [790, 799.999], [800, 809.999], [810, 819.999], [820, 829.999], [830, 839.999], [840, 849.999], [850, 859.999],
        [860, 869.999], [870, 879.999], [880, 889.999], [890, 899.999], [900, 909.999], [910, 919.999], [920, 929.999], [930, 939.999], [940, 949.999], [950, 959.999], [960, 969.999], [970, 979.999],
        [980, 989.999], [990, 999.999], [1000, 1009.999], [1010, 1019.999], [1020, 1029.999], [1030, 1039.999], [1040, 1049.999], [1050, 1059.999], [1060, 1069.999], [1070, 1079.999], [1080, 1089.999],
        [1090, 1099.999], [1100, 1109.999], [1110, 1119.999], [1120, 1129.999], [1130, 1139.999], [1140, 1149.999], [1150, 1159.999], [1160, 1169.999], [1170, 1179.999], [1180, 1189.999], [1190, 1199.999],
        [1200, 1209.999], [1210, 1219.999], [1220, 1229.999], [1230, 1239.999], [1240, 1249.999], [1250, 1259.999], [1260, 1269.999], [1270, 1279.999], [1280, 1289.999], [1290, 1299.999], [1300, 1309.999],
        [1310, 1319.999], [1320, 1329.999], [1330, 1339.999], [1340, 1349.999], [1350, 1359.999], [1360, 1369.999], [1370, 1379.999], [1380, 1389.999], [1390, 1399.999], [1400, 1409.999], [1410, 1419.999],
        [1420, 1429.999], [1430, 1439.999], [1440, 1449.999], [1450, 1459.999], [1460, 1469.999], [1470, 1479.999], [1480, 1489.999], [1490, 1499.999], [1500, 1509.999], [1510, 1519.999], [1520, 1529.999],
        [1530, 1539.999], [1540, 1549.999], [1550, 1559.999], [1560, 1569.999], [1570, 1579.999], [1580, 1589.999], [1590, 1599.999], [1600, 1609.999], [1610, 1619.999], [1620, 1629.999], [1630, 1639.999],
        [1640, 1649.999], [1650, 1659.999], [1660, 1669.999], [1670, 1679.999], [1680, 1689.999], [1690, 1699.999], [1700, 1709.999], [1710, 1719.999], [1720, 1729.999], [1730, 1739.999], [1740, 1749.999],
        [1750, 1759.999], [1760, 1769.999], [1770, 1779.999], [1780, 1789.999], [1790, 1799.999], [1800, 1809.999], [1810, 1819.999], [1820, 1829.999], [1830, 1839.999], [1840, 1849.999], [1850, 1859.999],
        [1860, 1869.999], [1870, 1879.999], [1880, 1889.999], [1890, 1899.999], [1900, 1909.999], [1910, 1919.999], [1920, 1929.999], [1930, 1939.999], [1940, 1949.999], [1950, 1959.999], [1960, 1969.999],
        [1970, 1979.999], [1980, 1989.999], [1990, 1999.999], [2000, 2009.999], [2010, 2019.999], [2020, 2029.999], [2030, 2039.999], [2040, 2049.999], [2050, 2059.999], [2060, 2069.999], [2070, 2079.999],
        [2080, 2089.999], [2090, 2099.999], [2100, 2109.999], [2110, 2119.999], [2120, 2129.999], [2130, 2139.999], [2140, 2149.999], [2150, 2159.999], [2160, 2169.999], [2170, 2179.999], [2180, 2189.999],
        [2190, 2199.999], [2200, 2209.999], [2210, 2219.999], [2220, 2229.999], [2230, 2239.999], [2240, 2249.999], [2250, 2259.999], [2260, 2269.999], [2270, 2279.999], [2280, 2289.999], [2290, 2299.999],
        [2300, 2309.999], [2310, 2319.999], [2320, 2329.999], [2330, 2339.999], [2340, 2349.999], [2350, 2359.999], [2360, 2369.999], [2370, 2379.999], [2380, 2389.999], [2390, 2399.999], [2400, 2409.999], 
        [2410, 2419.999], [2420, 2429.999], [2430, 2439.999], [2440, 2449.999], [2450, 2459.999], [2460, 2469.999], [2470, 2479.999], [2480, 2489.999], [2490, 2499.999], [2500, 2509.999], [2510, 2519.999],
        [2520, 2529.999], [2530, 2539.999], [2540, 2549.999], [2550, 2559.999], [2560, 2569.999], [2570, 2579.999], [2580, 2589.999], [2590, 2599.999], [2600, 2609.999], [2610, 2619.999], [2620, 2629.999],
        [2630, 2639.999], [2640, 2649.999], [2650, 2659.999], [2660, 2669.999], [2670, 2679.999], [2680, 2689.999], [2690, 2699.999], [2700, 2709.999], [2710, 2719.999], [2720, 2729.999], [2730, 2739.999],
        [2740, 2749.999], [2750, 2759.999], [2760, 2769.999], [2770, 2779.999], [2780, 2789.999], [2790, 2799.999], [2800, 2809.999], [2810, 2819.999], [2820, 2829.999], [2830, 2839.999], [2840, 2849.999],
        [2850, 2859.999], [2860, 2869.999], [2870, 2879.999], [2880, 2889.999], [2890, 2899.999], [2900, 2909.999], [2910, 2919.999], [2920, 2929.999], [2930, 2939.999], [2940, 2949.999], [2950, 2959.999],
        [2960, 2969.999], [2970, 2979.999], [2980, 2989.999]])
    
    linear_equisized_bins_100 = np.array([
        [0, 99.999], [100, 199.999], [200, 299.999], [300, 399.999], [400, 499.999], [500, 599.999], [600, 699.999], [700, 799.999], [800, 899.999], [900, 999.999],
        [1000, 1099.999], [1100, 1199.999], [1200, 1299.999], [1300, 1399.999], [1400, 1499.999], [1500, 1599.999], [1600, 1699.999], [1700, 1799.999], [1800, 1899.999],
        # [1900, 1999.999], [2000, 2099.999], [2100, 2199.999], [2200, 2299.999], [2300, 2399.999], [2400, 2499.999], [2500, 2599.999], [2600, 2699.999], [2700, 2799.999],
        # [2800, 2899.999]
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