import healpy as hp
import numpy as np

    
def alm_s2map(tlm, elm, blm, nsi):
    return hp.alm2map([tlm, elm, blm], nsi)


def create_difference_map(data_hm1, data_hm2):
    def _difference(data1, data2):
        ret = dict()
        for freq, val in data1.items():
            ret.update({freq: (data1[freq] - data2[freq])/2.})
        return ret
    ret_data = _difference(data_hm1, data_hm2)
    return ret_data


def create_mapsyn(spectrum, cf, freqcomb):
    synmap = dict()
    for freqc in freqcomb:
        synmap.update({
            freqc: hp.synfast(
                cls = [
                    spectrum[freqc]["TT"],
                    spectrum[freqc]["EE"],
                    spectrum[freqc]["BB"],
                    spectrum[freqc]["TE"]],
                nside = cf.nside[0] if int(freqc.split("-")[0])<100 else cf.nside[1],
                new=True)})

    #TODO return only numpy
    return np.array([])