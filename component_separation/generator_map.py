import healpy as hp
import numpy as np

    
def alm_s2map(tlm, elm, blm, nsi):
    return hp.alm2map([cmb_tlm, cmb_elm, cmb_blm], nsi)


def create_difference_map(data_hm1, data_hm2):
    def _difference(data1, data2):
        ret = dict()
        for freq, val in data1.items():
            ret.update({freq: (data1[freq] - data2[freq])/2.})
        return ret
    ret_data = _difference(data_hm1, data_hm2)
    return ret_data