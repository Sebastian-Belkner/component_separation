# ffp10 == full focal plane
# use this data to derive transferfunction

class cmb_len_ffp10:
    """ FFP10 input sim libraries, lensed alms. """
    def __init__(self):
        pass

    def hashdict(self):
        return {'sim_lib': 'ffp10 lensed scalar cmb inputs'}

    @staticmethod
    def get_sim_tlm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=1)

    @staticmethod
    def get_sim_elm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=2)

    @staticmethod
    def get_sim_blm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=3)

