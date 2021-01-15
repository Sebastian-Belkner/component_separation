import healpy as hp
q = hp.read_map("data/map/frequency/HFI_SkyMap_100-field_2048_R3.00_full.fits", field=1, dtype=None)
u = hp.read_map("data/map/frequency/HFI_SkyMap_100-field_2048_R3.00_full.fits", field=2, dtype=None)
pnside = hp.npix2nside(q.size)
lmax = 4000
lmax_mask = 8000
elm, blm = hp.map2alm_spin(
    [q, u],
    2,
    lmax=min(3 * pnside - 1, lmax + lmax_mask))