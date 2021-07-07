dev
===============

Collection of files for development of this software..





structure
===============

 * maps, alms, spectra



maps
********
preprocessing (sz, dead pixels, ..)

attributes
++++++++++++
DX12 or NPIPE
TEB or TQU
signal, fg, full, or noise
nside

processing
+++++++++++++++
create a map from other maps (e.g. noise)
calculate alms, cl from map
derive mask from map


in and output
+++++++++++++++


alms
********






wishlist
=============
 * unit test
 * run with subset?



pipeline
=========
run_map to generate noise maps. they vary in nside.
noise maps are preprocessed

`python3 run_map.py #check rm_config.py beforehand`


for all input specified in rp_config.json
    run_powspec to generate noise powerspectrum
    run_powspec to generate empiric/simulation data powerspectrum
    run_cmpmap to generate signal data powerspectrum

they vary in postprocessed