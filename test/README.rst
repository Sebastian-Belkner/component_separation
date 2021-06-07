Component Separation - Test
====================================

Testing results of the powerspectra and weightings using modified .fits file (found in `/data`). We use modified files to save storage capacity, computational time, but still be able to deliver a weighting-calculation-test to anyone cloning this repository
This submodule is in development. To create a sensible test, generate modified .fits files and put them into the correct directories as indicated. Set path='test/data/' in the `run.py` file and run,
 ```
 python3 component_separation/run.py
 ```

GitHub actions testing is in development and not fully supported.

Usage
===========================

To create the modified .fits-files, adapt and use `createfitsfiles.py`


Validating smica results
=============================

To validate the smica output,

    1. `run_map.py` to generate noise maps.
        it would be better to use the noise maps from the planck sim suite
    2. `run_powspec.py` to generate powerspectra planck maps, e.g. planck sim suite
    3. `run_powerspec_and_sim.py` to get smica cmb spectrum
        need noise estimate, cmb estimate, and powerspectra input. all three should be
        taken from planck sim suite
    4. `validate_smica.py` to calculate the transferfunction. takes cmb powerspectrum and smica-output as input.
        the cmb powerspectrum should be taken from planck sim suite, but cannot find the data..
    5. `run_draw.py` to plot the transferfunction 
