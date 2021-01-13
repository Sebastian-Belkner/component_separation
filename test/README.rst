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
