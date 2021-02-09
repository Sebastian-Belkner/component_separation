"""
test_spectrum.py: compare powerspectrum as being calculated by `powspec.py` to planck simulations to estimate the correctness

 - take planck simulation data, includes powspec_planck and frequency maps, map_planck
 - use map_planck to calculate component separation power spectra, powspec_cs
 - compare powspec_cs with powspec_planck
 - quantify
 - (make statistic over n simulations)

TODO how to take noise into account?
"""