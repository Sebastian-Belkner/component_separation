import sys
sys.path.insert(0, '../params')
# import idealized_example as parfile
import smicadx12_dd_seb as parfile_seb
import smicadx12_planck2018 as parfile
import healpy as hp
import numpy as np
import pylab as pl
from plancklens import utils

qlm = parfile.qlms_dd.get_sim_qlm('p_p', -1)
