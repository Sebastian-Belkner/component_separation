#provides some constants and file formatters

from enum import Enum

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