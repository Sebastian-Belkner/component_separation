#provides some constants and file formatters

from enum import Enum

class Planckf(Enum):
    LFI_1 = '030'
    LFI_2 = '070'
    HFI_1 = '100'
    HFI_2 = '143'
    HFI_3 = '217'
    HFI_4 = '353'
    HFI_5 = '545'
    HFI_6 = '857'

class Plancks(Enum):
    TT = "TT"
    EE = "EE"
    BB = "BB"
    TE = "TE"
    TB = "TB"
    EB = "EB"
    ET = "ET"
    BT = "BT"
    BE = "BE"