"""
interactive.py: runner script for calling component_separation using jupyter

"""

__author__ = "S. Belkner"

# %% interactive header
import matplotlib.pyplot as plt

# %% run header
import json
import logging
import logging.handlers
import os
import platform
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import component_separation.io as io
import component_separation.MSC.MSC.pospace as ps
import component_separation.powspec as pw
from component_separation.cs_util import Planckf, Plancks

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/config.json', "r") as f:
    cf = json.load(f)

LOGFILE = 'data/tmp/messages.log'
logger = logging.getLogger("")
handler = logging.handlers.RotatingFileHandler(
        LOGFILE, maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]


mskset = cf['pa']['mskset'] # smica or lens
freqdset = cf['pa']['freqdset'] # DX12 or NERSC

lmax = cf['pa']["lmax"]
lmax_mask = cf['pa']["lmax_mask"]
llp1 = cf['pa']["llp1"]
bf = cf['pa']["bf"]

num_sim = cf['pa']["num_sim"]

spec_path = cf[mch]['outdir']
indir_path = cf[mch]['indir']
specfilter = cf['pa']["specfilter"]
    
def set_logger(loglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)


# %% parameters
freqfilter= [
        "030",
        "044",
        "070",
        "545",
        "857"
        ]
set_logger(DEBUG)
freqcomb =  [
    "{}-{}".format(FREQ,FREQ2)
        for FREQ in PLANCKMAPFREQ
        if FREQ not in freqfilter
        for FREQ2 in PLANCKMAPFREQ
        if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]
speccomb  = [spec for spec in PLANCKSPECTRUM if spec not in specfilter]

filename = '{freqdset}_lmax_{lmax}-lmaxmsk_{lmax_mask}-msk_{mskset}-freqs_{freqs}_specs-{spec}_split-{split}.npy'.format(
    freqdset = freqdset,
    lmax = lmax,
    lmax_mask = lmax_mask,
    mskset = mskset,
    spec = ','.join([spec for spec in PLANCKSPECTRUM if spec not in specfilter]),
    freqs = ','.join([fr for fr in PLANCKMAPFREQ if fr not in freqfilter]),
    split = "Full" if cf['pa']["freqdatsplit"] == "" else cf['pa']["freqdatsplit"])

#%%
beamf = io.load_beamf(freqcomb=freqcomb)
freq = ["100", "143", "217", "353"]
import itertools
import matplotlib.gridspec as gridspec
TEB = {
        0: "T",
        1: "E",
        2: "B"
    }

for p in itertools.product(freq, freq):
    if int(p[0])<int(p[1]):
        for field1 in [0,1,2]:
            for field2 in [0,1,2]:
                if field2 >= field1:
                    aa = "{}-{}".format(p[0], p[0])
                    bb = "{}-{}".format(p[1], p[1])
                    ab = "{}-{}".format(p[0], p[1])
                    ab2 = beamf[aa][1].data.field(field1) * beamf[bb][1].data.field(field2)

                    plt.figure(figsize=(8,6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
                    ax1 = plt.subplot(gs[0])
                    plt.title(r"$W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{{{{b}}},{{{d}}}}} = B^{{\mathtt{{{a}}}}}_{{\mathtt{{{b}}}}}(l) B^{{\mathtt{{{c}}}}}_{{\mathtt{{{d}}}}}(l)$".format(
                            a=TEB[field1],
                            b="f_1",
                            c=TEB[field2],
                            d="f_2"))
                    plt.plot(
                        ab2,
                        label = r"estimated $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[1]),
                        linewidth=3,
                        alpha=0.8)
                    plt.plot(
                        beamf[ab][1].data.field(field1) *beamf[ab][1].data.field(field2),
                        label = r"true $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[1]),
                        linewidth=3,
                        alpha=0.8)
                    plt.plot(
                        beamf[aa][1].data.field(field1)*beamf[aa][1].data.field(field2),
                        label = r"true $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[0]),
                        linewidth=1)
                    plt.plot(
                        beamf[bb][1].data.field(field1)* beamf[bb][1].data.field(field2),
                        label = r"true $W(l)^{{\mathtt{{{{{a}}}{{{c}}}}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[1],
                                c=TEB[field2],
                                d=p[1]),
                        linewidth=1)
                    plt.ylabel('Windowfunction')
                    plt.legend()
                    plt.grid()

                    ax2 = plt.subplot(gs[1])
                    plt.plot(
                        (ab2-beamf[ab][1].data.field(field1)*beamf[ab][1].data.field(field2))/ab2,
                        color = "#cc7000",
                        label = r"($W(l)^{{\mathtt{{{{{a}}}{{{c}}}, estimate}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}} - W(l)^{{\mathtt{{{{{a}}}{{{c}}}, truth}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}) / W(l)^{{\mathtt{{{{{a}}}{{{c}}}, estimate}}}}_{{\mathtt{{{{{b}}},{{{d}}}}}}}$".format(
                                a=TEB[field1],
                                b=p[0],
                                c=TEB[field2],
                                d=p[1]),
                        linewidth=2)
                    plt.ylim((0.0001,10))
                    plt.yscale("log", nonpositive='clip')
                    plt.grid()
                    plt.legend()
                    plt.xlabel("Multipole")
                    plt.ylabel('Rel. difference')
                    plt.savefig("vis/beamf/beamf_{}{}_{}.jpg".format(TEB[field1], TEB[field2], ab))
# %%

# %%
