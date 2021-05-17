# %%
import numpy as np
import matplotlib.pyplot as plt

# name = "K_CMBNPIPE-msk_smica-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE_split-Full.npy"
name = "K_CMBNPIPE-msk_lens-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE_split-Full.npy"
data = np.load(name, allow_pickle=True).item()

print('test wip')
print(data)
# %%
plt.plot(data["EE"])
plt.ylim((-0.2,1))
# plt.xlim((900,920))

# %%
