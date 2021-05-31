# %%
import numpy as np
import matplotlib.pyplot as plt

# name = "K_CMBNPIPE-msk_smica-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE_split-Full.npy"
# name = "K_CMBNPIPE-msk_lens-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE_split-Full.npy"
name = "K_CMB-NPIPE-msk_lens-lmax_3000-lmaxmsk_6000-smoothing_5_2-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE_split-Full.npy"
data = np.load(name)

print('test wip')
print(data)


# %%
plt.plot(data[1,:,0])
plt.ylim((-0.2,1))
# plt.xlim((900,920))


# %%
a = np.where(data[1,:,2]==0)
print(data.shape)
print(a[0].shape)
print(a)

print(data[1,:,0])


# %%
