# %%
import dvc.api
import numpy as np
with dvc.api.open(
        'data/tmp/weights/weightsK_CMBNPIPE-msk_lens-lmax_3000-lmaxmsk_6000-freqs_030,044,070,100,143,217,353_specs-TT,EE,BB,TE.npy',
        # repo='https://github.com/Sebastian-Belkner/component_separation.git',
        mode = 'rb',
        encoding='utf-8') as f:
    buffer = f.read()
    weights = np.frombuffer(buffer, dtype=np.float64, offset=128)
    weights = weights.reshape(4,7,-1)



print(weights)
# %%
import os
print(os.getcwd())
# %%
