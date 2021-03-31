# %% Section I
"""
                                    ##########################################
                                    ##########################################
                                                    SECTION 1
                                    ##########################################
                                    ##########################################
""" 

# %% Load Spectra, currently this includes "smallpatch", "largepatch", and fullsky. All maps are however smica-masked.
with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/component_separation/draw/draw.json', "r") as f:
    dcf = json.load(f)
freqfilter = dcf['pa']["freqfilter"]
specfilter = dcf['pa']["specfilter"]
freqcomb =  [
    "{}-{}".format(FREQ,FREQ2)
        for FREQ in PLANCKMAPFREQ
        if FREQ not in freqfilter
        for FREQ2 in PLANCKMAPFREQ
        if (FREQ2 not in freqfilter) and (int(FREQ2)>=int(FREQ))]

dc = dcf["plot"]["spectrum"]
def _inpathname(freqc,spec, fname):
    return  "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+dc["indir_root"]+dc["indir_rel"]+spec+freqc+"-"+dc["in_desc"]+fname

dcf["pa"]["mskset"] =  "mask-spatch-smica"
fname = io.make_filenamestring(dcf)
speccs =  [spec for spec in PLANCKSPECTRUM if spec not in specfilter]
sspectrum = {freqc: {
    spec: np.array(io.load_cl(_inpathname(freqc,spec, fname)))
    for spec in speccs}  
    for freqc in freqcomb}

dcf["pa"]["mskset"] =  "mask-lpatch-smica"
fname = io.make_filenamestring(dcf)
lspectrum = {freqc: {
    spec: np.array(io.load_cl(_inpathname(freqc,spec, fname)))
    for spec in speccs}  
    for freqc in freqcomb}

dcf["pa"]["mskset"] =  "smica"
fname = io.make_filenamestring(dcf)
dc = dcf["plot"]["spectrum"]
inpath_name = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"+dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname
allspectrum = io.load_spectrum(inpath_name, fname)
allC_l  = pw.build_covmatrices(allspectrum, lmax, freqfilter, specfilter)


# %% Build Auto and Cross covariance Matrix
sC_l = np.nan_to_num(pw.build_covmatrices(sspectrum, lmax, freqfilter, specfilter)["EE"].T)[2:2000]
lC_l = np.nan_to_num(pw.build_covmatrices(lspectrum, lmax, freqfilter, specfilter)["EE"].T)[2:2000]
allC_l = np.nan_to_num(pw.build_covmatrices(allspectrum, lmax, freqfilter, specfilter)["EE"].T)[2:2000]


# %% Calculate Optimal Powerspectrum
cov_min_1 = np.nan_to_num(calculate_minimalcov2(sC_l))
cov_min_2 = np.nan_to_num(calculate_minimalcov2(lC_l))
cov_min_3 = np.nan_to_num(calculate_minimalcov2(allC_l))

cov_min_1ma = ma.masked_array(cov_min_1, mask=np.where(cov_min_1<=0, True, False))
cov_min_2ma = ma.masked_array(cov_min_2, mask=np.where(cov_min_2<=0, True, False))


# %% Build weights for Inverse Variance Weighting
C_full = np.zeros((allC_l.shape[0],2,2), float)
C_full[:,0,0] = np.array(
    [(2*cov_min_1[l] * cov_min_1[l])/((2*l+1)*0.23)
    for l in range(allC_l.shape[0])])
C_full[:,1,1] = np.array(
    [(2*cov_min_2[l] * cov_min_2[l])/((2*l+1)*0.71) for l in range(allC_l.shape[0])])
C_full[:,1,0] = 0*np.array([(2*cov_min_1[l] * cov_min_2[l])/((2*l+1)*np.sqrt(0.23*0.71))
        for l in range(allC_l.shape[0])])
C_full[:,0,1] = 0*np.array([(2*cov_min_2[l] * cov_min_1[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(allC_l.shape[0])])
print(C_full[10])


# %% Combine on alm level
import healpy as hp
alm1 = hp.synalm(cov_min_1)
alm2 = hp.synalm(cov_min_2)
weights_1 = np.zeros((C_full.shape[0], C_full.shape[1]))


# %%
for l in range(C_full.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_full[l,:,:]))
    except:
        pass
alm_opt_1 = np.zeros_like(alm1)
for idx in range(alm_opt_1.shape[0]):
    l, m = hp.Alm.getlm(C_full.shape[0]-1, idx)
    alm_opt_1[idx] = weights_1[l,0]*alm1[idx] + weights_1[l,1]*alm2[idx]


# %%
opt_12 = hp.alm2cl(alm_opt_1)
opt_12ma = ma.masked_array(opt_12, mask=np.where(opt_12<=0, True, False))


# %% Calculate combined powerspectrum via inverse variance weighting
weights_1 = np.zeros((C_full.shape[0], C_full.shape[1]))
for l in range(C_full.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_full[l,:,:]))
    except:
        pass
opt_1 = np.array([weights_1[l] @ np.array([cov_min_1, cov_min_2])[:,l] for l in range(C_full.shape[0])])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Plot
bins = np.logspace(np.log(1)/np.log(base), np.log(C_full.shape[0]+1)/np.log(base), nbins, base=base)
plt.figure(figsize=(8,6))
print("opt1_ma:")
mean, std, _ = std_dev_binned(opt_1ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="Combined spectrum on Cl level, weights*powerspectra",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    markersize=10,
    # color=CB_color_cycle[idx],
    alpha=0.9)


print("opt12_ma:")
mean, std, _ = std_dev_binned(opt_12ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="Combined spectrum on alm level, weights*powerspectra",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    markersize=10,
    # color=CB_color_cycle[idx],
    alpha=0.9)


print("cov_min_1ma")
mean, std, _ = std_dev_binned(cov_min_1ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="low noise patch",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    # color=CB_color_cycle[idx],
    alpha=0.9)

print("cov_min_2ma")
mean, std, _ = std_dev_binned(cov_min_2ma, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    yerr=std,
    label="high noise patch",# + freqc,
    capsize=3,
    elinewidth=2,
    fmt='x',
    # color=CB_color_cycle[idx],
    alpha=0.9)


plt.plot(spectrum_trth, label="Planck EE spectrum")
# plt.plot(cov_min_3, label = "all sky", lw=2, alpha=0.7)
# plt.plot(C_p1[:,5,5])
# plt.plot(C_p2[:,5,5]


plt.xscale("log", nonpositive='clip')
plt.yscale("log", nonpositive='clip')
plt.xlabel("Multipole l")
plt.ylabel("Powerspectrum")
plt.grid(which='both', axis='x')
plt.grid(which='major', axis='y')

plt.xlim((3e1,2e3))
plt.ylim((1e-1,1e2))

plt.legend()
plt.savefig('skypatches.jpg')
plt.show()


# %% Look at Variance of the powerspectra
plt.figure(figsize=(8,6))
print("opt1_ma:")
da = np.zeros((C_full.shape[0]))
for l in range(C_full.shape[0]):
    try:
        da[l] = np.nan_to_num(
            calculate_minimalcov2(C_full[l]))
    except:
        pass
dama = ma.masked_array(da, mask=np.where(da<=0, True, False))
mean, std, _ = std_dev_binned(dama, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    # yerr=std,
    label="Combined minimal variance",# + freqc,
    markersize=10,
    alpha=0.9)

print("cov_min_1ma")
dat = ma.masked_array(C_full[:,0,0], mask=np.where(C_full[:,0,0]<=0, True, False))
mean, std, _ = std_dev_binned(dat, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    # yerr=std,
    label="variance, low noise patch",# + freqc,
    # color=CB_color_cycle[idx],
    alpha=0.9)

print("cov_min_2ma")
dat = ma.masked_array(C_full[:,1,1], mask=np.where(C_full[:,1,1]<=0, True, False))
mean, std, _ = std_dev_binned(dat, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    # yerr=std,
    label="variance, high noise patch",# + freqc,
    # color=CB_color_cycle[idx],
    alpha=0.9)

print("cov_min_3")
fullskyvar = np.array(
    [(2*cov_min_3[l] * cov_min_3[l])/((2*l+1)*0.96)
    for l in range(allC_l.shape[0])])
dat = ma.masked_array(fullskyvar, mask=np.where(fullskyvar<=0, True, False))
mean, std, _ = std_dev_binned(dat, bins)
plt.errorbar(
    (_[1:] + _[:-1])/2,
    mean,
    # yerr=std,
    label="variance, all sky",# + freqc,
    # color=CB_color_cycle[idx],
    alpha=0.9)

plt.xscale("log", nonpositive='clip')
plt.yscale("log", nonpositive='clip')
plt.xlabel("Multipole l")
plt.ylabel("Variance / C_l")
plt.grid(which='both', axis='x')
plt.grid(which='major', axis='y')

plt.xlim((1e1,2e3))
plt.ylim((1e-3,1e2))
plt.legend()
plt.show()

# %%
dama = ma.masked_array(da, mask=np.where(da<=0, True, False))
mean_cp, std_cp, _ = std_dev_binned(dama, bins)

fullskyvar = np.array(
    [(2*cov_min_3[l] * cov_min_3[l])/((2*l+1)*0.96)
    for l in range(allC_l.shape[0])])
dat = ma.masked_array(fullskyvar, mask=np.where(fullskyvar<=0, True, False))
mean_fs, std, _ = std_dev_binned(dat, bins)
plt.plot((_[1:] + _[:-1])/2, (mean_fs-mean_cp)/mean_fs)
plt.ylim((0,1))
plt.xlim((200,2000))
plt.show()


# %% Section II
"""
                                    ##########################################
                                    ##########################################
                                                    SECTION 2
                                    ##########################################
                                    ##########################################
""" 


# %% plot noise+signal
plt.figure(figsize=(8,6))
for n in range(C_lN["EE"].shape[1]):
    plt.plot(C_lN["EE"].T[:,n,n], label='Noise {} Channel'.format(PLANCKMAPFREQ[n]))

plt.plot(C_lS[:,0,0])
plt.title("EE noise andd signal spectra")
plt.xscale('log')
plt.yscale('log')
plt.xlim((10,lmax))
plt.legend()
# plt.ylim((1e-2,1e6))
plt.show()




# %% Build toy-patches
C_lN1 = C_lN["EE"].T[2:] * 1./3.
C_lN2 = C_lN["EE"].T[2:] * 2./3.
C_lN3 = C_lN["EE"].T[2:]
C_p1 = C_lS + C_lF[2:] + C_lN1
C_p2 = C_lS + C_lF[2:] + C_lN2
C_p3 = C_lS + C_lF[2:] + C_lN3
cov_min_1ana = np.nan_to_num(calculate_minimalcov(C_lN1, C_lS[2:], C_lF[2:]))
cov_min_2ana = np.nan_to_num(calculate_minimalcov(C_lN2, C_lS[2:], C_lF[2:]))
cov_min_3ana = np.nan_to_num(calculate_minimalcov(C_lN3, C_lS[2:], C_lF[2:]))

C_fullana = np.zeros((lmax-1,2,2), float)
C_fullana[:,0,0] = np.array([(2*cov_min_1ana[l] * cov_min_1ana[l])/((2*l+1)*0.23) for l in range(C_fullana.shape[0])])
C_fullana[:,1,1] = np.array([(2*cov_min_2ana[l] * cov_min_2ana[l])/((2*l+1)*0.71) for l in range(C_fullana.shape[0])])
C_fullana[:,1,0] = 0*np.array([(2*cov_min_1ana[l] * cov_min_2ana[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(C_fullana.shape[0])])
C_fullana[:,0,1] = 0*np.array([(2*cov_min_2ana[l] * cov_min_1ana[l])/((2*l+1)*np.sqrt(0.23*0.71)) for l in range(C_fullana.shape[0])])



# %% Idea 1: patched cov_{min}_l = sum_i weights_il * C_{full}_l, with weights_il = cov(c,c)^{-1}1/(1^Tcov(c,c)^{-1}1)
weights_1 = np.zeros((C_fullana.shape[0], C_fullana.shape[1]))
for l in range(C_fullana.shape[0]):
    try:
        # np.linalg.inv()
        weights_1[l] = calculate_weights(np.linalg.inv(C_fullana[l,:,:]))
    except:
        pass
opt_1 = np.array([weights_1[l] @ np.array([cov_min_1ana, cov_min_2ana])[:,l] for l in range(C_fullana.shape[0])])
opt_1ma = ma.masked_array(opt_1, mask=np.where(opt_1<=0, True, False))


# %% Idea 3: cov_{min} =  calculate_minimalcov2((Cp1_l, Cp2_l))
opt_3 = np.zeros((C_fullana.shape[0]))
for l in range(C_fullana.shape[0]):
    try:
        opt_3[l] = np.nan_to_num(
            calculate_minimalcov2(np.array([
                [cov_min_1ana[l], 0],
                [0, cov_min_2ana[l]]])))
    except:
        pass
opt_3ma = ma.masked_array(opt_3, mask=np.where(opt_3<=0, True, False))


# %% Idea 4: cov_{min} =  calculate_minimalcov2(cosmic_variance(Cp1_l, Cp2_l))
opt_4 = np.zeros((C_fullana.shape[0]))
for l in range(C_fullana.shape[0]):
    try:
        opt_4[l] = np.nan_to_num(
            calculate_minimalcov2(C_fullana[l]))
    except:
        pass
opt_4ma = ma.masked_array(opt_4, mask=np.where(opt_4<=0, True, False))



# %% plot the combination
# plt.plot(opt_1ma, label = 'Combined spectrum, weights*powerspectra', alpha=0.5, lw=3)
plt.plot(opt_3ma, label='minimalcov(cosmic_variance)', alpha=0.5, lw=3)
plt.plot(C_fullana[:,0,0], label= "cosmic variance low noise patch")
plt.plot(C_fullana[:,1,1], label= "cosmic variance high noise patch")

plt.plot(opt_4ma, label='minimalcov(powerspectra)', alpha=0.5, lw=3)
plt.plot(spectrum_trth, label="Planck EE spectrum")
plt.plot(cov_min_1ana, label = "Powerspectrum low noise patch", lw=1)
plt.plot(cov_min_2ana, label = "Powerspectrum high noise patch", lw=1)
# plt.plot(cov_min_3ana, label = "all noise", lw=1)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1e-4,1e3))
plt.xlim((2e1,3e3))
plt.legend()


# %% plot the variance of the Spectra

plt.plot(opt_1ma, label = 'Combined spectrum, weights*powerspectra', alpha=0.5, lw=3)
# plt.plot(opt_2ma, label='minimalcov(minimalpowerspectra)', alpha=0.5, ls='--', lw=3)
plt.plot(spectrum_trth, label="Planck EE spectrum")
# plt.plot(cov_min_patched, label = 'patched 4/5 + 1/5 noise', alpha=0.5)
plt.plot(cov_min_1ana, label = "low noise patch", lw=1)
plt.plot(cov_min_2ana, label = "high noise patch", lw=1)
plt.plot(cov_min_3ana, label = "all noise", lw=1)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1e-1,1e2))
plt.xlim((2e1,2e3))
plt.legend()


# %% Section III