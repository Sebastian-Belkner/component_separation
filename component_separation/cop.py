def build_covmatrices(df: Dict, lmax: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, np.ndarray]:
    """Calculates the covariance matrices from the data
    Args:
        df (Dict): A "2D"-DataFrame of powerspectra with spectrum and frequency-combinations in the columns
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
    Returns:
        Dict[str, np.ndarray]: The covariance matrices of Dimension [Nspec,Nspec,lmax]
    """
    NFREQUENCIES = len([FREQ for FREQ in PLANCKMAPFREQ if FREQ not in freqfilter])
    cov = {spec: np.zeros(shape=(NFREQUENCIES, NFREQUENCIES, lmax+1))
                for spec in PLANCKSPECTRUM if spec not in specfilter}

    ifreq, ifreq2, ispec = -1, -1, 0
    for FREQ in PLANCKMAPFREQ:
        ifreq2 = -1
        if FREQ not in freqfilter:
            ifreq+=1
            for FREQ2 in PLANCKMAPFREQ:
                if FREQ2 not in freqfilter:
                    if int(FREQ2) >= int(FREQ):
                        ifreq2+=1
                        ispec=-1
                        for spec in PLANCKSPECTRUM:
                            if spec not in specfilter:
                                ispec+=1
                                cov[spec][ifreq][ifreq2] = df[spec][FREQ+'-'+FREQ2]
                                cov[spec][ifreq2][ifreq] = df[spec][FREQ+'-'+FREQ2]
    return cov

#%% slice along l (3rd axis) and invert
@log_on_start(INFO, "Starting to invert convariance matrix {cov}")
@log_on_end(DEBUG, "Inversion successful: '{result}' ")
def invert_covmatrices(cov: Dict[str, np.ndarray], lmax: int, freqfilter: List[str], specfilter: List[str]):
    """Inverts a covariance matrix
    Args:
        cov (Dict[str, np.ndarray]): The covariance matrices of Dimension [Nspec,Nspec,lmax]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
    """
    def is_invertible(a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    cov_inv_l = {
        spec: {
            l: np.linalg.inv(cov[spec][:,:,l])
                for l in range(lmax) if is_invertible(cov[spec][:,:,l])
            }for spec in PLANCKSPECTRUM 
                if spec not in specfilter
        }
    return cov_inv_l

# %% Calculate weightings and store in df
@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: Dict, lmax: int, freqfilter: List[str], specfilter: List[str]) -> Dict[str, DataFrame]:
    """Calculates weightings of the respective Frequency channels
    Args:
        cov (Dict): The inverted covariance matrices of Dimension [Nspec,Nspec,lmax]
        lmax (int): Maximum multipol of data to be considered
        freqfilter (List[str]): Frequency channels which are to be ignored
        specfilter (List[str]): Bispectra which are to be ignored, e.g. ["TT"]
    Returns:
        Dict[str, DataFrame]: The weightings of the respective Frequency channels
    """
    elaw = np.ones(len([dum for dum in PLANCKMAPFREQ if dum not in freqfilter]))
    weighting = {spec: np.array([(cov[spec][l] @ elaw) / (elaw.T @ cov[spec][l] @ elaw)
                        for l in range(lmax) if l in cov[spec].keys()])
                    for spec in PLANCKSPECTRUM if spec not in specfilter}
    weights = {spec:
                pd.DataFrame(
                    data=weighting[spec],
                    columns = ["channel @{}GHz".format(FREQ)
                                for FREQ in PLANCKMAPFREQ
                                if FREQ not in freqfilter])
                    for spec in PLANCKSPECTRUM
                    if spec not in specfilter}
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter:
            weights[spec].index.name = 'multipole'
    return weights