import math
import numpy as np

from pyroomacoustics.doa import DOA
import matplotlib.pyplot as plt


class aMUSIC(DOA):
    """
    Class to apply MUltiple SIgnal Classication (MUSIC) direction-of-arrival
    (DoA) for a particular microphone array.
    .. note:: Run locate_source() to apply the MUSIC algorithm.
    Parameters
    ----------
    L: numpy array
        Microphone array positions. Each column should correspond to the
        cartesian coordinates of a single microphone.
    fs: float
        Sampling frequency.
    nfft: int
        FFT length.
    c: float
        Speed of sound. Default: 343 m/s
    num_src: int
        Number of sources to detect. Default: 1
    mode: str
        'far' or 'near' for far-field or near-field detection
        respectively. Default: 'far'
    r: numpy array
        Candidate distances from the origin. Default: np.ones(1)
    azimuth: numpy array
        Candidate azimuth angles (in radians) with respect to x-axis.
        Default: np.linspace(-180.,180.,30)*np.pi/180
    colatitude: numpy array
        Candidate elevation angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    frequency_normalization: bool
        If True, the MUSIC pseudo-spectra are normalized before averaging across the frequency axis, default:False
    """

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        alpha=2,
        p=1,
        mode="far",
        r=None,
        azimuth=None,
        colatitude=None,
        frequency_normalization=False,
        **kwargs
    ):

        DOA.__init__(
            self,
            L=L,
            fs=fs,
            nfft=nfft,
            c=c,
            num_src=num_src,
            mode=mode,
            r=r,
            azimuth=azimuth,
            colatitude=colatitude,
            **kwargs
        )

        self.alpha = alpha
        if self.alpha == 0:
            self.alpha_case = 'Tyler'
        elif self.alpha > 0 and self.alpha < 2:
            self.alpha_case = 'Fixed'
        elif self.alpha == 2.0:
            self.alpha_case = 'Gaussian'
        elif self.alpha > 2:
            self.alpha_case = 'Estimated'
        else:
            raise ValueError(f'Error in choosing alpha, got {alpha}')

        self.p = p
        self.Pssl = None
        self.frequency_normalization = frequency_normalization

    def compute_alpha(self, X_MFT):
        def factor_int(n):
            val = math.ceil(math.sqrt(n) / 2)
            val2 = 2 * int(n/val)
            while val2 * val != float(n):
                val -= 1
                val2 = int(n/val)
            return val, val2, n

        def shuffle_along_axis(a, axis):
            idx = np.random.rand(*a.shape).argsort(axis=axis)
            return np.take_along_axis(a, idx, axis=axis)

        K = int(self.F * self.T)

        K1, K2, _ = factor_int(K)
        rnd_X_FTM = shuffle_along_axis(X_MFT.transpose(1, 2, 0), axis=0)
        rnd_X_FTxM = np.array(shuffle_along_axis(rnd_X_FTM,
                                   axis=1)).reshape(-1, rnd_X_FTM.shape[-1])
        Y_K2M = np.zeros((K2, self.M)).astype(np.complex64)
        for k2 in range(K2):
            Y_K2M[k2] = rnd_X_FTxM[k2*K1:  K1 + k2*K1, :].sum(axis=0)
        logXnorm_FT = np.log(np.linalg.norm(rnd_X_FTxM, axis=-1))
        logYnorm_FT = np.log(np.linalg.norm(Y_K2M, axis=-1))
        alpha = (1 / np.log(K1) *
                      (1/K2 * logYnorm_FT.sum() -
                      1/K * logXnorm_FT.sum())) ** (-1)
        assert alpha > 0 or alpha < 2
        return alpha

    def compute_Ralpha_cov(self, X_MFT):
        """Estimate the elliptical covariance matrix of the K x N data S
        """

        self.M = X_MFT.shape[0]
        self.F = X_MFT.shape[1]
        self.T = X_MFT.shape[2]

        R_FMM = np.zeros((self.F,self.M,self.M),dtype=np.complex64)
        gamma = 0.5772156 # Euler constant
        Cste = -gamma * (self.alpha **(-1) - 1.) - np.log(2.0)

        mean_MF = np.mean(np.log(np.abs(X_MFT) + 1e-10), axis=-1)
        cov_MF = np.exp((mean_MF+Cste) * self.alpha ** (-1))
        diag_MF = 2. * (cov_MF) ** (2./self.alpha)
        for m in range(self.M):
            R_FMM[:, m, m] = diag_MF[m] 
        for m in range(self.M):
            for l in range(m):
                if l != m:
                    coeff_cov_F = (X_MFT[m,...] * X_MFT[l,...].conj() * (np.abs(X_MFT[l,...]) ** (self.p - 2.))).sum(axis=-1)
                    coeff_cov_F /= (np.abs(X_MFT[l]) ** (self.p) + 1e-10).sum(axis=-1)
                    R_FMM[:, m, l] = coeff_cov_F * cov_MF[l] * 2 ** (self.alpha/2.) * R_FMM[:, l, l] ** (-(self.alpha-2.)/2.)
                    R_FMM[:, l, m] = np.conj(R_FMM[:, m, l])
        return R_FMM

    def _process(self, X_MFT):
        """
        Perform MUSIC for given frame in order to estimate steered response
        spectrum.
        """
        self.minF = self.freq_hz[0]
        self.maxF = self.freq_hz[-1]

        X_MFT = X_MFT[:,self.freq_bins,:]
        
        self.M = X_MFT.shape[0]
        self.F = X_MFT.shape[1]
        self.T = X_MFT.shape[2]
        
        if self.alpha_case == 'Fixed':
            # print("Given alpha={}".format(self.alpha))
            R_FMM = self.compute_Ralpha_cov(X_MFT)

        elif self.alpha_case == 'Gaussian':
            R_FMM = (X_MFT[:, None] * X_MFT[None].conj()).mean(axis=-1).transpose(-1, 0, 1)

        elif self.alpha_case == 'Tyler':
            R_FMM = np.tile(np.eye(self.M), [self.F, 1, 1]).astype(np.complex64)

            for i in range(20):
                invR_FMM = np.linalg.inv(R_FMM)
                num_MMFT = X_MFT[:, None] * X_MFT[None].conj()
                den_FT =  (X_MFT[:, None].conj() *\
                        invR_FMM.transpose(-1, -2, 0)[..., None] *\
                        X_MFT[None]).sum(axis=(0,1))
                R_FMM = self.M * (num_MMFT / den_FT[None, None]).mean(axis=-1).transpose(-1, 0, 1)

        elif self.alpha_case == 'Estimated':
            self.alpha = self.compute_alpha(X_MFT)
            self.p = np.abs(self.alpha - 1) / 2 + 1
            # print("Estimated alpha={}".format(self.alpha))
            R_FMM = self.compute_Ralpha_cov(X_MFT)

        # THIS IS MUSIC
        # # compute steered response
        # self.Pssl = np.zeros((self.num_freq, self.grid.n_points))
        # C_hat = self._compute_correlation_matricesvec(X)
        # subspace decomposition
        Es, En, ws, wn = self._subspace_decomposition(R_FMM[None, ...])
        # compute spatial spectrum
        identity = np.zeros((self.num_freq, self.M, self.M))
        identity[:, list(np.arange(self.M)), list(np.arange(self.M))] = 1
        cross = identity - np.matmul(Es, np.moveaxis(np.conjugate(Es), -1, -2))
        self.Pssl = self._compute_spatial_spectrumvec(cross)
        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.Pssl, axis=1) / self.num_freq))

    def _apply_frequency_normalization(self):
        """
        Normalize the MUSIC pseudo-spectrum per frequency bin
        """
        self.Pssl = self.Pssl / np.max(self.Pssl, axis=0, keepdims=True)

    def plot_individual_spectrum(self):
        """
        Plot the steered response for each frequency.
        """

        # check if matplotlib imported
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        # only for 2D
        if self.grid.dim == 3:
            pass
        else:
            import warnings

            warnings.warn("Only for 2D.")
            return

        # plot
        for k in range(self.num_freq):

            freq = float(self.freq_bins[k]) / self.nfft * self.fs
            azimuth = self.grid.azimuth * 180 / np.pi

            plt.plot(azimuth, self.Pssl[k, 0 : len(azimuth)])

            plt.ylabel("Magnitude")
            plt.xlabel("Azimuth [degrees]")
            plt.xlim(min(azimuth), max(azimuth))
            plt.title("Steering Response Spectrum - " + str(freq) + " Hz")
            plt.grid(True)

    def _compute_spatial_spectrumvec(self, cross):
        mod_vec = np.transpose(
            np.array(self.mode_vec[self.freq_bins, :, :]), axes=[2, 0, 1]
        )
        # timeframe, frequ, no idea
        denom = np.matmul(
            np.conjugate(mod_vec[..., None, :]), np.matmul(cross, mod_vec[..., None])
        )
        return 1.0 / np.abs(denom[..., 0, 0])

    def _compute_spatial_spectrum(self, cross, k):

        P = np.zeros(self.grid.n_points)

        for n in range(self.grid.n_points):
            Dc = np.array(self.mode_vec[k, :, n], ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k, :, n], ndmin=2))
            denom = np.dot(np.dot(Dc_H, cross), Dc)
            P[n] = 1 / np.abs(denom)

        return P

    # non-vectorized version
    def _compute_correlation_matrices(self, X):
        C_hat = np.zeros([self.num_freq, self.M, self.M], dtype=complex)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            for s in range(self.num_snap):
                C_hat[i, :, :] = C_hat[i, :, :] + np.outer(
                    X[:, k, s], np.conjugate(X[:, k, s])
                )
        return C_hat / self.num_snap

    # vectorized version
    def _compute_correlation_matricesvec(self, X):
        # change X such that time frames, frequency microphones is the result
        X = np.transpose(X, axes=[2, 1, 0])
        # select frequency bins
        X = X[..., list(self.freq_bins), :]
        # Compute PSD and average over time frame
        C_hat = np.matmul(X[..., None], np.conjugate(X[..., None, :]))
        # Average over time-frames
        C_hat = np.mean(C_hat, axis=0)
        return C_hat

    # vectorized versino
    def _subspace_decomposition(self, R):
        # eigenvalue decomposition!
        # This method is specialized for Hermitian symmetric matrices,
        # which is the case since R is a covariance matrix
        w, v = np.linalg.eigh(R)

        # This method (numpy.linalg.eigh) returns the eigenvalues (and
        # eigenvectors) in ascending order, so there is no need to sort Signal
        # comprises the leading eigenvalues Noise takes the rest

        Es = v[..., -self.num_src :]
        ws = w[..., -self.num_src :]
        En = v[..., : -self.num_src]
        wn = w[..., : -self.num_src]

        return (Es, En, ws, wn)