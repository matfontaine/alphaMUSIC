from tkinter import E
import numpy as np
from alphaMusic.utility import steering_vector_farfield, farfield_plane
from skimage.feature.peak import peak_local_max

import scipy.signal as si

import time
import glob as glob
import os as os
import math
import json
from alphaMusic.detect_peak import detect_peaks

import pyroomacoustics as pra
from pyroomacoustics.doa import DOA


import matplotlib.pyplot as plt

def convert_to_NumpyArray(self, data):
    if self.xp == np:
        return data
    else:
        return self.xp.asnumpy(data)


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

        self.p = 1.5
        # self.p2 = 1.5
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
                    coeff_cov_F = (X_MFT[m] * X_MFT[l].conj() *
                                  (np.abs(X_MFT[l]) ** (self.p - 2.))).sum(axis=-1)
                    coeff_cov_F /= (np.abs(X_MFT[l]) ** (self.p) + 1e-10).sum(axis=-1)

                    R_FMM[:, m, l] = coeff_cov_F * cov_MF[l] *\
                                    2 ** (self.alpha/2.) *\
                                    R_FMM[:, l, l] ** (-(self.alpha-2.)/2.)
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
            # print("Estimated alpha={}".format(self.alpha))
            R_FMM = self.compute_Ralpha_cov(X_MFT)


        # # compute steered response
        self.Pssl = np.zeros((self.num_freq, self.grid.n_points))
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
        return 1.0 / abs(denom[..., 0, 0])

    def _compute_spatial_spectrum(self, cross, k):

        P = np.zeros(self.grid.n_points)

        for n in range(self.grid.n_points):
            Dc = np.array(self.mode_vec[k, :, n], ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k, :, n], ndmin=2))
            denom = np.dot(np.dot(Dc_H, cross), Dc)
            P[n] = 1 / abs(denom)

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


class AlphaMUSIC():

    def __init__(self, L, fs, nfft, 
                    c=343.0, num_src=1, mode='far', 
                    P_prime=30, alpha=1.8, seed=1,
                    distance=None, azimuth=None, elevation=None, xp=np):

        super(AlphaMUSIC, self).__init__()

        self.xp = xp

        self.mic_pos = L
        self.fs = fs
        self.nfft = nfft

        self.N = num_src
        self.num_src = num_src
        
        self.c = c
        self.ac_model = mode

        self.M = L.shape[1]
        self.D = L.shape[0]

        self.P_prime = P_prime
        self.alpha = alpha
        self.seed = seed
        self.rand_s = self.xp.random.RandomState(self.seed)

        if distance is None:
            self.r = [1]
        else:
            self.r = distance

        if azimuth is None:
            self.az_grid = xp.rad2deg(xp.linspace(0, 360., 360, endpoint=False))
        else:
            self.az_grid = azimuth
        
        if elevation is None:
            self.el_grid = [0]
        else:
            self.el_grid = elevation
        
    def load_spectrogram(self, X_FTM, minF, maxF, nfft, fs):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.nfft = nfft
        self.fs = fs
        self.maxF = int(maxF * self.nfft / self.fs)
        self.minF = int(minF * self.nfft / self.fs)
        self.X_MFT = self.xp.asarray(X_FTM[self.minF:self.maxF], dtype=self.xp.complex).transpose(-1, 0, 1)
        self.M, self.F, self.T = self.X_MFT.shape
        if self.alpha > 2.0:
            self.compute_alpha()

        if self.ac_model == 'far':
            self.steeringVector_PFM, self.doa_grid = steering_vector_farfield(
                                                        R_3M=self.mic_pos, n_fft=self.nfft,
                                                        az_grid=self.az_grid, el_grid = self.el_grid,
                                                        sr=self.fs)
            self.steeringVector_PFM = self.xp.array(self.steeringVector_PFM)
            self.power_steeringVector_PF = (self.steeringVector_PFM.conj() * self.steeringVector_PFM).sum(axis=2)
            self.P = self.steeringVector_PFM.shape[0]
    
    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)


    def compute_alpha(self):
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

        K = (self.maxF - self.minF) * self.T
        # K = self.n_freq * self.n_time
        K1, K2, _ = factor_int(K)
        rnd_X_FTM = shuffle_along_axis(self.convert_to_NumpyArray(self.X_MFT.transpose(1, 2, 0)),
                                       axis=0)
        rnd_X_FTxM = self.xp.array(shuffle_along_axis(rnd_X_FTM,
                                   axis=1)).reshape(-1, rnd_X_FTM.shape[-1])
        Y_K2M = self.xp.zeros((K2, self.M)).astype(self.xp.complex64)
        for k2 in range(K2):
            Y_K2M[k2] = rnd_X_FTxM[k2*K1:  K1 + k2*K1, :].sum(axis=0)
        logXnorm_FT = self.xp.log(self.xp.linalg.norm(rnd_X_FTxM, axis=-1))
        logYnorm_FT = self.xp.log(self.xp.linalg.norm(Y_K2M, axis=-1))
        self.alpha = (1 / self.xp.log(K1) *
                      (1/K2 * logYnorm_FT.sum() -
                      1/K * logXnorm_FT.sum())) ** (-1)
        print("alpha value:{}".format(self.alpha))

    def compute_Ralpha_levyExp(self):
        """Estimate the elliptical covariance matrix of the K x N data S
        * P: number of thetas"""

        def levy_exponent(indices,weights,S,theta):
            """computes the levy exponents I(\sum_l e_indices(l) * weights(l), theta)
            * indices: list
            * weights: list same length as indices
            * X: M x T
            * theta: P"""
            projection = 0
            for (l,w) in zip(indices,weights):
                projection += X[l, :].conj()*w # dimension T


            tmp = self.xp.real(self.xp.conj(theta[:,None]) * projection[None, :]) #tmp=real( theta.H * proj) : PxFxT
            tmp = self.xp.exp(1j/2**(1./self.alpha) * tmp) #tmp=exp(i/2**(1/alpha) * tmp)

            phi = self.xp.mean(tmp, axis=-1) # P
            I = -2.*self.xp.log(self.xp.abs(phi)) # P
            return I

        #draw random thetas

        # thetas = 1j * self.xp.array([np.pi * k /25. for k in range(30, 35)])
        thetas = self.rand_s.randn(self.P_prime) + 1j * self.rand_s.randn(self.P_prime)
        # thetas = 1j * self.rand_s.randn(self.P_prime)
        # thetas = np.concatenate((thetas, self.rand_s.randn(self.P_prime)), axis=0)
        # angle = self.xp.linspace(0., np.pi/2., num=self.P_prime)
        # thetas = self.xp.abs(self.rand_s.randn(self.P_prime)) * (self.xp.cos(angle) + 1j * self.xp.sin(angle))
        # thetas/=self.xp.abs(thetas)
        self.R_FMM = self.xp.zeros((self.F,self.M,self.M),dtype=self.xp.complex64)
        for f in range(self.F):
            X = self.X_MFT[:, f]
            for m in range(self.M):
                # thetas = self.steeringVector_PFM[:, f, m]
                self.R_FMM[f, m, m] = self.xp.mean((2./self.xp.abs(thetas)**2)*levy_exponent((m,),(1.,),X,thetas)**(2./self.alpha))
            for m in range(self.M):
                for l in range(m):
                    # thetas = self.steeringVector_PFM[:, f, m] * self.steeringVector_PFM[:, f, l]
                    self.R_FMM[f, m, l] = (-0.5j-0.5)*(self.R_FMM[f, m, m]+self.R_FMM[f, l, l]) + \
                              self.xp.mean((1./self.xp.abs(thetas)**2)*(levy_exponent((m,l),(1.,1.),X,thetas)**(2./self.alpha)
                                                                    +1j*levy_exponent((m,l),(1.,-1j),X,thetas)**(2./self.alpha) ) )
                    self.R_FMM[f, l, m]=self.xp.conj(self.R_FMM[f, m, l])

    def compute_Ralpha_cov(self):
        """Estimate the elliptical covariance matrix of the K x N data S
        """
        self.p = 1.5
        # self.p2 = 1.5
        self.R_FMM = self.xp.zeros((self.F,self.M,self.M),dtype=self.xp.complex64)
        gamma = 0.5772156 # Euler constant
        Cste = -gamma * (self.alpha **(-1) - 1.) - self.xp.log(2.0)
        # X = self.X_MFT[:, f]

        mean_MF = self.xp.mean(self.xp.log(self.xp.abs(self.X_MFT) + 1e-10), axis=-1)
        cov_MF = self.xp.exp((mean_MF+Cste) * self.alpha ** (-1))
        diag_MF = 2. * (cov_MF) ** (2./self.alpha)
        for m in range(self.M):
            self.R_FMM[:, m, m] = diag_MF[m]
        for m in range(self.M):
            for l in range(m):
                if l != m:
                    coeff_cov_F = (self.X_MFT[m] * self.X_MFT[l].conj() *
                                  (self.xp.abs(self.X_MFT[l]) ** (self.p - 2.))).sum(axis=-1)
                    coeff_cov_F /= (self.xp.abs(self.X_MFT[l]) ** (self.p) + 1e-10).sum(axis=-1)
                    # coeff_cov2 = (X[m] * X[l].conj() * (self.xp.abs(X[l]) ** (self.p2 - 2.))).sum()
                    # coeff_cov2 /= (self.xp.abs(X[l]) ** (self.p2) + 1e-10).sum()
                    # coeff_cov += coeff_cov2 / 2.

                    self.R_FMM[:, m, l] = coeff_cov_F * cov_MF[l] *\
                                          2 ** (self.alpha/2.) *\
                                          self.R_FMM[:, l, l] ** (-(self.alpha-2.)/2.)
                    self.R_FMM[:, l, m] = self.xp.conj(self.R_FMM[:, m, l])


    def MUSIC(self):
        """ MUSIC algorithm for sound source localization
        Returns:
        -----------
            localization_result: np.array [ n_direction ]
        """
        eig_val_FM, eig_vec_FMM = self.xp.linalg.eigh(self.R_FMM)
        eig_vec_FMM = eig_vec_FMM[:, :, ::-1]
        noise_steeringVector_FxNnxM = eig_vec_FMM[:, :, self.N:].transpose(0, 2, 1)
        localization_result = np.zeros([self.P])
        for p in range(self.P):
            tmp_FxNn = (noise_steeringVector_FxNnxM.conj() @ self.steeringVector_PFM[p, self.minF:self.maxF, :, None])[:, :, 0]
            localization_result[p] = (self.power_steeringVector_PF[p, self.minF:self.maxF] / (tmp_FxNn.conj() * tmp_FxNn).sum(axis=1)).mean().real
        self.power_spec_P = localization_result.real

        # plt.plot(localization_result)
        # plt.show()

    def find_n_max_in_heatmap(self, array2d, n_max):
        array2d = array2d / np.max(np.abs(array2d))
        peaks = peak_local_max(array2d, num_peaks=n_max, min_distance=0)
        return self.convert_to_NumpyArray(peaks.T)


    def apply_norm_music(self, P):
        return P / np.max(self.P, axis=0, keepdims=True)


    def find_peaks(self, k=1):
        # make circular
        val_ext = np.append(self.power_spec_P, self.power_spec_P[:10])

        # run peak finding
        indexes = detect_peaks(val_ext, show=False) % self.P
        candidates = np.unique(indexes)  # get rid of duplicates, if any

        # Select k largest
        peaks = self.power_spec_P[candidates]
        max_idx = np.argsort(peaks)[-k:]

        # return the indices of peaks found
        return candidates[max_idx]

    def locate_sources(self, X, freq_range=[500., 4000.]):
        
        self.load_spectrogram(X, freq_range[0], freq_range[1], self.nfft, self.fs)

        self.start_time = time.time()

        if self.alpha < 2.0 and self.alpha > 0.0 and self.alpha != 2.0:
            self.compute_Ralpha_cov()
            name = "alpha={}".format(self.alpha)
        elif self.alpha == 2.0:
            name = "Gaussian"
            self.alpha = name
            self.R_FMM = (self.X_MFT[:, None] * self.X_MFT[None].conj()).mean(axis=-1).transpose(-1, 0, 1)
        elif self.alpha == 0.0:
            name = "Tyler"
            self.alpha = name
            self.R_FMM = self.xp.tile(self.xp.eye(self.M), [self.F, 1, 1]).astype(self.xp.complex)

            for i in range(20):
                invR_FMM = self.xp.linalg.inv(self.R_FMM)
                num_MMFT = self.X_MFT[:, None] * self.X_MFT[None].conj()
                den_FT =  (self.X_MFT[:, None].conj() *\
                        invR_FMM.transpose(-1, -2, 0)[..., None] *\
                        self.X_MFT[None]).sum(axis=(0,1))
                self.R_FMM = self.M * (num_MMFT / den_FT[None, None]).mean(axis=-1).transpose(-1, 0, 1)
        else:
            print("Please specify  0 < alpha <= 2")
            raise ValueError

        self.MUSIC()
        self.end_time = time.time()
        if self.ac_model == 'near':
            self.power_spec_P = self.xp.reshape(self.power_spec_P, (self.n_x, self.n_y))
        elif self.ac_model == 'far':
            self.est_pos_N = self.doa_grid[self.find_peaks(k=self.N)]
            
        self.azimuth_recon = self.est_pos_N 
        return self.est_pos_N


class AlphaMUSIC_old():
    def __init__(self, n_source=2,
                 seed=1, alpha=1.8, ac_model='far', x_acc=5, y_acc=10,
                 mic_pos=None, source_pos=None, P_prime=30, xp=np):

        super(AlphaMUSIC_old, self).__init__()
        self.N = n_source
        self.P_prime = P_prime
        self.mic_pos = mic_pos
        self.xp = xp
        # self.pos_N2 = self.convert_to_NumpyArray(source_pos)
        self.pos_N = self.convert_to_NumpyArray(source_pos)
        self.alpha = alpha
        self.seed = seed
        self.rand_s = self.xp.random.RandomState(self.seed)
        self.ac_model = ac_model
        if ac_model == 'far':
            self.az_acc, self.el_acc = x_acc, y_acc
            self.az_val, self.el_val = [0, 360], 0.0

    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def load_spectrogram(self, X_FTM, minF, maxF, nfft, fs):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.nfft = nfft
        self.fs = fs
        self.maxF = int(maxF * self.nfft / self.fs)
        self.minF = int(minF * self.nfft / self.fs)
        self.X_MFT = self.xp.asarray(X_FTM[self.minF:self.maxF], dtype=self.xp.complex).transpose(-1, 0, 1)
        self.M, self.F, self.T = self.X_MFT.shape
        if self.alpha > 2.0:
            self.compute_alpha()

        if self.ac_model == 'far':
            # self.steeringVector_PFM,self.n_az, self.n_el, self.az_val, self.el_val = farfield_sphere(R_3M=self.mic_pos, n_fft=self.nfft,
            #                                                                                          az_acc=self.az_acc, el_acc=self.el_acc,
            #                                                                                          az_val=self.az_val, el_val=self.el_val,
            #                                                                                          sr=self.fs)

            self.steeringVector_PFM,self.n_az, self.az_val = farfield_plane(R_3M=self.mic_pos, n_fft=self.nfft,
                                                                            az_acc=self.az_acc, az_val=self.az_val,
                                                                            el_val = self.el_val,
                                                                            sr=self.fs)
            self.steeringVector_PFM = self.xp.array(self.steeringVector_PFM)
            self.power_steeringVector_PF = (self.steeringVector_PFM.conj() * self.steeringVector_PFM).sum(axis=2)
            self.P = self.steeringVector_PFM.shape[0]

    def compute_alpha(self):
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

        K = (self.maxF - self.minF) * self.T
        # K = self.n_freq * self.n_time
        K1, K2, _ = factor_int(K)
        rnd_X_FTM = shuffle_along_axis(self.convert_to_NumpyArray(self.X_MFT.transpose(1, 2, 0)),
                                       axis=0)
        rnd_X_FTxM = self.xp.array(shuffle_along_axis(rnd_X_FTM,
                                   axis=1)).reshape(-1, rnd_X_FTM.shape[-1])
        Y_K2M = self.xp.zeros((K2, self.M)).astype(self.xp.complex64)
        for k2 in range(K2):
            Y_K2M[k2] = rnd_X_FTxM[k2*K1:  K1 + k2*K1, :].sum(axis=0)
        logXnorm_FT = self.xp.log(self.xp.linalg.norm(rnd_X_FTxM, axis=-1))
        logYnorm_FT = self.xp.log(self.xp.linalg.norm(Y_K2M, axis=-1))
        self.alpha = (1 / self.xp.log(K1) *
                      (1/K2 * logYnorm_FT.sum() -
                      1/K * logXnorm_FT.sum())) ** (-1)
        print("alpha value:{}".format(self.alpha))

    def compute_Ralpha_levyExp(self):
        """Estimate the elliptical covariance matrix of the K x N data S
        * P: number of thetas"""

        def levy_exponent(indices,weights,S,theta):
            """computes the levy exponents I(\sum_l e_indices(l) * weights(l), theta)
            * indices: list
            * weights: list same length as indices
            * X: M x T
            * theta: P"""
            projection = 0
            for (l,w) in zip(indices,weights):
                projection += X[l, :].conj()*w # dimension T


            tmp = self.xp.real(self.xp.conj(theta[:,None]) * projection[None, :]) #tmp=real( theta.H * proj) : PxFxT
            tmp = self.xp.exp(1j/2**(1./self.alpha) * tmp) #tmp=exp(i/2**(1/alpha) * tmp)

            phi = self.xp.mean(tmp, axis=-1) # P
            I = -2.*self.xp.log(self.xp.abs(phi)) # P
            return I

        #draw random thetas

        # thetas = 1j * self.xp.array([np.pi * k /25. for k in range(30, 35)])
        thetas = self.rand_s.randn(self.P_prime) + 1j * self.rand_s.randn(self.P_prime)
        # thetas = 1j * self.rand_s.randn(self.P_prime)
        # thetas = np.concatenate((thetas, self.rand_s.randn(self.P_prime)), axis=0)
        # angle = self.xp.linspace(0., np.pi/2., num=self.P_prime)
        # thetas = self.xp.abs(self.rand_s.randn(self.P_prime)) * (self.xp.cos(angle) + 1j * self.xp.sin(angle))
        # thetas/=self.xp.abs(thetas)
        self.R_FMM = self.xp.zeros((self.F,self.M,self.M),dtype=self.xp.complex64)
        for f in range(self.F):
            X = self.X_MFT[:, f]
            for m in range(self.M):
                # thetas = self.steeringVector_PFM[:, f, m]
                self.R_FMM[f, m, m] = self.xp.mean((2./self.xp.abs(thetas)**2)*levy_exponent((m,),(1.,),X,thetas)**(2./self.alpha))
            for m in range(self.M):
                for l in range(m):
                    # thetas = self.steeringVector_PFM[:, f, m] * self.steeringVector_PFM[:, f, l]
                    self.R_FMM[f, m, l] = (-0.5j-0.5)*(self.R_FMM[f, m, m]+self.R_FMM[f, l, l]) + \
                              self.xp.mean((1./self.xp.abs(thetas)**2)*(levy_exponent((m,l),(1.,1.),X,thetas)**(2./self.alpha)
                                                                    +1j*levy_exponent((m,l),(1.,-1j),X,thetas)**(2./self.alpha) ) )
                    self.R_FMM[f, l, m]=self.xp.conj(self.R_FMM[f, m, l])

    def compute_Ralpha_cov(self):
        """Estimate the elliptical covariance matrix of the K x N data S
        """
        self.p = 1.5
        # self.p2 = 1.5
        self.R_FMM = self.xp.zeros((self.F,self.M,self.M),dtype=self.xp.complex64)
        gamma = 0.5772156 # Euler constant
        Cste = -gamma * (self.alpha **(-1) - 1.) - self.xp.log(2.0)
        # X = self.X_MFT[:, f]

        mean_MF = self.xp.mean(self.xp.log(self.xp.abs(self.X_MFT) + 1e-10), axis=-1)
        cov_MF = self.xp.exp((mean_MF+Cste) * self.alpha ** (-1))
        diag_MF = 2. * (cov_MF) ** (2./self.alpha)
        for m in range(self.M):
            self.R_FMM[:, m, m] = diag_MF[m]
        for m in range(self.M):
            for l in range(m):
                if l != m:
                    coeff_cov_F = (self.X_MFT[m] * self.X_MFT[l].conj() *
                                  (self.xp.abs(self.X_MFT[l]) ** (self.p - 2.))).sum(axis=-1)
                    coeff_cov_F /= (self.xp.abs(self.X_MFT[l]) ** (self.p) + 1e-10).sum(axis=-1)
                    # coeff_cov2 = (X[m] * X[l].conj() * (self.xp.abs(X[l]) ** (self.p2 - 2.))).sum()
                    # coeff_cov2 /= (self.xp.abs(X[l]) ** (self.p2) + 1e-10).sum()
                    # coeff_cov += coeff_cov2 / 2.

                    self.R_FMM[:, m, l] = coeff_cov_F * cov_MF[l] *\
                                          2 ** (self.alpha/2.) *\
                                          self.R_FMM[:, l, l] ** (-(self.alpha-2.)/2.)
                    self.R_FMM[:, l, m] = self.xp.conj(self.R_FMM[:, m, l])

    def MUSIC(self):
        """ MUSIC algorithm for sound source localization
        Returns:
        -----------
            localization_result: np.array [ n_direction ]
        """
        eig_val_FM, eig_vec_FMM = self.xp.linalg.eigh(self.R_FMM)
        eig_vec_FMM = eig_vec_FMM[:, :, ::-1]
        noise_steeringVector_FxNnxM = eig_vec_FMM[:, :, self.N:].transpose(0, 2, 1)
        localization_result = np.zeros([self.P])
        for p in range(self.P):
            tmp_FxNn = (noise_steeringVector_FxNnxM.conj() @ self.steeringVector_PFM[p, self.minF:self.maxF, :, None])[:, :, 0]
            localization_result[p] = (self.power_steeringVector_PF[p, self.minF:self.maxF] / (tmp_FxNn.conj() * tmp_FxNn).sum(axis=1)).mean().real
        self.power_spec_P = localization_result.real

    def find_n_max_in_heatmap(self, array2d, n_max):
        array2d = array2d / np.max(np.abs(array2d))
        peaks = peak_local_max(array2d, num_peaks=n_max, min_distance=0)
        return self.convert_to_NumpyArray(peaks.T)

    def find_peaks(self, k=1):
        # make circular
        val_ext = np.append(self.power_spec_P, self.power_spec_P[:10])

        # run peak finding
        indexes = detect_peaks(val_ext, show=False) % self.P
        candidates = np.unique(indexes)  # get rid of duplicates, if any

        # Select k largest
        peaks = self.power_spec_P[candidates]
        max_idx = np.argsort(peaks)[-k:]

        # return the indices of peaks found
        return candidates[max_idx]

    def compute_RMS(self):
        self.est_pos_N = np.sort(self.est_pos_N)
        self.pos_N = np.sort(self.pos_N)
        results = np.zeros((self.N, self.N))
        for n1 in range(self.N):
            sq_error_val = []
            for n2 in range(self.N):
                values = [(self.est_pos_N[n1] - self.pos_N[n2]) ** 2,
                          (self.est_pos_N[n1] - self.pos_N[n2] - 2 * np.pi) ** 2,
                           (self.est_pos_N[n1] - self.pos_N[n2] + 2 * np.pi) ** 2]
                results[n1, n2] = min(values)

        scores = np.min(results, axis=1)
        self.RMS = np.sqrt((scores).mean())

    def localize(self, save_parameter=False, save_pos=False,
                 save_RMS=False, save_corr=False, SAVE_PATH=None):
        self.make_filename_suffix()

        self.start_time = time.time()

        if self.alpha < 2.0 and self.alpha > 0.0 and self.alpha != 2.0:
            self.compute_Ralpha_cov()
            name = "alpha={}".format(self.alpha)

        elif self.alpha == 2.0:
            name = "Gaussian"
            self.alpha = name
            self.R_FMM = (self.X_MFT[:, None] * self.X_MFT[None].conj()).mean(axis=-1).transpose(-1, 0, 1)

        elif self.alpha == 0.0:
            name = "Tyler"
            self.alpha = name
            self.R_FMM = self.xp.tile(self.xp.eye(self.M), [self.F, 1, 1]).astype(self.xp.complex)

            for i in range(20):
                invR_FMM = self.xp.linalg.inv(self.R_FMM)
                num_MMFT = self.X_MFT[:, None] * self.X_MFT[None].conj()
                den_FT =  (self.X_MFT[:, None].conj() *\
                           invR_FMM.transpose(-1, -2, 0)[..., None] *\
                           self.X_MFT[None]).sum(axis=(0,1))
                self.R_FMM = self.M * (num_MMFT / den_FT[None, None]).mean(axis=-1).transpose(-1, 0, 1)
        else:
            print("Please specify  0 < alpha <= 2")
            raise ValueError

        self.MUSIC()
        self.end_time = time.time()
        if self.ac_model == 'near':
            self.power_spec_P = self.xp.reshape(self.power_spec_P, (self.n_x, self.n_y))
        elif self.ac_model == 'far':
            # self.power_spec_P /= self.power_spec_P.max()
            # self.power_spec_P = self.xp.reshape(self.power_spec_P, (self.n_az, self.n_el), order='F')
            # self.shift = 10
            # self.prom = 1.0
            # localization_result_al = np.concatenate((self.power_spec_P[-self.shift:],
            #                                          self.power_spec_P),
            #                                          axis=0)
            #
            # # ax.scatter(x=[-30, 30], y=[max_loc, max_loc],
            # #            s=20, c='blue', label='estimated position')
            # est_pos, prop = si.find_peaks(self.power_spec_P,
            #                               prominence=self.prom)
            # # while (len(est_pos) < self.N):
            # #     i=0
            # #     est_pos, prop = si.find_peaks(self.power_spec_P,
            # #                                   prominence=self.prom + i)
            # #     i += 0.1
            # if len(est_pos) > self.N:
            #     if np.argmax(self.power_spec_P) in est_pos:
            #         idx = prop['prominences'].argsort()[-self.N:][::-1]
            #         print(idx)
            #         est_pos = [est_pos[idx[i]] for i in range(len(idx))]
            # elif len(est_pos) < self.N:
            #     if np.argmax(self.power_spec_P) in est_pos:
            #         idx = prop['prominences'].argsort()[-self.N:][::-1]
            #         print(idx)
            #         est_pos = [est_pos[idx[i]] for i in range(len(idx))]
            # # for i in range(len(est_pos)):
            # #     est_pos[i] = est_pos[i] - self.shift
            #
            # self.est_pos_N = [self.az_val[est_pos[i]] for i in range(len(est_pos))]
            self.est_pos_N = self.az_val[self.find_peaks(k=self.N)]
            # print ("real_pos:{}, est_pos:{}".format(self.pos_N, self.est_pos_N))

        # self.est_pos_N2 = self.find_n_max_in_heatmap(self.convert_to_NumpyArray(self.power_spec_P), self.convert_to_NumpyArray(self.N)).T # numpy
        # self.pos_N2 = self.pos_N2.T
        # self.est_pos_N2 = [[self.az_val[self.est_pos_N2[n,0]], self.el_val[self.est_pos_N2[n,1]]] for n in range(self.convert_to_NumpyArray(self.N))]
        # import matplotlib.pyplot as plt
        # plt.plot(self.az_val, self.power_spec_P)
        # plt.imshow(self.power_spec_P, extent=np.rad2deg([self.el_val[0], self.el_val[-1], self.az_val[0], self.az_val[-1]]), origin="lower")
        # plt.savefig("{}.png".format(name))
        # print ("real_pos:{}, est_pos:{}".format(self.pos_N2, self.est_pos_N2))
        # import ipdb; ipdb.set_trace()
        if save_pos or save_RMS or save_corr or save_parameter:
            results = {}
            if save_pos:
                if self.ac_model == 'near':
                    pos = {'x':self.est_pos_N2[:, 0],
                           'y':self.est_pos_N2[:, 1],
                           'true_x':self.pos_N2[:, 0],
                           'true_y':self.pos_N2[:, 1]}
                elif self.ac_model == 'far':
                    pos = {'est_az': self.est_pos_N.tolist(),
                           'true_az': self.pos_N.tolist()}
                results.update(pos)
            if save_RMS:
                self.compute_RMS()
                if self.ac_model == 'near':
                    RMS = {'x': self.RMS[0], 'y': self.RMS[1]}
                elif self.ac_model == 'far':
                    RMS = {'az_RMS': self.RMS}
                print('RMS:{}'.format(RMS))
                results.update(RMS)
            # if save_corr:
            #     self.compute_corr(self.est_pos_N2, self.source_pos_N2)
            #     if self.ac_model == 'near':
            #         corr = {'x': self.corr[0],
            #          'y': self.corr[1]}
            #     elif self.ac_model == 'far':
            #         corr = {'az': self.corr[0], 'el': self.corr[1]}
            #     results.update(corr)
            if save_parameter:
                param = {'alpha': [self.alpha],
                        # 'alpha': self.convert_to_NumpyArray(self.alpha).tolist(),
                         'true_pos':[self.convert_to_NumpyArray(self.pos_N)],
                         'time': [self.convert_to_NumpyArray(self.end_time - self.start_time)],
                         'power_spec':[self.convert_to_NumpyArray(self.power_spec_P).tolist()]
                }
                results.update(param)
            # json.dump(results, open(os.path.join(SAVE_PATH, "loc_res-{}.json".format(self.filename_suffix)), 'w' ))
        return self.est_pos_N
    #
    def make_filename_suffix(self):
        if hasattr(self, "file_id"):
            self.filename_suffix = "{}".format(self.file_id)