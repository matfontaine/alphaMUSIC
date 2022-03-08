import numpy as np
from tqdm import tqdm

def normalize(x):
    return x/np.max(np.abs(x))


def normalize_and_zero_mean(x):
    return normalize(x - np.mean(x))

def diffuse_noise(R, L, Fs, c=343, N=256, mode='sphere'):
    '''
    Generating sensor signals for a 1D sensor array in a spherically
    isotropic noise field [1,2]
    Python implementation of
    https: // github.com/ehabets/INF-Generator/blob/master/sinf_3D.m
    Input
        R: sensor positions
        L: desired data length
        Fs: sample frequency
        c: sound velocity in m/s
        N_phi: number of cylindrical angles
    Output:
        z : output signal
    '''

    D, M = R.shape  # Number of sensors
    assert D in [2, 3]
    nfft = int(2 ** np.ceil(np.log2(L))) # Number of frequency bins

    X = np.zeros([M, nfft//2+1], dtype=np.complex64)

    w = 2*np.pi*Fs*np.arange(nfft//2+1)/nfft

    # Generate N points that are near-uniformly distributed over S ^ 2
    theta = np.zeros(N)
    h = np.linspace(-1, 1, N)
    phi = np.arccos(h)
    for k in range(N):
        if k == 0 or k == N-1:
            theta[k] = 0
        else:
            theta[k] = (theta[k-1] + 3.6/np.sqrt(N*(1-h[k]**2))) % (2*np.pi)

    ## Calculate relative positions
    R_rel = np.zeros([3, M])
    for m in range(M):
        R_rel[: , m] = R[: , m] - R[: , 0]

    ## Calculate sensor signals in the frequency domain
    for k in tqdm(range(N)):
        X_prime = np.random.randn(nfft//2+1) \
                  + 1j*np.random.randn(nfft//2+1)
        X[0, :] = X[0, :] + X_prime
        for m in range(1, M):
            v = np.array([
                    np.sin(phi[k])*np.cos(theta[k]),
                    np.sin(phi[k])*np.sin(theta[k]),
                    np.cos(phi[k])])
            Delta = v @ R_rel[:, m]
            X[m, :] = X[m, :] + X_prime*np.exp(-1j * Delta * w/c)
    X = X/np.sqrt(N)

    ## Transform to time domain
    X_out = np.zeros([M, nfft], dtype=np.complex64)
    X_out[:, 0] = np.sqrt(nfft)*np.real(X[:, 0])
    X_out[:, 1:nfft//2] = np.sqrt(nfft//2)*X[:, 1:nfft//2]
    X_out[:, nfft//2] = np.sqrt(nfft)*np.real(X[:, nfft//2])
    X_out[:, nfft//2+1:] = np.sqrt(nfft//2)*np.flip(np.conj(X[:, 1:nfft//2]), axis=1)

    z = np.real(np.fft.ifft(X, nfft))

    ## Truncate output signals
    return z[:,:L]
