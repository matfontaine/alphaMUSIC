from re import A
import numpy as np

def farfield_sphere(R_3M, n_fft=1024, az_acc=5, el_acc=20, sr=16000,
                    az_val=[-150, 150], el_val=[0, 90], do_normalize=True,
                    coeff_bw=False):
    az_val_min, az_val_max = (np.deg2rad(az_val[0]), np.deg2rad(az_val[1]))
    el_val_min, el_val_max = (np.deg2rad(el_val[0]), np.deg2rad(el_val[1]))
    az_val = np.linspace(az_val_min, az_val_max,
                         num=int((az_val_max - az_val_min)/np.deg2rad(az_acc)),
                         endpoint=False)
    el_val = np.linspace(el_val_min, el_val_max,
                         num=int((el_val_max - el_val_min)/np.deg2rad(el_acc)),
                         endpoint=True)

    c = 343.  # speed of sound in the air

    # frequencies to consider
    freq_F = np.fft.rfftfreq(n_fft, 1. / sr).astype(np.float32)

    Az_val, El_val = np.meshgrid(az_val, el_val)
    sph_D3 = np.stack((np.cos(Az_val) * np.sin(El_val),
                       np.sin(Az_val) * np.sin(El_val),
                       np.cos(El_val)), axis=2)
    sph_D3 = np.reshape(sph_D3, (len(az_val) * len(el_val), 3), order='F')
    r_DM = np.einsum('dp, pm -> dm', sph_D3, R_3M, optimize=True)

    TDOA_DFM = np.einsum('dm, f -> dfm', -2j * np.pi * r_DM, freq_F / c)
    steeringVector_DFM = np.exp(TDOA_DFM)

    if do_normalize:  # useful ? I don't know
        steeringVector_DFM /= np.linalg.norm(steeringVector_DFM, axis=-1,
                                             keepdims=True)

    if coeff_bw:  # useful ? I don't know
        coeff = np.load("steeringVector/coeff_bandwidth.npz")['coeff']

        # Importance bandwidth function
        Iw_F = coeff[0] * freq_F ** 4 +\
               coeff[1] * freq_F ** 3 +\
               coeff[2] * freq_F ** 2 +\
               coeff[3] * freq_F +\
               coeff[4]
        steeringVector_DFM *= Iw_F[None, :, None]
    return steeringVector_DFM, len(az_val), len(el_val), az_val, el_val

def farfield_plane(R_3M, n_fft=1024, az_acc=5, el_val=0, sr=16000,
                   do_normalize=False, az_val=[-150, 150], coeff_bw=False):

    az_val_min, az_val_max = (np.deg2rad(az_val[0]), np.deg2rad(az_val[1]))
    az_val = np.linspace(az_val_min, az_val_max,
                         num=int((az_val_max - az_val_min)/np.deg2rad(az_acc)),
                         endpoint=False)
    el_val = np.deg2rad(el_val)
    c = 343.  # speed of sound in the air

    # frequencies to consider
    freq_F = np.fft.rfftfreq(n_fft, 1. / sr).astype(np.float32)

    if el_val != 0:
        sph_D3 = np.stack((np.cos(az_val) * np.sin(el_val),
                           np.sin(az_val) * np.sin(el_val),
                           np.ones(len(az_val)) * np.cos(el_val)), axis=1)
    else:
        sph_D3 = np.stack((np.cos(az_val),
                           np.sin(az_val),
                           np.zeros(len(az_val))), axis=1)
    r_DM = np.einsum('dp, pm -> dm', sph_D3, R_3M, optimize=True)

    TDOA_DFM = np.einsum('dm, f -> dfm', 2j * np.pi * r_DM, freq_F / c)
    steeringVector_DFM = np.exp(TDOA_DFM)
    # import ipdb; ipdb.set_trace()
    if do_normalize:  # useful ? I don't know
        steeringVector_DFM /= np.linalg.norm(steeringVector_DFM, axis=-1,
                                             keepdims=True)
    if coeff_bw: # useful ? I don't know
        coeff = np.load("steeringVector/coeff_bandwidth.npz")['coeff']

        # Importance bandwidth function
        Iw_F = coeff[0] * freq_F ** 4 +\
               coeff[1] * freq_F ** 3 +\
               coeff[2] * freq_F ** 2 +\
               coeff[3] * freq_F +\
               coeff[4]
        steeringVector_DFM *= Iw_F[None, :, None]
    return steeringVector_DFM, len(az_val), az_val

def steering_vector_farfield(R_3M, az_grid, el_grid, n_fft=1024, sr=16000, c=343.,
                            do_normalize=False, coeff_bw=False):
 
    # az_val = np.linspace(az_val_min, az_val_max,
    #                      num=int((az_val_max - az_val_min)/np.deg2rad(az_acc)),
    #                      endpoint=False)
    # el_val = np.linspace(el_val_min, el_val_max,
    #                      num=int((el_val_max - el_val_min)/np.deg2rad(el_acc)),
    #                      endpoint=True)

    c = 343.  # speed of sound in the air
    # frequencies to consider
    freq_F = np.fft.rfftfreq(n_fft, 1. / sr).astype(np.float32)

    assert np.max(np.abs(az_grid)) < 6.29
    assert np.max(np.abs(el_grid)) < 3.15

    # az-el grid
    if len(el_grid) < 2:
        # 1D localization - PLANE
        doa_grid = az_grid
        sph_D3 = np.stack((np.cos(doa_grid),
                           np.sin(doa_grid),
                           np.zeros(len(doa_grid))), axis=1)
    
    else:
        # 2D localization - SPHERE
        doa_grid = np.stack(np.meshgrid(az_grid, el_grid)) # 2 x Az x El
        az_val = doa_grid[0,...]
        el_val = doa_grid[1,...]
        sph_D3 = np.stack((np.cos(az_val) * np.sin(el_val),
                           np.sin(az_val) * np.sin(el_val),
                           np.ones(len(az_val)) * np.cos(el_val)), axis=1)

    r_DM = np.einsum('dp, pm -> dm', sph_D3, R_3M, optimize=True)

    TDOA_DFM = np.einsum('dm, f -> dfm', 2j * np.pi * r_DM, freq_F / c)
    steeringVector_DFM = np.exp(TDOA_DFM)
    # import ipdb; ipdb.set_trace()
    if do_normalize:  # useful ? I don't know
        steeringVector_DFM /= np.linalg.norm(steeringVector_DFM, axis=-1,
                                             keepdims=True)
    if coeff_bw: # useful ? I don't know
        coeff = np.load("steeringVector/coeff_bandwidth.npz")['coeff']

        # Importance bandwidth function
        Iw_F = coeff[0] * freq_F ** 4 +\
               coeff[1] * freq_F ** 3 +\
               coeff[2] * freq_F ** 2 +\
               coeff[3] * freq_F +\
               coeff[4]
        steeringVector_DFM *= Iw_F[None, :, None]
    return steeringVector_DFM, doa_grid
