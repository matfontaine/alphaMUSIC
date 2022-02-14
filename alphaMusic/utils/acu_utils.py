import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from pathlib import Path

from collections import namedtuple
import matplotlib.pyplot as plt

import alphaMusic.utils.geo_utils as geo
import alphaMusic.utils.dsp_utils as dsp

# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern, center, n_mics')
SourceSetup = namedtuple('SourceSetup', 'doa, signal, path_to_signal, fs')

def get_echo_array(n_mics, mic_radius, mic_center=None):

	if mic_center is None:
		mic_center = np.zeros([3,1])

	R_3M = pra.circular_2D_array(mic_center[:2, 0], n_mics, 0, mic_radius)
	R_3M = np.concatenate((R_3M, np.ones((1, n_mics)) * mic_center[2, 0]), axis=0)

	return ArraySetup(arrayType='circular',
		orV = None,
		mic_pos = R_3M,
		mic_orV = None,
		mic_pattern = 'omni',
		center = mic_center,
		n_mics = n_mics
	) 

def get_linear_array(n_mics, spacing, mic_center=None):
	if mic_center is None:
		mic_center = np.zeros([3,1])

	R_3M = pra.linear_2D_array(mic_center[:2, 0], n_mics, 0, spacing)
	R_3M = np.concatenate((R_3M, np.ones((1, n_mics)) * mic_center[2, 0]), axis=0)

	return ArraySetup(arrayType='linear',
		orV = None,
		mic_pos = R_3M,
		mic_orV = None,
		mic_pattern = 'omni',
		center = mic_center,
		n_mics = n_mics
	) 


echo_array_setup = get_echo_array(5, 0.04)
linear_array_setup = get_linear_array(5, 0.04)

class AcousticScene:

	""" Acoustic scene class.
	It contains everything needed to simulate a moving sound source moving recorded
	with a microphone array in a reverberant room.
	It can also store the results from the DOA estimation.
	"""

	def __init__(self,  room_dim, RT60, SNR, array_setup, mic_center, 
						DOAs, source_signals, fs,
						path_to_noise):

		self.room_dim = room_dim				# Room size
		self.RT60 = RT60						# Reverberation time of the simulated room
		self.SNR = SNR							# Signal to (omnidirectional) Noise Ration to simulate
		self.array_setup = array_setup			# Named tuple with the characteristics of the array
		self.mic_center = mic_center		    # Position of the center of the array
		self.source_signals = source_signals  	# Source signal
		self.fs = fs							# Sampling frequency of the source signal and the simulations
		self.DOAs = DOAs                    	# 2D Direction of Arrival (list of tuple [(d1,az1,el1), (d2,az2,el2)])
		self.path_to_noise = path_to_noise
		assert len(self.DOAs) == len(self.source_signals)

		self.true_positions = []
		self.acoustic_params = []

		if self.path_to_noise is None:
			self.noise = None
		else:
			self.noise = sf.read(self.path_to_noise)[0].T

		# elif self.noise_environment == 'office': self.noise = sf.read(Path('.','data','office_noise.wav'))[0].T
		# elif self.noise_environment == 'cafet':  self.noise = sf.read(Path('.','data','cafet_noise.wav'))[0].T
		# elif self.noise_environment == 'living': self.noise = sf.read(Path('.','data','living_noise.wav'))[0].T

		# hard-coded vars
		self.l_filter = 40
		

	def common_step(self,):
		# set up room
		if self.RT60 == 0:
			e_absorption = 0.99
			max_order = 0
		else:
			e_absorption, max_order = pra.inverse_sabine(self.RT60, self.room_dim)
		
		# print(f'Simulating with, abs={e_absorption}, order={max_order}')

		# reverberant room
		room = pra.ShoeBox(
    			self.room_dim, fs=self.fs, 
				materials=pra.Material(e_absorption), max_order=max_order,
		)
		# anechoic room
		aroom = pra.ShoeBox(
    			self.room_dim, fs=self.fs, 
				materials=pra.Material(0.99), max_order=0,
		)

		eroom = pra.ShoeBox(
    			self.room_dim, fs=self.fs, 
				materials=pra.Material(e_absorption), max_order=1,
		)

		# add microphone array
		mic_pos = self.array_setup.mic_pos + self.mic_center
		room.add_microphone_array(mic_pos)
		aroom.add_microphone_array(mic_pos)
		eroom.add_microphone_array(mic_pos)
		return room, aroom, eroom


	def setup_room(self,):
		room, aroom, eroom = self.common_step()
		
		# add sources 
		n_srcs = len(self.DOAs)
		for j in range(n_srcs):
			geo_gt = {}

			dist = self.DOAs[j].distance
			az = self.DOAs[j].azimuth
			el = self.DOAs[j].elevation
			assert self.DOAs[j].deg

			src_sph = np.array([[dist, az, el]]).T
			src_cart_rel = geo.sph2cart(src_sph, deg=True)
			src_cart_abs = src_cart_rel + self.mic_center
			
			s = dsp.normalize_and_zero_mean(self.source_signals[j])
			room.add_source(src_cart_abs, signal=s)
			aroom.add_source(src_cart_abs, signal=s)
			eroom.add_source(src_cart_abs, signal=s)

			geo_gt['abs'] = src_cart_abs
			geo_gt['sph'] = src_sph

			self.true_positions.append(geo_gt)
		
		return room, aroom, eroom


	def simulate_diffuse_noise(self, signal):
		
		room, aroom = self.common_step()
		
		L, W, H = self.room_dim
		# add sources in the four corners
		diff_sources_pos = np.array([
			[  0.2,   0.2, H/2],
			[L-0.2,   0.2, H/2],
			[  0.2, W-0.2, H/2],
			[L-0.2, W-0.2, H/2],
		]).T
		
		for j in range(4):
			src_cart_abs = diff_sources_pos[:,j]
			room.add_source(src_cart_abs, signal=signal)
		
		room.simulate()
		
		return room.mic_array.signals


	def get_rir_matrix(self,):

		room, aroom = self.setup_room()
		room.simulate()
		
		I = len(room.rir)
		J = len(room.rir[0])
		L = max([len(room.rir[i][j]) for i in range(I) for j in range(J)])
		
		rirs = np.zeros([L, I, J])
		for i in range(I):
			for j in range(J):
				l = len(room.rir[i][j])
				rirs[:l,i,j] = room.rir[i][j]

		return rirs


	def simulate(self,):
		""" Get the array recording using pyroomacoustics to perform the acoustic simulations.
		"""

		# set up room
		room, aroom, eroom = self.setup_room()
		# simulate reverberation
		room.simulate()
		aroom.simulate()
		eroom.simulate()

		I = len(room.rir) 		# n_mics
		J = len(room.rir[0]) 	# n_srcs

		mixtures = room.mic_array.signals
		# normalize std of the mixtures
		std_mixtures = np.std(mixtures, axis=1, keepdims=True)
		mixtures = mixtures / std_mixtures
		
		# simulate diffuse bubble noise
		if self.SNR < 100:
			L = mixtures.shape[1]
			# n = dsp.normalize_and_zero_mean(self.noise_signal)
			# noise = self.simulate_diffuse_noise(n)
			if self.noise is None:
				noise = 2*np.random.random([I,L]) - 1
			else:
				assert self.noise.shape[0] < self.noise.shape[1]
				noise = self.noise[:I,:L]

			# scale according to the SNR
			noise = noise / np.std(noise, axis=1, keepdims=True)
			sigma_n = np.sqrt(10 ** (- self.SNR / 10))
			noise = sigma_n * noise
		
			assert mixtures.shape[0] == I
			assert noise.shape[0] == I
			
			L = mixtures.shape[1]
			mixtures = mixtures[:,:L] + noise[:,:L]
		
		# compute DRR
		DRRs = []
		DERs = []
		for j in range(J):
			for i in range(I):
				# get the direct path TOA
				
				h_dir = aroom.rir[i][j]
				h_eco = eroom.rir[i][j]
				h_rev =  room.rir[i][j]

				E_dir = np.sum(np.abs(h_dir)**2)
				E_eco = np.sum(np.abs(h_eco)**2)
				E_rev = np.sum(np.abs(h_rev)**2)
				
				drr = 10*np.log10(E_dir / (E_rev - E_dir))
				der = 10*np.log10(E_dir / (E_eco - E_dir))
		
				DRRs.append(drr)
				DERs.append(der)

		self.acoustic_params = {
			'DRR' : np.mean(DRRs),
			'DER' : np.mean(DERs),
			'RT60' : self.RT60,
			'SNR'  : self.SNR,
		}

		return mixtures


	def plot(self,):
		aroom = self.setup_room()[1]
		aroom.plot()