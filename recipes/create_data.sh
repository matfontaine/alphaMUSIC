for SEED in $(seq 20);
do for SOURCE in 2 3;
do for RT60 in 0.256 0.512 1.024;
do for N_MIC in 10;
do for SNR in 0 5 10;
do for NOISE_TYPE in office cafet living;
do
  python create_dataset.py --gpu 0 --n_srcs ${SOURCE} --seed ${SEED}\
  --RT60 ${RT60} --n_mics ${N_MIC} --snr ${SNR} --noise_type ${NOISE_TYPE}
done; done; done; done; done; done
