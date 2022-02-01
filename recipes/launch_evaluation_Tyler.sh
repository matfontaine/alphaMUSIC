for ALPHA in  0.0;
do for SOURCE in 2 3;
do for RT60 in 0.256 0.512 1.024;
do for SNR in 0 5 10;
do for N_MIC in 5 8 10;
do for NOISE_TYPE in office cafet living;
do
  python launch_evaluation.py --gpu 0 --n_srcs ${SOURCE}\
  --alpha ${ALPHA} --RT60 ${RT60} --n_mics ${N_MIC} --snr ${SNR} --noise_type ${NOISE_TYPE}
done; done; done; done; done; done
