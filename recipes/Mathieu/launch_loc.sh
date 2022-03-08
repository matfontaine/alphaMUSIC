for SEED in $(seq 20);
do for ALPHA in  1.2;
do for SOURCE in 3;
do for RT60 in 0.256;
do
  python launch_experiment.py --gpu 0 --n_srcs ${SOURCE} --seed ${SEED}\
  --alpha ${ALPHA} --RT60 ${RT60}
done; done; done; done
