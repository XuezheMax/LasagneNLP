THEANO_FLAGS='floatX=float32' python cifar100.py --num_epochs 1000 --batch_size 128 --decay_rate 0.005 \
 --delta 0.0 --regular l2 --gamma 1e-3 --patience 5