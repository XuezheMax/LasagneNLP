THEANO_FLAGS='floatX=float32' python cifar10.py --num_epochs 1000 --batch_size 128 --decay_rate 0.005 \
 --delta 0.0 --regular none --gamma 1e-6 --patience 5