THEANO_FLAGS='floatX=float32' python cifar100.py --num_epochs 1000 --batch_size 64 --learning_rate 0.01 --decay_rate 0.01 --update momentum \
 --delta 0.0 --regular none --gamma 1e-6 --patience 5