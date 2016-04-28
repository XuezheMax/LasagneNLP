THEANO_FLAGS='floatX=float32' python smiles.py --depth 2 --embedd_dim 10 --num_filters 10 --filter_size 6 --activation tanh --drop_input 0.0 --drop_hidden 0.0 \
 --num_epochs 100 --batch_size 5 --learning_rate 0.001 --decay_rate 0.05 --update sgd \
 --regular none --gamma 1e-6