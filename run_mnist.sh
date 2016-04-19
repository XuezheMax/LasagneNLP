THEANO_FLAGS='floatX=float32' python mnist.py --depth 2 --num_units 1024 --activation rectify \
 --num_epochs 1000 --batch_size 500 --learning_rate 0.1 --decay_rate 0.05 --update momentum --regular none --gamma 1e-6 \
 --patience 5