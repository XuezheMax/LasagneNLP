THEANO_FLAGS='floatX=float32' python cifar10.py --architecture allConvB --num_epochs 1000 --batch_size 10 \
 --learning_rate_cnn 0.05 --learning_rate_dense 0.1 --decay_rate 0.1 \
 --momentum0 0.9 --momentum1 0.9 --momentum_type normal \
 --delta 0.0 --regular l2 --gamma 1e-3 --mc 1 --batch_mc 50 --patience 5
