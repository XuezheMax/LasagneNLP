THEANO_FLAGS='floatX=float32' python cifar100.py --architecture allConvB  --opt linear --num_epochs 400 --batch_size 128 \
 --learning_rate_cnn 0.05 --decay_rate_cnn 0.1 --learning_rate_dense 0.02 --decay_rate_dense 0.01 \
 --momentum0 0.9 --momentum1 0.9 --momentum_type nesterov \
 --delta 0.0 --regular l2 --gamma 1e-3 --max_norm 2.0 --mc 1 --batch_mc 10 --patience 5
