THEANO_FLAGS='floatX=float32' python parser.py --mode parse --embedding random --rnn LSTM --update adam \
 --batch_size 1 --num_units 200 --num_filters 30 --momentum 0.9 --grad_clipping 2 --max_norm 1.0 \
 --learning_rate 0.001 --decay_rate 0.1 --regular none --gamma 1e-3 --eta 1.0 \
 --train "data/PTB3.0/PTB3.0-Penn2Malt/tmp1.original" --dev "data/PTB3.0/PTB3.0-Penn2Malt/tmp1.original" --test "data/PTB3.0/PTB3.0-Penn2Malt/tmp1.original" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --patience 5 --punctuation ", . \`\` : ''"