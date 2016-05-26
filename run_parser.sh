THEANO_FLAGS='floatX=float32' python parser.py --mode both --embedding random --rnn LSTM --update momentum \
 --batch_size 1 --num_units 200 --num_filters 30 --momentum 0.9 --grad_clipping 5 --max_norm 4.0 --regular none --eta 1.0 --peepholes --in_to_out \
 --learning_rate_bottom 0.05 --decay_rate_bottom 0.1 --learning_rate_top 0.05 --decay_rate_top 0.1 \
 --train "data/PTB3.0/PTB3.0-Penn2Malt/tmp1.original" --dev "data/PTB3.0/PTB3.0-Penn2Malt/tmp1.original" --test "data/PTB3.0/PTB3.0-Penn2Malt/tmp1.original" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --patience 5 --punctuation ", . \`\` : ''"