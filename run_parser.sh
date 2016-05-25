THEANO_FLAGS='floatX=float32' python parser.py --mode parse --embedding random --update momentum \
 --batch_size 1 --num_units 200 --num_filters 30 --momentum 0.9 --grad_clipping 5 --regular none \
 --learning_rate_bottom 0.01 --decay_rate_bottom 0.05 --learning_rate_top 0.002 --decay_rate_top 0.05 \
 --train "data/PTB3.0/PTB3.0-Penn2Malt/tmp.original" --dev "data/PTB3.0/PTB3.0-Penn2Malt/tmp.original" --test "data/PTB3.0/PTB3.0-Penn2Malt/tmp.original" \
 --embedding_dict "data/senna/senna.50d.gz" --patience 5 --punctuation ", . \`\` : ''"