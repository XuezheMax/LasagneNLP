THEANO_FLAGS='floatX=float32' python parser.py --mode parse --embedding senna --update adadelta \
 --batch_size 1 --num_units 100 --num_filters 20 --learning_rate 0.005 --decay_rate 0.1 --momentum 0.9 --grad_clipping 5 --regular none \
 --train "data/PTB3.0/PTB3.0-Penn2Malt/tmp.original" --dev "data/PTB3.0/PTB3.0-Penn2Malt/tmp.original" --test "data/PTB3.0/PTB3.0-Penn2Malt/tmp.original" \
 --embedding_dict "data/senna/senna.50d.gz" --patience 5 --punctuation ", . \`\` : ''"