THEANO_FLAGS='floatX=float32' python parser.py --mode pos --embedding senna --update momentum \
 --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
 --train "data/PTB3.0/PTB3.0-Penn2Malt/eng.train.original" --dev "data/PTB3.0/PTB3.0-Penn2Malt/eng.dev.original" --test "data/PTB3.0/PTB3.0-Penn2Malt/eng.test.original" \
 --embedding_dict "data/senna/senna.50d.gz" --patience 5