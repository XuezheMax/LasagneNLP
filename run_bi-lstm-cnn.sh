THEANO_FLAGS='floatX=float32' python bi_lstm_cnn.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 100 --num_filters 20 --grad_clipping 3 --gamma 1e-6 --regular none \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz"
