THEANO_FLAGS='floatX=float32' python bi_lstm.py --embedding glove --oov embedding --update sgd --batch_size 10 --num_units 100 --grad_clipping 0 --regular none \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz"
