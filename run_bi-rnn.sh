THEANO_FLAGS='floatX=float32' python bi_rnn.py --embedding word2vec --oov embedding --batch_size 10 --num_units 100 --regular none \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original"
