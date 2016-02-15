THEANO_FLAGS='floatX=float32' python bi_rnn.py --fine_tune --embedding senna --oov embedding --update momentum \
 --batch_size 10 --num_units 100 --grad_clipping 3 --gamma 1e-6 --regular dima \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" \
 --embedding_dict "data/senna/senna.50d.gz"
