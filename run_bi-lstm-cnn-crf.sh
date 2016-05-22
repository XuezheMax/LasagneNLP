THEANO_FLAGS='floatX=float32' python bi_lstm_cnn_crf.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --patience 5
