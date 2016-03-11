THEANO_FLAGS='floatX=float32' python bi_lstm_highcnn.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 100 --num_filters 20 --learning_rate 0.1 --decay_rate 0.1 --grad_clipping 3 --regular none --dropout \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --output_prediction --patience 5
