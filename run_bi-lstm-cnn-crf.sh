THEANO_FLAGS='floatX=float32' python bi_lstm_cnn_crf.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 100 --num_filters 20 --learning_rate 0.01 --grad_clipping 3 --gamma '1e-6' --regular l2 \
 --train "data/conll2003/eng.train.bio.conll" --dev "data/conll2003/eng.dev.bio.conll" --test "data/conll2003/eng.test.bio.conll" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz"
