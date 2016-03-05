THEANO_FLAGS='floatX=float32,device=gpu' python bi_lstm_cnn.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 100 --num_filters 20 --learning_rate 0.1 --grad_clipping 3 --gamma '1e-6' --regular dropout \
 --train "data/conll2003/eng.train.bioes.conll" --dev "data/conll2003/eng.dev.bioes.conll" --test "data/conll2003/eng.test.bioes.conll" \
 --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --output_prediction --patience 5
