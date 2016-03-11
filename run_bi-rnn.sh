THEANO_FLAGS='floatX=float32' python bi_rnn.py --fine_tune --embedding senna --oov embedding --update momentum \
 --batch_size 10 --num_units 100 --grad_clipping 3 --gamma '1e-6' --regular none --dropout \
 --train "data/conll2003/eng.train.conll" --dev "data/conll2003/eng.dev.conll" --test "data/conll2003/eng.test.conll" \
 --embedding_dict "data/senna/senna.50d.gz" --output_prediction
