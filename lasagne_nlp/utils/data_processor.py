__author__ = 'max'

import numpy as np
import theano

from alphabet import Alphabet
from lasagne_nlp.utils import utils as utils

root_symbol = "##ROOT##"
root_label = "<ROOT>"
word_end = "##WE##"
MAX_LENGTH = 120
MAX_CHAR_LENGTH = 30
logger = utils.get_logger("LoadData")


def read_conll_sequence_labeling(path, word_alphabet, label_alphabet, word_column=1, label_column=4):
    """
    read data from file in conll format
    :param path: file path
    :param word_column: the column index of word (start from 0)
    :param label_column: the column of label (start from 0)
    :param word_alphabet: alphabet of words
    :param label_alphabet: alphabet -f labels
    :return: sentences of words and labels, sentences of indexes of words and labels.
    """

    word_sentences = []
    label_sentences = []

    word_index_sentences = []
    label_index_sentences = []

    words = []
    labels = []

    word_ids = []
    label_ids = []

    num_tokens = 0
    with open(path) as file:
        for line in file:
            if line.strip() == "":
                if 0 < len(words) <= MAX_LENGTH:
                    word_sentences.append(words[:])
                    label_sentences.append(labels[:])

                    word_index_sentences.append(word_ids[:])
                    label_index_sentences.append(label_ids[:])

                    num_tokens += len(words)
                else:
                    if len(words) != 0:
                        logger.info("ignore sentence with length %d" % (len(words)))

                words = []
                labels = []

                word_ids = []
                label_ids = []
            else:
                tokens = line.strip().split()
                word = tokens[word_column]
                label = tokens[label_column]

                words.append(word)
                labels.append(label)

                word_id = word_alphabet.get_index(word)
                label_id = label_alphabet.get_index(label)

                word_ids.append(word_id)
                label_ids.append(label_id)

    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences), num_tokens))
    return word_sentences, label_sentences, word_index_sentences, label_index_sentences


def generate_character_data(sentences_train, sentences_dev, sentences_test, max_sent_length, char_embedd_dim=30):
    """
    generate data for charaters
    :param sentences_train:
    :param sentences_dev:
    :param sentences_test:
    :param max_sent_length:
    :return: C_train, C_dev, C_test, char_embedd_table
    """

    def get_character_indexes(sentences):
        index_sentences = []
        max_length = 0
        for words in sentences:
            index_words = []
            for word in words:
                index_chars = []
                if len(word) > max_length:
                    max_length = len(word)

                for char in word[:MAX_CHAR_LENGTH]:
                    char_id = char_alphabet.get_index(char)
                    index_chars.append(char_id)

                index_words.append(index_chars)
            index_sentences.append(index_words)
        return index_sentences, max_length

    def construct_tensor_char(index_sentences):
        C = np.empty([len(index_sentences), max_sent_length, max_char_length], dtype=np.int32)
        word_end_id = char_alphabet.get_index(word_end)

        for i in range(len(index_sentences)):
            words = index_sentences[i]
            sent_length = len(words)
            for j in range(sent_length):
                chars = words[j]
                char_length = len(chars)
                for k in range(char_length):
                    cid = chars[k]
                    C[i, j, k] = cid
                # fill index of word end after the end of word
                C[i, j, char_length:] = word_end_id
            # Zero out C after the end of the sentence
            C[i, sent_length:, :] = 0
        return C

    def build_char_embedd_table():
        scale = np.sqrt(3.0 / char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [char_alphabet.size(), char_embedd_dim]).astype(
            theano.config.floatX)
        return char_embedd_table

    char_alphabet = Alphabet('character')
    char_alphabet.get_index(word_end)

    index_sentences_train, max_char_length_train = get_character_indexes(sentences_train)
    index_sentences_dev, max_char_length_dev = get_character_indexes(sentences_dev)
    index_sentences_test, max_char_length_test = get_character_indexes(sentences_test)

    # close character alphabet
    char_alphabet.close()

    max_char_length = min(MAX_CHAR_LENGTH, max(max_char_length_train, max_char_length_dev, max_char_length_test))
    logger.info("Maximum character length of training set is %d" % max_char_length_train)
    logger.info("Maximum character length of dev set is %d" % max_char_length_dev)
    logger.info("Maximum character length of test set is %d" % max_char_length_test)
    logger.info("Maximum character length used for training is %d" % max_char_length)

    # fill character tensor
    C_train = construct_tensor_char(index_sentences_train)
    C_dev = construct_tensor_char(index_sentences_dev)
    C_test = construct_tensor_char(index_sentences_test)

    return C_train, C_dev, C_test, build_char_embedd_table()


def load_dataset_sequence_labeling(train_path, dev_path, test_path, word_column=1, label_column=4,
                                   label_name='pos', oov='embedding', fine_tune=False, embedding="word2Vec",
                                   embedding_path="data/word2vec/GoogleNews-vectors-negative300.bin",
                                   use_character=False):
    """
    load data from file
    :param train_path: path of training file
    :param dev_path: path of dev file
    :param test_path: path of test file
    :param word_column: the column index of word (start from 0)
    :param label_column: the column of label (start from 0)
    :param label_name: name of label, such as pos or ner
    :param oov: embedding for oov word, choose from ['random', 'embedding']. If "embedding", then add words in dev and
                test data to alphabet; if "random", not.
    :param fine_tune: if fine tune word embeddings.
    :param embedding: embeddings for words, choose from ['word2vec', 'senna'].
    :param embedding_path: path of file storing word embeddings.
    :param use_character: if use character embeddings.
    :return: X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test,
            embedd_table (if fine tune), label_alphabet, C_train, C_dev, C_test, char_embedd_table
    """

    def get_max_length(word_sentences):
        max_len = 0
        for sentence in word_sentences:
            length = len(sentence)
            if length > max_len:
                max_len = length
        return max_len

    def construct_tensor_fine_tune(word_index_sentences, label_index_sentences):
        X = np.empty([len(word_index_sentences), max_length], dtype=np.int32)
        Y = np.empty([len(word_index_sentences), max_length], dtype=np.int32)
        mask = np.zeros([len(word_index_sentences), max_length], dtype=theano.config.floatX)

        for i in range(len(word_index_sentences)):
            word_ids = word_index_sentences[i]
            label_ids = label_index_sentences[i]
            length = len(word_ids)
            for j in range(length):
                wid = word_ids[j]
                label = label_ids[j]
                X[i, j] = wid
                Y[i, j] = label - 1

            # Zero out X after the end of the sequence
            X[i, length:] = 0
            # Copy the last label after the end of the sequence
            Y[i, length:] = Y[i, length - 1]
            # Make the mask for this sample 1 within the range of length
            mask[i, :length] = 1
        return X, Y, mask

    def build_embedd_table(embedd_dict, embedd_dim, caseless):
        scale = np.sqrt(3.0 / embedd_dim)
        embedd_table = np.empty([word_alphabet.size(), embedd_dim], dtype=theano.config.floatX)
        embedd_table[word_alphabet.default_index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
        for word, index in word_alphabet.iteritems():
            ww = word.lower() if caseless else word
            embedd = embedd_dict[ww] if ww in embedd_dict else np.random.uniform(-scale, scale, [1, embedd_dim])
            embedd_table[index, :] = embedd
        return embedd_table

    def generate_dataset_fine_tune():
        """
        generate data tensor when fine tuning
        :return: X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, embedd_table, label_size
        """

        embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict(embedding, embedding_path, logger)
        logger.info("Dimension of embedding is %d, Caseless: %d" % (embedd_dim, caseless))
        # fill data tensor (X.shape = [#data, max_length], Y.shape = [#data, max_length])
        X_train, Y_train, mask_train = construct_tensor_fine_tune(word_index_sentences_train,
                                                                  label_index_sentences_train)
        X_dev, Y_dev, mask_dev = construct_tensor_fine_tune(word_index_sentences_dev, label_index_sentences_dev)
        X_test, Y_test, mask_test = construct_tensor_fine_tune(word_index_sentences_test, label_index_sentences_test)
        C_train, C_dev, C_test, char_embedd_table = generate_character_data(word_sentences_train, word_sentences_dev,
                                                                            word_sentences_test,
                                                                            max_length) if use_character else (
            None, None, None, None)
        return X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
               build_embedd_table(embedd_dict, embedd_dim, caseless), label_alphabet, \
               C_train, C_dev, C_test, char_embedd_table

    def construct_tensor_not_fine_tune(word_sentences, label_index_sentences, unknown_embedd, embedd_dict,
                                       embedd_dim, caseless):
        X = np.empty([len(word_sentences), max_length, embedd_dim], dtype=theano.config.floatX)
        Y = np.empty([len(word_sentences), max_length], dtype=np.int32)
        mask = np.zeros([len(word_sentences), max_length], dtype=theano.config.floatX)

        # bad_dict = dict()
        # bad_num = 0
        for i in range(len(word_sentences)):
            words = word_sentences[i]
            label_ids = label_index_sentences[i]
            length = len(words)
            for j in range(length):
                word = words[j].lower() if caseless else words[j]
                label = label_ids[j]
                embedd = embedd_dict[word] if word in embedd_dict else unknown_embedd
                X[i, j, :] = embedd
                Y[i, j] = label - 1

                # if word not in embedd_dict:
                #     bad_num += 1
                #     if word in bad_dict:
                #         bad_dict[word] += 1
                #     else:
                #         bad_dict[word] = 1

            # Zero out X after the end of the sequence
            X[i, length:] = np.zeros([1, embedd_dim], dtype=theano.config.floatX)
            # Copy the last label after the end of the sequence
            Y[i, length:] = Y[i, length - 1]
            # Make the mask for this sample 1 within the range of length
            mask[i, :length] = 1

        # for w, c in bad_dict.items():
        #     if c >= 100:
        #         print "%s: %d" % (w, c)
        # print bad_num

        return X, Y, mask

    def generate_dataset_not_fine_tune():
        """
        generate data tensor when not fine tuning
        :return: X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, None, label_size
        """

        embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict(embedding, embedding_path, logger)
        logger.info("Dimension of embedding is %d, Caseless: %s" % (embedd_dim, caseless))

        # fill data tensor (X.shape = [#data, max_length, embedding_dim], Y.shape = [#data, max_length])
        unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])
        X_train, Y_train, mask_train = construct_tensor_not_fine_tune(word_sentences_train,
                                                                      label_index_sentences_train, unknown_embedd,
                                                                      embedd_dict, embedd_dim, caseless)
        X_dev, Y_dev, mask_dev = construct_tensor_not_fine_tune(word_sentences_dev, label_index_sentences_dev,
                                                                unknown_embedd, embedd_dict, embedd_dim, caseless)
        X_test, Y_test, mask_test = construct_tensor_not_fine_tune(word_sentences_test, label_index_sentences_test,
                                                                   unknown_embedd, embedd_dict, embedd_dim, caseless)
        C_train, C_dev, C_test, char_embedd_table = generate_character_data(word_sentences_train, word_sentences_dev,
                                                                            word_sentences_test,
                                                                            max_length) if use_character else (
            None, None, None, None)

        return X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
               None, label_alphabet, C_train, C_dev, C_test, char_embedd_table

    word_alphabet = Alphabet('word')
    label_alphabet = Alphabet(label_name)

    # read training data
    logger.info("Reading data from training set...")
    word_sentences_train, _, word_index_sentences_train, label_index_sentences_train = read_conll_sequence_labeling(
        train_path, word_alphabet, label_alphabet, word_column, label_column)

    # if oov is "random" and do not fine tune, close word_alphabet
    if oov == "random" and not fine_tune:
        logger.info("Close word alphabet.")
        word_alphabet.close()

    # read dev data
    logger.info("Reading data from dev set...")
    word_sentences_dev, _, word_index_sentences_dev, label_index_sentences_dev = read_conll_sequence_labeling(
        dev_path, word_alphabet, label_alphabet, word_column, label_column)

    # read test data
    logger.info("Reading data from test set...")
    word_sentences_test, _, word_index_sentences_test, label_index_sentences_test = read_conll_sequence_labeling(
        test_path, word_alphabet, label_alphabet, word_column, label_column)

    # close alphabets
    word_alphabet.close()
    label_alphabet.close()

    logger.info("word alphabet size: %d" % (word_alphabet.size() - 1))
    logger.info("label alphabet size: %d" % (label_alphabet.size() - 1))

    # get maximum length
    max_length_train = get_max_length(word_sentences_train)
    max_length_dev = get_max_length(word_sentences_dev)
    max_length_test = get_max_length(word_sentences_test)
    max_length = min(MAX_LENGTH, max(max_length_train, max_length_dev, max_length_test))
    logger.info("Maximum length of training set is %d" % max_length_train)
    logger.info("Maximum length of dev set is %d" % max_length_dev)
    logger.info("Maximum length of test set is %d" % max_length_test)
    logger.info("Maximum length used for training is %d" % max_length)

    if fine_tune:
        logger.info("Generating data with fine tuning...")
        return generate_dataset_fine_tune()
    else:
        logger.info("Generating data without fine tuning...")
        return generate_dataset_not_fine_tune()
