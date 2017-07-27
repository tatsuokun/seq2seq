import util
import numpy as np
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from keras.models import load_model, model_from_json
from keras.preprocessing.sequence import pad_sequences


def test(model, source_test, target_test, batch_size,
         en_word2idx, de_word2idx, de_idx2word):

    source_test_sentences, _, _ = util.sentence2idx(source_test,
                                                    vocab_size=False,
                                                    word_idx=en_word2idx)
    encoder_maxlen = max([len(x) for x in source_train_sentences])
    decoder_maxlen = max([len(x) for x in target_train_sentences]) - 1  # -1 is needed because of removing EOS tag

    X = pad_sequences(source_test_sentences, maxlen=encoder_maxlen, padding='post', truncating='post')
    with open(target_test, mode='r') as f, open(source_test, mode='r') as f2:
        Y = [sentence.strip() for sentence in f]
        s = [sentence.strip() for sentence in f2]

    references = [sentence.split() for sentence in Y]
    encoder_dummy = np.zeros([batch_size-1,encoder_maxlen])
    decoder_dummy = np.zeros([batch_size-1,decoder_maxlen])
    hypotheses = []

    for i in range(len(X)):
        decoded_word = "</s>"
        encoder_input = np.array([X[i]])
        _encoder_input = np.concatenate((encoder_input, encoder_dummy), axis=0)
        decoded_words = np.array([de_word2idx[decoded_word]])
        output_sentence = []
        print('================================')
        print('source: '+s[i])
        print('true: '+Y[i])
        print('pred: ', end='')

        for _ in range(decoder_maxlen-1):
            decoder_input = pad_sequences([decoded_words], maxlen=decoder_maxlen, padding='post', truncating='post')
            _decoder_input = np.concatenate((decoder_input, decoder_dummy), axis=0)
            pred = model.predict([_encoder_input, _decoder_input], batch_size=batch_size)[0]
            pred[0] = -1
            decoded_idx = pred.argmax()
            if de_idx2word.get(decoded_idx) == '<EOS>':
                    break
            print(de_idx2word[decoded_idx], end=' ')
            output_sentence.append(de_idx2word[decoded_idx])
            decoded_words = np.append(decoded_words, decoded_idx)
        print('')
        hypotheses.append(output_sentence)
    print(corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1))


if __name__ == '__main__':
    V = 5000
    batch_size = 500
    source_train = "../small_parallel_enja/train.en"
    target_train = "../small_parallel_enja/train.ja"
    source_train_sentences, encode_word2idx, _ = util.sentence2idx(source_train,
                                                                   vocab_size=V)
    target_train_sentences, decode_word2idx, decode_idx2word = util.sentence2idx(target_train,
                                                                                 vocab_size=V,
                                                                                 train=True)
    # model = load_model("./test_model.h5")
    model = model_from_json(json.load(open("my_model.json")))
    model.load_weights("epoch_20.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    en_test = "../small_parallel_enja/test.en"
    ja_test = "../small_parallel_enja/test.ja"
    test(model, en_test, ja_test, batch_size, encode_word2idx, decode_word2idx, decode_idx2word)
