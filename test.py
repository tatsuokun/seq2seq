import util
import numpy as np
import json
from keras.models import load_model, model_from_json
from keras.preprocessing.sequence import pad_sequences


def test(model, source_test, target_test, encoder_maxlen, decoder_maxlen,
         en_word2idx, de_word2idx, de_idx2word):

    source_test_sentences, _, _ = util.sentence2idx(source_test,
                                                    vocab_size=False,
                                                    word_idx=en_word2idx)
    # target_train_sentences, target_word2idx, _ = util.sentence2idx(target_train,
    #                                                                vocab_size=V)
    encoder_maxlen = max([len(x) for x in source_train_sentences])
    # decoder_maxlen = max([len(x) for x in target_train_sentences])

    X = pad_sequences(source_test_sentences, maxlen=encoder_maxlen, padding='post', truncating='post')
    with open(target_test, mode='r') as f, open(source_test, mode='r') as f2:
        Y = [sentence.strip() for sentence in f]
        s = [sentence.strip() for sentence in f2]
    # sentence_num = len(X)
    decoded_word = "私"
    encoder_input = np.array([X[0]])
    decoder_input = np.array([de_word2idx[decoded_word]])

    print('================================')
    print('source: '+s[0])
    print('true: '+Y[0])

    for _ in range(encoder_maxlen):
        pred = model.predict([encoder_input, decoder_input])[0][1:]
        decoded_idx = pred.argmax()+1
        if decoded_idx in de_idx2word:
             print(de_idx2word[decoded_idx], end=' ')
        decoder_input = np.array([decoded_idx])
    print('')

    #     for j in range(len(decoded)):
    #         if not decoded[j]:
    #             print('<pad>', end=' ')
    #         else:
    #             print(de_idx2word[decoded[j]], end=' ')

    # for i in range(len(decode)):
    #     decoded = [(decode[i][j]/decode[i][j].sum()).argmax() for j in range(decoder_maxlen)]
    #     print('================================')
    #     print(s[i])
    #     for j in range(len(decoded)):
    #         if not decoded[j]:
    #             print('<pad>', end=' ')
    #         else:
    #             print(de_idx2word[decoded[j]], end=' ')
    #     print('')


if __name__ == '__main__':
    encoder_maxlen = 18
    decoder_maxlen = 18
    V = 5000
    source_train = "../small_parallel_enja/train.en"
    target_train = "../small_parallel_enja/train.ja"
    source_train_sentences, encode_word2idx, _ = util.sentence2idx(source_train,
                                                                   vocab_size=V)
    target_train_sentences, decode_word2idx, decode_idx2word = util.sentence2idx(target_train,
                                                                                 vocab_size=V)
    # model = load_model("./test_model.h5")
    model = model_from_json(json.load(open("./my_model.json")))
    model.load_weights("./epoch_5.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    en_test = "../small_parallel_enja/train.en"
    ja_test = "../small_parallel_enja/train.ja"
    _ = ''
    test(model, en_test, ja_test, encoder_maxlen, decoder_maxlen, encode_word2idx, decode_word2idx, decode_idx2word)
