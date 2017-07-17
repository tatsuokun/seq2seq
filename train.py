import numpy as np
import util
import models
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


def train(batch_size, hidden_size, epoch, vocabulary_size, source_train, target_train,
          source_word2idx, idx_word, target_word2idx):

    # X: source_train_sentences
    # Y: target_train_sentences
    X, _, _ = util.sentence2idx(source_train,
                                vocab_size=V,
                                word_idx=source_word2idx)
    Y, _, _ = util.sentence2idx(target_train,
                                vocab_size=V,
                                word_idx=target_word2idx,
                                reverse=True)

    encoder_vocab_size = len(source_word2idx) + 1  # +1 == unk tag
    decoder_vocab_size = len(target_word2idx) + 2  # +2 == unk tag + BOS tag 

    sentence_num = len(X)
    if len(X) != len(Y):
        raise TypeError(str(len(X))+'!='+str(len(Y)))

    encoder_maxlen = max([len(x) for x in X])
    decoder_maxlen = max([len(x) for x in Y])

    model = models.seq2seq_attention(encoder_vocab_size, encoder_maxlen,
                                     decoder_vocab_size, decoder_maxlen,
                                     hidden_size, save=True)

    print("model loaded")
    print("start training")
    for _epoch in range(epoch):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        loss = 0
        for i in range(0, sentence_num, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            encoder_input = pad_sequences(X_batch, maxlen=encoder_maxlen, padding='post', truncating='post')
            label_pad = pad_sequences(Y_batch, maxlen=encoder_maxlen, padding='post', truncating='post')
            label = list(zip(*label_pad))[1:]
            for j in range(decoder_maxlen-1):
                Y_batch_jth_sequence = [Y_batch[k][:j+1] for k in range(batch_size)]
                decoder_input = pad_sequences(Y_batch_jth_sequence, maxlen=encoder_maxlen, padding='post', truncating='post')
                label_categorical = np_utils.to_categorical(label[j], decoder_vocab_size)
                loss += model.train_on_batch([encoder_input, decoder_input], label_categorical)
        print('epoch', _epoch, loss)
        if not _epoch % 5:
            model.save_weights('epoch_'+str(_epoch)+'.h5')
    model.save('test_model.h5')


if __name__ == '__main__':
    batch_size = 1000
    embedding_dim = 100
    hidden_size = 256
    V = 5000
    epoch = 31

    en_train = "../small_parallel_enja/train.en"
    ja_train = "../small_parallel_enja/train.ja"

    en_test = "../small_parallel_enja/test.en"
    ja_test = "../small_parallel_enja/test.ja"

    _, source_word2idx, idx_word = util.sentence2idx(en_train,
                                                     vocab_size=V)
    _, target_word2idx, _ = util.sentence2idx(ja_train,
                                              vocab_size=V,
                                              reverse=True)

    train(batch_size, hidden_size, epoch, V, en_train, ja_train, source_word2idx, idx_word, target_word2idx)
