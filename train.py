import numpy as np
import util
import models
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


def train(batch_size, hidden_size, epoch, vocabulary_size, source_train, target_train):

    source_train_sentences, source_word2idx, idx_word = util.sentence2idx(source_train,
                                                                   vocab_size=V)
    target_train_sentences, target_word2idx, _ = util.sentence2idx(target_train,
                                                                   vocab_size=V,
                                                                   reverse=True)
    encoder_vocab_size = len(source_word2idx) + 1  # +1 == unk tag
    decoder_vocab_size = len(target_word2idx) + 1

    sentence_num = len(source_train_sentences)
    if len(source_train_sentences) != len(target_train_sentences):
        raise TypeError(str(len(source_train_sentences))+'!='+str(len(target_train_sentences)))

    encoder_maxlen = max([len(x) for x in source_train_sentences])
    decoder_maxlen = max([len(x) for x in target_train_sentences])

    X = pad_sequences(source_train_sentences, maxlen=encoder_maxlen)
    Y = pad_sequences(target_train_sentences, maxlen=decoder_maxlen)

    model = models.seq2seq(encoder_vocab_size, encoder_maxlen,
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
            Y_batch = np.array(list(zip(*Y_batch)))
            for j in range(encoder_maxlen-1):
                _Y_batch = Y_batch[j]
                label = np_utils.to_categorical(np.append(_Y_batch[1:], 0), decoder_vocab_size)
                loss += model.train_on_batch([X_batch, _Y_batch], label)
        print('epoch', _epoch, loss)
        if not _epoch % 5:
            model.save_weights('epoch_'+str(_epoch)+'.h5')
    model.save('test_model.h5')

if __name__ == '__main__':
    batch_size = 1000
    embedding_dim = 100
    hidden_size = 100
    V = 5000
    epoch = 31

    en_train = "../small_parallel_enja/train.en"
    ja_train = "../small_parallel_enja/train.ja"
    train(batch_size, hidden_size, epoch, V, en_train, ja_train)
