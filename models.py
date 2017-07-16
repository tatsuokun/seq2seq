import json
from keras.layers import Input, Dense, concatenate, Activation, LSTM, Embedding, SpatialDropout1D, RepeatVector, TimeDistributed
from keras.models import Model


def seq2seq(encoder_vocab_size, encoder_maxlen,
            decoder_vocab_size, decoder_maxlen,
            hidden_size, save=False):

    # Encoder
    encoder_input = Input(shape=(encoder_maxlen,), name='encoder_input')
    encoder_emb = Embedding(output_dim=hidden_size,
                            input_dim=encoder_vocab_size,
                            input_length=encoder_maxlen)(encoder_input)
    encoder_emb = SpatialDropout1D(0.2)(encoder_emb)  # 0.2 is a dropping rate
    encoder = LSTM(hidden_size)(encoder_emb)
    encoder = RepeatVector(decoder_maxlen)(encoder)

    # Decoder
    decoder_input = Input(shape=(1,), name='decoder_input')  # input for the decorder is the t-1 latter
    decoder_emb = Embedding(decoder_vocab_size,
                            hidden_size,
                            input_length=1)(decoder_input)
    decoder_emb = SpatialDropout1D(0.2)(decoder_emb)  # 0.2 is a dropping rate

    # Encoder-Decoder
    enc_dec = concatenate([encoder, decoder_emb], axis=-2)
    enc_dec = LSTM(hidden_size, return_sequences=True)(enc_dec)
    enc_dec = TimeDistributed(Dense(hidden_size))(enc_dec)

    main_output = LSTM(hidden_size, return_sequences=False)(enc_dec)
    main_output = Dense(decoder_vocab_size)(main_output)
    main_output = Activation('softmax', name='main_output')(main_output)

    model = Model(outputs=main_output, inputs=[encoder_input, decoder_input])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.summary()

    if save:
        with open('my_model.json', 'w') as f:
                json.dump(model.to_json(), f)

    return model
