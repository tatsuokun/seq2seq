import json
from keras.layers import Input, Dense, Lambda, concatenate, multiply, add, Activation, LSTM, Embedding, SpatialDropout1D, RepeatVector, TimeDistributed, Flatten
from keras.models import Model
from keras import backend as K
from keras import activations


def seq2seq(encoder_vocab_size, encoder_maxlen,
            decoder_vocab_size, decoder_maxlen,
            hidden_size, save=False):

    '''
    Encoder
        Input is sequence of word indices after padding.
        ['I', 'am', 'Hnako', '.'] => conver word to index => [3, 4, 500, 1] 
        [3, 4, 500, 1]  => padding => [3, 4, 500, 1, 0, 0, 0, 0, 0, 0] <- This is the input
    '''

    encoder_input = Input(shape=(encoder_maxlen,), name='encoder_input')
    encoder_emb = Embedding(output_dim=hidden_size,
                            input_dim=encoder_vocab_size,
                            input_length=encoder_maxlen)(encoder_input)
    encoder_emb = SpatialDropout1D(0.2)(encoder_emb)  # 0.2 is dropping rate
    encoder = LSTM(hidden_size)(encoder_emb)
    encoder = RepeatVector(decoder_maxlen)(encoder)

    '''
    Decoder
        Input is [t-1:] word indices after padding.
        source sentence: I am Hanako
        target sentence: 私 は 花子 です 。
            ['私', 'は', '花子', 'です', '。'] => conver word to index => [3, 5,330, 4, 1] 
            [3, 5, 330, 4, 1] => add a start symbol => [5000, 3, 5, 330, 4, 1]

            When we predict the word 'は' given '私'
                [5000, 3]  => padding => [5000, 3, 0, 0, 0, 0, 0, 0 0, 0, 0, 0, 0,] <- This is the output
            When we predict the word 'です' given '私 は 花子'
                [5000, 3, 5, 330]  => padding => [5000, 3, 5, 330, 0, 0, 0, 0 0, 0, 0, 0, 0,] <- This is the output
    '''

    decoder_input = Input(shape=(decoder_maxlen,), name='decoder_input')  
    decoder_emb = Embedding(decoder_vocab_size,
                            hidden_size,
                            input_length=decoder_maxlen)(decoder_input)
    decoder_emb = SpatialDropout1D(0.2)(decoder_emb)  # 0.2 is dropping rate

    '''
    Encoder-Decoder
        Input for encoder-decoder is a pair of an encoder's output and a decoder's input.
        Process:
            concatenate inputs => densificate fully connected concatenated vectors into one dense vector by LSTM 
            => scale the vector to the vector whose hidden size is given vocaburaly size => softmax
    '''

    enc_dec = concatenate([encoder, decoder_emb], axis=-1)
    enc_dec = LSTM(hidden_size, return_sequences=True)(enc_dec)
    enc_dec = TimeDistributed(Dense(hidden_size))(enc_dec)
    enc_dec = Flatten()(enc_dec)
    main_output = Dense(decoder_vocab_size)(enc_dec)
    main_output = Activation('softmax', name='main_output')(main_output)

    '''
    Input for the Encoder-Decoder
        source sentence: I am Hanako .
        target sentence: 私 は 花子 です 。
        When we predict the word 'は' given '私'
            Encoder input: [3, 4, 500, 1, 0, 0, 0, 0, 0, 0]
            Decoder input: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Training Label: [0, 0, 0, 1, 0, ..., 0, 0, 0,]
                This is a one hot vector for the word '私', whose element corresponds to word index.
                The length of this one hot vector corresponds to vocaburaly size.
            Output of Encoder-Decoder: [0.01, 0.01, 0.01, 0.30, ..., 0.01] <- Summation of this is 1.
    '''
    model = Model(outputs=[main_output], inputs=[encoder_input, decoder_input])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.summary()

    if save:
        with open('my_model.json', 'w') as f:
                json.dump(model.to_json(), f)

    return model

# function for Lambda layers
def last_layer(x):
    get_last_layer = lambda x:x[:,-1,:]
    return get_last_layer(x)

def last_layer_output(input_shape):
    return (input_shape[0], input_shape[2])

def sum_tensor(x):
    return K.sum(x, axis=1)

def sum_tensor_output(input_shape):
    return (input_shape[0],input_shape[2])


def seq2seq_attention(encoder_vocab_size, encoder_maxlen,
                      decoder_vocab_size, decoder_maxlen,
                      hidden_size, save=False):

    encoder_input = Input(shape=(encoder_maxlen,), name='encoder_input')
    encoder_emb = Embedding(output_dim=hidden_size,
                            input_dim=encoder_vocab_size,
                            input_length=encoder_maxlen)(encoder_input)
    encoder_emb = SpatialDropout1D(0.2)(encoder_emb)  # 0.2 is dropping rate
    encoder = LSTM(hidden_size, return_sequences=True)(encoder_emb)  # save sequences for attention
    encoder_last_layer = Lambda(last_layer, output_shape=last_layer_output)(encoder)
    encoder_out = RepeatVector(decoder_maxlen)(encoder_last_layer)

    decoder_input = Input(shape=(decoder_maxlen,), name='decoder_input')  
    decoder_emb = Embedding(decoder_vocab_size,
                            hidden_size,
                            input_length=decoder_maxlen)(decoder_input)
    decoder_emb = SpatialDropout1D(0.2)(decoder_emb)  # 0.2 is dropping rate

    enc_dec = concatenate([encoder_out, decoder_emb], axis=-1)
    enc_dec_hidden = LSTM(hidden_size)(enc_dec)

    attention_input = RepeatVector(encoder_maxlen)(enc_dec_hidden)
    attention = concatenate([encoder, attention_input], axis=-1)
    attention = TimeDistributed(Dense(1))(attention)
    attention = Lambda(lambda x: activations.softmax(x, axis=1))(attention)
    attention = multiply([encoder, attention])
    attention = Lambda(sum_tensor, output_shape=sum_tensor_output)(attention)

    main_output = concatenate([enc_dec_hidden, attention], axis=-1)
    main_output = Dense(hidden_size, activation='tanh')(main_output)
    main_output = Dense(decoder_vocab_size)(main_output)
    main_output = Activation('softmax', name='main_output')(main_output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[main_output])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.summary()

    if save:
        with open('my_model.json', 'w') as f:
                json.dump(model.to_json(), f)

    return model
