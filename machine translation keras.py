import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

# Load English data
english_sentences = load_data('data/small_vocab_en')
# Load French data
french_sentences = load_data('data/small_vocab_fr')
 
# Vocabulary
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

# Preprocess
## Tokenize
def tokenize(x):
    """
    Tokenize x
    """
    # TODO: Implement
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

## Padding
def pad(x, length=None):
    """
    Pad x
    """
    if length is None:
        length=max([len(sentence) for sentence in x])
        print(length)
    return pad_sequences(x, maxlen=length, padding ='post')

## Preprocess Pipeline
def preprocess(x, y):
    """
    Preprocess x and y
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

# Models
## Ids Back to Text
def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

############################ Model 1: RNN ############################
# def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
#     """
#     Build and train a basic RNN on x and y
#     """
#     
#     learning_rate=2e-3
#     input_seq= Input(input_shape[1:])
#     rnn=GRU(64, return_sequences=True)(input_seq)
#     logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    
#     model = Model(input_seq, Activation('softmax')(logits))
#     model.compile(loss=sparse_categorical_crossentropy,
#                   optimizer=Adam(learning_rate),
#                   metrics=['accuracy'])
#     return model

# # Reshaping the input to work with a basic RNN
# tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
# tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# # Train the neural network
# simple_rnn_model = simple_model(
#     tmp_x.shape,
#     max_french_sequence_length,
#     english_vocab_size,
#     french_vocab_size)
# simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=50, validation_split=0.2)

# # Print prediction(s)
# print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
###########################################################################

############################## Model 2: Embedding #########################
# def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
#     """
#     Build and train a RNN model using word embedding on x and y
#     """
#     batch = 256
#     learning_rate = 2e-3
    

#     rnn = GRU(batch, return_sequences=True)
#     print(input_shape[1])
#     embedding=Embedding(french_vocab_size, batch, input_length=input_shape[1])
#     logits = TimeDistributed(Dense(french_vocab_size, activation='softmax'))
    
#     model = Sequential()
#     model.add(embedding)
#     model.add(rnn)
#     model.add(logits)
#     model.compile(loss=sparse_categorical_crossentropy,
#                   optimizer=Adam(learning_rate),
#                   metrics=['accuracy'])
    
    
#     return model

# tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
# tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

# model_with_embdding= embed_model(tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)
# model_with_embdding.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# print(logits_to_text(model_with_embdding.predict(tmp_x[:1])[0], french_tokenizer))
###############################################################################

########################### Model 3: Bidirectional RNNs #######################
# def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
#     """
#     Build and train a bidirectional RNN model on x and y
#     """
#     batch = 256
#     learning_rate = 2e-3
    
#     rnn = GRU(batch, return_sequences=True)
#     bi = Bidirectional(rnn, input_shape=input_shape[1:])
#     logits = TimeDistributed(Dense(french_vocab_size, activation='softmax'))
    
#     model = Sequential()
#     model.add(bi)
#     model.add(logits)
#     model.compile(loss=sparse_categorical_crossentropy,
#                   optimizer=Adam(learning_rate),
#                   metrics=['accuracy'])

#     return model

# tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
# tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# bidirectional_model= bd_model(tmp_x.shape, preproc_french_sentences.shape[1], english_vocab_size, french_vocab_size)
# bidirectional_model.fit(tmp_x, preproc_french_sentences, batch_size=512, epochs=10, validation_split=0.2)
################################################################################

########################### Model 4: Encoder-Decoder ###########################
def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    """
    batch = 256
    learning_rate = 3e-3

    rnn = GRU(batch, return_sequences=True, dropout=0.01)
    repeatVector=RepeatVector(output_sequence_length)
    logits = TimeDistributed(Dense(french_vocab_size, activation='softmax'))
    
    model = Sequential()
    model.add(GRU(batch, input_shape = input_shape[1:]))
    model.add(repeatVector)
    model.add(rnn)
    model.add(logits)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model

tests.test_encdec_model(encdec_model)

tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

encdec_model= encdec_model(tmp_x.shape, preproc_french_sentences.shape[1], english_vocab_size, french_vocab_size)
encdec_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)
###################################################################################

################################### Prediction ####################################
def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    """
    tmp_x = pad(preproc_english_sentences)
    
    model= model_final(tmp_x.shape, preproc_french_sentences.shape[1], english_vocab_size, french_vocab_size)
    model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)
    
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)