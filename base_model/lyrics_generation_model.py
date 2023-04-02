''' NB : base model was trained on Google Colab with lyrics_generator.ipynb file '''

import re
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

dataset = pd.read_csv("./base_model/dataset/Songs.csv", sep=',')
dataset = dataset.sample(n=300).reset_index(drop=True)

class Generator():
    ''' model class '''
    
    def __init__(self, data, max_seq_len=25):

        self.max_seq_len = max_seq_len
        self.data = data

        self.tokenizer = Tokenizer()

        self.words = [char for char in sorted(list(set(re.split(r'\s|\n|\n\n', self.data)))) if char != ''] # vocabulary
        self.vocabulary = len(self.words) # vocabulary size
        self.mapped_words = dict((i, c) for i, c in enumerate(self.words)) # mapping vocab to idx

    def stack_layers(self, vocab_size, inputs, outputs):
        ''' create & compile model '''

        model = Sequential()
        model.add(Embedding(vocab_size, 160, input_length=self.max_seq_len-1))
        model.add(Bidirectional(LSTM(200, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(vocab_size/2, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(vocab_size, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, model, inputs, outputs, callbacks=None, epochs=20):
        ''' train model '''

        model.fit(inputs, outputs, epochs=epochs, batch_size=32, shuffle=True, verbose=1, callbacks=callbacks)

        return model

    def cleaner(self):
        ''' clean data '''

        lyrics = self.data.split('\n')

        for item in range(len(lyrics)):
            lyrics[item] = lyrics[item].rstrip() # remove trailing spaces

        lyrics = [item for item in lyrics if item != '']

        return lyrics

    def tokenize(self):
        ''' tokenize data '''

        lyrics = self.cleaner()

        self.tokenizer.fit_on_texts(lyrics)

        with open('./tokenizers/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # write fitted tokenizer to pickle file

        return self.tokenizer, lyrics

    def get_sequences(self, tokenizer, lyrics):
        ''' get sequences & pad to max_seq_len '''

        seq = []
        for item in lyrics:
            sequences = tokenizer.texts_to_sequences([item])[0]

            for i in range(1, len(sequences)):
                n_gram = sequences[:i+1]
                seq.append(n_gram)

        seq = np.array(pad_sequences(seq, maxlen=self.max_seq_len, padding='pre'))
        vocab_size = len(tokenizer.word_index)+1 # set vocab_size to vocab_size+1 to avoid out of bounds error
    
        return sequences, seq, vocab_size

    def generate(self, model, tokenizer, seed, lyric_length):
        ''' generate next lyrics based on seed '''

        for _ in range(lyric_length):
            token_list = tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_seq_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted = np.random.choice([x for x in range(len(predicted_probs))], p=predicted_probs) # get predicted word idx

            output = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output = word # get word that matches predicted word from vocabulary (word/idx mapping)
                    break

            seed += " " + output

        return ''.join(seed)

if __name__ == '__main__':

    generator = Generator(data=data)

    tokenizer, lyrics = generator.tokenize()
    sequences, seq, vocab_size = generator.get_sequences(tokenizer, lyrics)
    input_sequences, output_labels = seq[:,:-1], seq[:,-1]
    one_hot_labels = to_categorical(output_labels, num_classes=vocab_size)

    filepath = 'base-model.h5'
    callbacks  = [
                EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True),
                ModelCheckpoint(filepath=filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            ]

    stacked_layers = generator.stack_layers(vocab_size, input_sequences, output_labels)
    model = generator.train(stacked_layers, input_sequences, one_hot_labels, callbacks=callbacks, epochs=30)