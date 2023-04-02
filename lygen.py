import pathlib
import os
import json
import re
import keras
import torch
import random
import markovify
from unidecode import unidecode

import tensorflow as tf
import numpy as np
import pandas as pd
import lyricsgenius as lg

from tqdm import tqdm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, Flatten
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from base_model.lyrics_generation_model import Generator
from gpt2_finetuned_model.gpt2_finetuning import *

# tensorflow compatibility
tf.compat.v1.experimental.output_all_intermediates(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

API_KEY = "YOUR_API_KEY"

if API_KEY == "YOUR_API_KEY" or API_KEY == "":
    raise ValueError("Not so fast. Don't forget to instantiate your API key by going on https://genius.com/api-clients.")

base_model = keras.models.load_model('./base_model/base-model.h5')
tokenizer = pd.read_pickle('./base_model/tokenizer.pickle')

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def homepage(request: Request):
    ''' get homepage template '''

    return templates.TemplateResponse(
        "index.html", context={'request': request}
    )

def lr_decay(epoch, lr):
    ''' learning rate decay for base model finetuning (if epochs < 10) '''

    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1 * epoch)

def clean_data(artist, max_songs=10):
    ''' data preprocessing + creation of csv file with artist songs '''

    genius = lg.Genius(API_KEY, verbose=False, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)", "(Reprise)", "(Instrumental)"], remove_section_headers=True)

    if not pathlib.Path(f"logs/{artist.name}-songs.csv").exists():
        artist = genius.search_artist(artist.name, max_songs=max_songs, sort="popularity")
        songs_list = [song.title for song in artist.songs]

        lyrics = []
        for i, song in zip(tqdm(range(len(songs_list))), songs_list):
            if genius.search_song(song, artist.name) is not None:
                data = re.sub(f'{song} Lyrics\n', '', genius.search_song(song, artist.name).lyrics)
                data = re.sub(f'{song} Lyrics', '', data)
                data = re.sub(f'^.* Lyrics\n', '', data)
                data = re.sub('[0-9]{0,2}?Embed', '', data)
                data = re.sub('[0-9]{0,2}?EmbedShare URLCopyEmbedCopy', '', data)
                data = re.sub('You might also like', '', data)
                data = re.sub('TranslationsFrançais', '', data)
                data = re.sub('TranslationsEspañol', '', data)
                data = re.sub('Português', '', data)
                data = re.sub('Italiano', '', data)
                data = re.sub('Deutsch', '', data)
                data = re.sub('lyrics', '', data)
                lyrics.append(data)
            else:
                lyrics.append("NaN")

        data = {
            "artist": [artist.name]*len(songs_list),
            "title": songs_list,
            "lyrics": lyrics
        }

        df = pd.DataFrame(data)

        df.to_csv(f'logs/{artist.name}-songs.csv', sep=',')

    else:
        df = pd.read_csv(f'logs/{artist.name}-songs.csv', sep=',')

        if len(df) < max_songs:
            artist = genius.search_artist(artist.name, max_songs=max_songs, sort="popularity")
            songs_list = [song.title for song in artist.songs]

            lyrics = []
            for i, song in zip(tqdm(range(len(songs_list))), songs_list):
                if genius.search_song(song, artist.name) is not None:
                    data = re.sub(f'{song} Lyrics\n', '', genius.search_song(song, artist.name).lyrics)
                    data = re.sub(f'{song} Lyrics', '', data)
                    data = re.sub(f'^.* Lyrics\n', '', data)
                    data = re.sub('[0-9]{0,2}?Embed', '', data)
                    data = re.sub('[0-9]{0,2}?EmbedShare URLCopyEmbedCopy', '', data)
                    data = re.sub('You might also like', '', data)
                    data = re.sub('TranslationsFrançais', '', data)
                    data = re.sub('TranslationsEspañol', '', data)
                    data = re.sub('TranslationsPortuguês', '', data)
                    data = re.sub('lyrics', '', data)
                    lyrics.append(data)
                else:
                    lyrics.append("NaN")

            data = {
                "artist": [artist.name]*len(songs_list),
                "title": songs_list,
                "lyrics": lyrics
            }

            df = pd.DataFrame(data)

            df.to_csv(f'logs/{artist.name}-songs.csv', sep=',')
            df = pd.read_csv(f'logs/{artist.name}-songs.csv', sep=',')
        else:
            df = pd.read_csv(f'logs/{artist.name}-songs.csv', sep=',').sample(n=max_songs)


    data = ""
    for i in range(len(df)):
        if df.lyrics[i] != "NaN":
            data = str(data) + "\n\n" + str(df.lyrics[i])

    return data

def finetune(artist, generator, epochs=50):
    ''' finetuning of base pre-trained model '''

    global base_model

    filepath = f'base_model/finetuned_models/{artist}-finetuned-model.h5'

    if not pathlib.Path(filepath).exists(): # if the doesn't have a finetuned model, train

        callbacks  = [
            LearningRateScheduler(schedule=lr_decay, verbose=1),
            EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True),
            ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        ]

        _, lyrics = generator.tokenize()
        sequences, seq, vocab_size = generator.get_sequences(tokenizer, lyrics)
        input_sequences, output_labels = seq[:,:-1], seq[:,-1]
        one_hot_labels = to_categorical(output_labels, num_classes=vocab_size)

        # to avoid 'layer name already exists' error when finetuning multiple models + root scope name error due to non ascii chars (eg. Björk -> Bjork)
        formated_name = '_'.join([unidecode(word) for word in artist.split(' ')])

        for layer in base_model.layers:
            layer.trainable = False # freezing base model layers

        base_model.add(Dense(vocab_size, activation='softmax', name=f'output_dense_{formated_name}')) # adding dense layer for finetuning
        base_model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
        base_model.summary()
        generator.train(base_model, input_sequences, one_hot_labels, callbacks=callbacks, epochs=epochs)
    else:  # if the artist already has a finetuned model, don't retrain
        base_model = keras.models.load_model(filepath)

    return base_model

def get_generation(generator, model, seed):
    ''' get generation for finetuned base model '''

    return generator.generate(model, tokenizer, seed, lyric_length=100) 

def gpt2_finetune(dataset, epochs, seed):
    ''' finetune gpt2 on artist lyrics '''

    dataset = SongLyrics(dataset, truncate=True)
    generator = GPT2LyricsGenerator(dataset)

    model = generator.train(epochs=epochs)

    return generator.text_generation(model, seed)

def gpt2_generate(df, generated_lyrics):
    ''' generate lyrics with finetuned gpt2 '''

    generation = generated_lyrics[-1]
    generation = re.sub("\<\|beginoftext\|\>", '', generation)
    generation = re.sub("\<\|endoftext\|\>", '', generation)

    if generation in [None, '']:
        generation = 'Sorry, GPT-2 couldn\'t generate lyrics. Try uping the number of songs.'

    return generation

@app.post("/request")
async def generate(request: Request, artist_name: str = Form(None), epochs: int = Form(None), max_songs: int = Form(None), model: str = Form(...)):

    if not pathlib.Path("logs/").exists(): # if the logs directory doesn't exist, we create it
        pathlib.Path("logs/").mkdir(parents=True, exist_ok=True)

    if artist_name is None: # send error message if artist field is empty
        return templates.TemplateResponse(
        'error.html', context={'request': request, 'error':f'I unfortunately don\'t know of many artists with no name...'}
    )
    elif max_songs is None: # send error message if max_songs field is empty
        return templates.TemplateResponse(
        'error.html', context={'request': request, 'error':f'Aren\'t you forgetting something? How am I supposed to learn without material to train on?'}
    )
    elif epochs is None: # send error message if epochs field is empty
        return templates.TemplateResponse(
        'error.html', context={'request': request, 'error':f'I\'m just a machine. I need some training to learn this stuff. Don\'t forget to fill in the epochs field.'}
    )
    else:
        dictionary = {"request":{
            "artist": ' '.join([token.lower() for token in artist_name.split(' ')]),
            "epochs": epochs,
            "max_songs": max_songs,
            "model": model}
        }

        pathlib.Path("logs/request.json").open("w").write(json.dumps(dictionary))
        logs = json.loads(open("./logs/request.json").read())

        genius = lg.Genius(API_KEY, verbose=False, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)", "(Reprise)", "(Instrumental)"], remove_section_headers=True)
        artist = genius.search_artist(logs['request']['artist'], max_songs=1) # checking if artist exists with the lyricsgenius (if artist doesn't exist, response is None)

        if artist is None: # if artist doesn't exist
            return templates.TemplateResponse(
            'error.html', context={'request': request, 'error':f'{artist_name}? Never heard of them. Must be quite niche.'}
        )
        else: # if artist does exist
            data = clean_data(artist, max_songs) # clean dataset + create csv file of artist's songs

            lines = []
            for line in data.split('\n'):
                if len(line) > 30: # only keeping lines over 30 characters (smaller seeds generate worse outputs)
                    lines.append(line)

            seed = random.choice(lines) # randomly select seed (random line from artist's entire repertoire)
            seed = f"These are the lyrics to a song written by {artist.name}:\n" + seed + '\n'

            if model == 'base': # if model is pre-trained base model

                generator = Generator(data=data)
                model = finetune(artist.name, generator, epochs)
                generation = get_generation(generator, model, seed)

                return templates.TemplateResponse(
                'index.html', context={'request': request, 'artist_name': artist.name, 'epochs': epochs, 'max_songs': max_songs, 'model': model, 'generation': generation}
        )
            elif model == 'hmm': # if model is hmm

                text_model = markovify.Text(data)
                generation = text_model.make_short_sentence(500) # using markovify to generate short sentence

                if generation is None: # error message if not enough songs were chosen for generation
                    generation = "Sorry, couldn't make a prediction for this one. Try again with more songs."
                else: # else formating output
                    lst = re.findall('[A-Z][^A-Z]*', generation)
                    generation = "[Chorus:]\n" + '\n'.join(lst)

                return templates.TemplateResponse(
                'index.html', context={'request': request, 'artist_name': artist.name, 'epochs': epochs, 'max_songs': max_songs, 'model': model, 'generation': generation}
        )
            else: # if model is gpt2

                df = pd.read_csv(f'logs/{artist.name}-songs.csv')

                generated_lyrics = gpt2_finetune(df, epochs, f"<|beginoftext|>{seed}")
                generation = gpt2_generate(df, generated_lyrics)

                return templates.TemplateResponse(
                'index.html', context={'request': request, 'artist_name': artist.name, 'epochs': epochs, 'max_songs': max_songs, 'model': model, 'generation': str(generation)}
        )

@app.on_event("shutdown")
async def shutdown_event(): # when app shuts down
    os.system("rm -r logs/") # remove logs
    os.system("rm -f base_model/finetuned_models/*-finetuned-model.h5") # remove all finetuned models
    os.system("rm -f gpt2_finetuned_model/finetuned_models/gpt2-finetuned-model.h5") # remove gpt2 finetuned model