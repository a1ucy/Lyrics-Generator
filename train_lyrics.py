import numpy as np
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers.legacy import Adam, RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
# tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], 'GPU')

min_freq = 5
emb_size = 256

df = pd.read_csv('drake_songs_clean.csv')
data = list(df.songs.values)
texts = " ".join(data)

# build vocab
def yield_tokens(data):
    yield data.split(" ")

vocab = build_vocab_from_iterator(yield_tokens(texts), min_freq=min_freq, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_file_path = 'vocab.pth'
torch.save(vocab, vocab_file_path)

input_seq = []
for idx, i in enumerate(data):
    tokens = i.split(" ")
    text_idx = vocab(tokens)
    for j in range(1, len(text_idx)):
        n_gram = text_idx[:idx+1]
        input_seq.append(n_gram)

max_len = max([len(i) for i in input_seq])
input_seq = np.array(pad_sequences(input_seq, max_len, padding='pre'))

X = input_seq[:,:-1]
Y = input_seq[:,-1]
Y = np.array(tf.keras.utils.to_categorical(Y,num_classes=len(vocab)))

model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=emb_size, input_length=max_len-1))
# model.add(LSTM(128))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(vocab), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=128, shuffle=False)
model_path = f'lyrics_model.h5'
model.save(model_path)








