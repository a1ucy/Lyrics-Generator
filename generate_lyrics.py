import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

model = load_model('lyrics_model.h5')
vocab = torch.load('vocab.pth')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predict(input_text):
    max_text = 100
    pred = input_text
    while max_text > 0:
        lower_text = pred.lower()
        tokens = lower_text.split(" ")
        token_lis = vocab(tokens)
        token_lis = pad_sequences([token_lis], maxlen=100-1, padding='pre')

        predict = model.predict(token_lis)[0]
        # choices = np.argpartition(predict, -10)[-10:]
        # random_choice = random.choice(choices)

        next_index = sample(predict, 1.5)

        word = vocab.lookup_token(next_index)

        if word == "'newline":
            continue
        elif word == tokens[-1]:
            continue
        else:
            pred += " " + word
        max_text -= 1
    return pred

print(predict("You used to call me on my"))
