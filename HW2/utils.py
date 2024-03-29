import re

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Hyper-Parameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
DROPOUT = 0.1
NUM_EPOCHS = 30
BATCH_SIZE = 50
SEQ_LENGTH = 32
LEARNING_RATE = 0.001

stop_words = set(stopwords.words('english'))
label_dict = {'happiness': 2,
              'neutral': 1,
              'sadness': 0
              }


def reverse_label(label):
    if label == 0:
        return 'sadness'
    elif label == 1:
        return 'neutral'
    else:
        return 'happiness'


def label_mapper(x):
    return label_dict[x]


def load_data(path, stats=True):
    df = pd.read_csv(path)
    if stats:
        print(f'Loaded {len(df)} tweets')
    return df


def tokenizer(sentence):
    porter = PorterStemmer()
    sentence = clean_tweets(sentence)
    tokens = sentence.split(" ")
    tokens = [porter.stem(token.lower()) for token in tokens if not token.lower() in stop_words]
    return tokens


def map_class(sentiment):
    return torch.tensor([label_dict[sentiment]], dtype=torch.long)


def clean_tweets(text):
    link_re_pattern = "https?:\/\/t.co/[\w]+"
    tags = r'@[^\s]+'
    mention_re_pattern = "@\w+"
    text = re.sub(link_re_pattern, "", text)
    text = re.sub(mention_re_pattern, "", text)
    text = re.sub(tags, '', text)
    return text.lower()


def encode_and_pad(tweet, length, index):
    sos = [index["<SOS>"]]
    eos = [index["<EOS>"]]
    pad = [index["<PAD>"]]

    if len(tweet) < length - 2:  # -2 for SOS and EOS
        n_pads = length - 2 - len(tweet)
        encoded = []
        for w in tweet:
            if w not in index:
                index[w] = 0
            encoded.append(index[w])
        # encoded = [index[w] for w in tweet]
        return sos + encoded + eos + pad * n_pads
    else:  # tweet is longer than possible; truncating
        encoded = []
        for w in tweet:
            if w not in index:
                index[w] = 0
            encoded.append(index[w])
        # encoded = [index[w] for w in tweet]
        truncated = encoded[:length - 2]
        return sos + truncated + eos


def build_index(df):
    index2word = ["<PAD>", "<SOS>", "<EOS>"]
    for tokens in df.content:
        for token in tokens:
            if token not in index2word:
                index2word.append(token)

    word2index = {token: idx for idx, token in enumerate(index2word)}
    return word2index


def preprocess(df, train=True):
    if train:
        df = df.dropna()
    df.content = df.content.apply(lambda x: tokenizer(x))
    return df
