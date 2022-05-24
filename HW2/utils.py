import re

import numpy as np
import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
label_dict = {'happiness': 2,
              'neutral': 1,
              'sadness': 0
              }


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def label_mapper(x):
    return label_dict[x]


def load_data(path):
    df = pd.read_csv(path)
    print(f'Loaded {len(df)} tweets')
    return df


def tokenizer(sentence):
    porter = PorterStemmer()
    sentence = clean_tweets(sentence)
    tokens = sentence.split(" ")
    tokens = [porter.stem(token.lower()) for token in tokens if not token.lower() in stop_words]
    return tokens


def map_word_vocab(sentence, vocab):
    idxs = [vocab[w] for w in sentence]
    return torch.tensor(idxs, dtype=torch.long)


def map_class(sentiment):
    return torch.tensor([label_dict[sentiment]], dtype=torch.long)


def prepare_sequence(sentence, vocab):
    # create the input feature vector
    input = map_word_vocab(sentence, vocab)
    return input


def clean_tweets(text):
    text_cleaning_regex = "@S+|https?:S+|http?:S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_regex, ' ', str(text).lower()).strip()
    return text
    # return " ".join(tokens)


def preprocess(df):
    df.content = df.content.apply(lambda x: tokenizer(x))
    return df
