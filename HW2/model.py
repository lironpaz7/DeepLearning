import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size, vocab):
        super(RNN, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, input_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word, hidden):
        embeds = self.word_embeddings(word)
        combined = torch.cat((embeds.view(1, -1), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def get_vocab(self):
        return self.vocab


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, batch_size, vocab):
        super().__init__()
        self.batch_size = batch_size
        self.vocab = vocab  # save the vocabulary from the train so we can use it on test

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embs, hidden)

        # Dropout is applied to the output and fed to the FC layer
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch=None):
        if batch:
            return torch.zeros(1, batch, 32), torch.zeros(1, batch, 32)
        return torch.zeros(1, self.batch_size, 32), torch.zeros(1, self.batch_size, 32)
