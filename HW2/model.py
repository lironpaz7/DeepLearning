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
