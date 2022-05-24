import argparse
import time
# from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim, autograd

from model import RNN
from utils import *
# from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


def train(df, model, criterion, optimizer, num_epochs=30):
    if torch.cuda.is_available():
        model = model.cuda()

    # build a list of tweets & labels
    train_tweets = df.content.tolist()
    train_labels = df.emotion.tolist()
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, num_epochs + 1):
        loss_lst = []
        print(f'Running epoch: [{epoch}/{num_epochs}]')
        for i in range(len(train_tweets)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            hidden = model.init_hidden()
            optimizer.zero_grad()
            # Step 2. Get our inputs ready for the network, that is, turn them nto
            # Tensors of word indices.
            sentence, label = train_tweets[i], train_labels[i]
            sentence_in = prepare_sequence(sentence, model.vocab)
            target_label = map_class(label)

            if torch.cuda.is_available():
                hidden = hidden.cuda()
                sentence_in = sentence_in.cuda()
                target_label = target_label.cuda()

            # Step 3. Run our forward pass.
            for w in sentence_in:
                output, hidden = model(w, hidden)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(output, target_label)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_lst.append(loss.data)

        print(f'epoch: [{epoch}/{num_epochs}] | loss: {sum(loss_lst) / len(loss_lst): .4f}')


if __name__ == '__main__':
    # Hyper-Parameters
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 10
    num_epochs = 30

    t = time.time()
    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_file', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    print('Loading Data...')
    df = load_data(args.input_file)
    print('------------------- Stage 1 completed -------------------')

    print('Preprocessing...')
    df = preprocess(df)
    print('------------------- Stage 2 completed -------------------')

    print('Building vocabulary...')
    vocab, vocab_index = {}, 0
    for tokens in df.content.values:
        for key, token in enumerate(tokens):
            if token not in vocab:
                vocab[token] = vocab_index
                vocab_index += 1
    print('------------------- Stage 3 completed -------------------')

    print('Building model and setting SGD optimizer...')
    # creating an instance of RNN
    rnn = RNN(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(label_dict), vocab)

    # Setting the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)
    print('Number Of Parameters: ', sum(param.numel() for param in rnn.parameters()))
    print('------------------- Stage 4 completed -------------------')

    # stage 5: training
    print('Training Model...')
    train(df, rnn, criterion, optimizer, num_epochs)
    print('------------------- Stage 5 completed -------------------')

    # stage 6: save model
    model_name = 'model.pkl'
    print(f'Saving {model_name}...')
    torch.save(rnn, model_name)
    print('------------------- Stage 6 completed -------------------')

    print('Finished!')
    print(f'Total Time Taken: {time.time() - t}')
