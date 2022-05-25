import argparse
import time

import pandas as pd
import torch.cuda
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from predict import eval
from model import RNN, LSTM
from utils import *


def train(train_dl, test_dl, model, criterion, optimizer, num_epochs=30):
    if torch.cuda.is_available():
        model = model.cuda()

    # extract predicted labels for accuracy and confusion matrix
    sentiment_class = list(label_dict.keys())
    loss_lst, acc_lst = [], []
    acc_lst_tst, loss_lst_tst = [], []

    for epoch in range(1, num_epochs + 1):
        print(f'Running Epoch: [{epoch}/{num_epochs}]')
        h0, c0 = model.init_hidden()
        y_pred, y_actual = [], []
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        for batch_idx, batch in enumerate(train_dl):
            input, target = batch[0], batch[1]
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output, hidden = model(input, (h0, c0))

                # extract predicted labels
                _, preds = torch.max(output, 1)
                preds = preds.cpu().tolist()
                y_pred.extend(preds)
                y_actual.extend(target.tolist())

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        loss_lst.append(loss.item())
        acc = accuracy_score(y_actual, y_pred)
        acc_lst.append(acc)
        print(f'Epoch: [{epoch}/{num_epochs}] | loss: {loss.data: .4f}')
        print(f'Train Accuracy: {acc}')

        # eval on test
        test_acc, test_loss = eval(test_dl, model, criterion, train=True)
        acc_lst_tst.append(test_acc)
        loss_lst_tst.append(test_loss)
        print(f'Test Loss: {test_loss}, Test Acc: {test_acc}')
        print('---------------------------------------------------------')

    print(sentiment_class)
    y_actual_str = [reverse_label(x) for x in y_actual]
    y_pred_str = [reverse_label(x) for x in y_pred]
    print(confusion_matrix(y_actual_str, y_pred_str, labels=sentiment_class))
    print(accuracy_score(y_actual, y_pred))

    # outputs results for analysis
    pd.DataFrame({
        'acc': acc_lst,
        'loss': loss_lst
    }).to_csv('metrics_train.csv', index=False)

    pd.DataFrame({
        'pred': y_pred_str,
        'true': y_actual_str
    }).to_csv('train_pred.csv', index=False)

    # outputs results for analysis
    pd.DataFrame({
        'acc': acc_lst_tst,
        'loss': loss_lst_tst
    }).to_csv('metrics_test.csv', index=False)


if __name__ == '__main__':
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
    vocab = build_index(df)
    print(f'Vocabulary size: {len(vocab)}')
    print('------------------- Stage 3 completed -------------------')

    print('Encoding and padding tweets...')
    train_data = [(encode_and_pad(tweet, SEQ_LENGTH, vocab), label_mapper(label)) for tweet, label in
                  zip(df.content, df.emotion)]
    print('------------------- Stage 4 completed -------------------')

    print('Preparing dataset...')
    train_x = np.array([tweet for tweet, label in train_data])
    train_y = np.array([label for tweet, label in train_data])

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    # getting Test Data
    df2 = load_data('testEmotions.csv', stats=False)
    df2 = preprocess(df2, train=False)
    test_data = [(encode_and_pad(tweet, SEQ_LENGTH, vocab), label_mapper(label)) for tweet, label in
                 zip(df2.content, df2.emotion)]

    test_x = np.array([tweet for tweet, label in test_data])
    test_y = np.array([label for tweet, label in test_data])

    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

    print('------------------- Stage 5 completed -------------------')

    print('Building LSTM model and setting Adam optimizer...')
    # creating an instance of RNN
    # model = RNN(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(label_dict), vocab)
    model = LSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, BATCH_SIZE, vocab)

    # Setting the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Number Of Parameters: ', sum(param.numel() for param in model.parameters()))
    print('------------------- Stage 6 completed -------------------')

    # stage 5: training
    print('Training Model...')
    train(train_dl, test_dl, model, criterion, optimizer, NUM_EPOCHS)
    print('------------------- Stage 7 completed -------------------')

    # stage 6: save model
    model_name = 'model.pkl'
    print(f'Saving {model_name}...')
    torch.save(model, model_name)
    print('------------------- Stage 8 completed -------------------')

    print('Finished!')
    print(f'Total Time Taken: {time.time() - t}')
