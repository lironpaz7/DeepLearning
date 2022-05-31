import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils import *


def eval(dl, model, criterion, train=False):
    """
    Predicting labels using the given model on the given data set
    :param dl: Data loader object
    :param model: Model object
    :param criterion: Loss criterion
    :param train: If train is set to True then this method will return the loss and accuracy. Otherwise, prints the
    confusion
    :return:
    """
    # Converts the sentiment_class from set to list
    sentiment_class = list(label_dict.keys())
    loss_lst, acc_lst = [], []
    y_pred, y_actual = [], []

    # predict
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), ascii=False, desc='Predicting...', ncols=100):
            input, target = batch[0], batch[1]
            h0, c0 = model.init_hidden(len(input))
            if torch.cuda.is_available():
                h0 = h0.cuda()
                c0 = c0.cuda()
                input = input.cuda()
                target = target.cuda()

            output, hidden = model(input, (h0, c0))
            loss = criterion(output, target)

            # extract predicted labels
            _, preds = torch.max(output, 1)
            preds = preds.cpu().tolist()
            y_pred.extend(preds)
            y_actual.extend(target.tolist())

            acc_lst.append(accuracy_score(y_actual, y_pred))
            loss_lst.append(loss.item())

    if train:
        return np.mean(acc_lst), np.mean(loss_lst)
    else:
        print(sentiment_class)
        y_actual_str = [reverse_label(x) for x in y_actual]
        y_pred_str = [reverse_label(x) for x in y_pred]
        print(confusion_matrix(y_actual_str, y_pred_str, labels=sentiment_class))
        print(f'Test accuracy: {accuracy_score(y_actual, y_pred)}')
        return y_actual_str, y_pred_str


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
    df = preprocess(df, train=False)
    print('------------------- Stage 2 completed -------------------')

    print('Loading Model...')
    model_name = 'model_epoch_9.pkl'
    model = torch.load(model_name)
    print('------------------- Stage 3 completed -------------------')

    print('Encoding and padding tweets...')
    test_data = [(encode_and_pad(tweet, SEQ_LENGTH, model.vocab), label_mapper(label)) for tweet, label in
                 zip(df.content, df.emotion)]
    print('------------------- Stage 4 completed -------------------')

    print('Preparing dataset...')
    test_x = np.array([tweet for tweet, label in test_data])
    test_y = np.array([label for tweet, label in test_data])

    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    print('------------------- Stage 5 completed -------------------')

    print('Evaluating...')
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred = eval(test_dl, model, criterion, train=False)
    print('------------------- Stage 6 completed -------------------')

    print('Exporting to prediction.csv...')
    d = load_data(args.input_file, stats=False)

    prediction_df = pd.DataFrame(
        {'emotion': y_pred,
         'content': d.content
         }
    )
    prediction_df.to_csv("prediction.csv", index=False, header=True)

    metrics = True
    if metrics:
        pd.DataFrame(
            {'pred': y_pred,
             'true': y_true
             }).to_csv('test_pred.csv')
    print('Finished!')
    print(f'Total Time Taken: {time.time() - t}')
