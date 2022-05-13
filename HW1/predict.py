import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from utils import *
from sklearn import metrics


def eval(model, loader, criterion, test=True):
    """
    Evaluates the model and calculates loss, accuracy, f1 and Roc-Auc scores
    :param model: Model object - for example: CNN
    :param loader: DataLoader object
    :param criterion: Loss criterion
    :param test: When True, this function will print the scores and return a pandas.DataFrame object. When False,
    this function will return all the scores as a tuple: (loss, roc_auc, f1, accuracy)
    :return: DataFrame or scores tuple
    """
    with torch.no_grad():
        # model.eval()
        loss_lst, roc_auc_lst, f1_lst, acc_lst = [], [], [], []
        loss, roc_auc, f1, acc = 0, 0, 0, 0
        all_labels = []
        # true_labels = []
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze())
            _, predicted = outputs.topk(1, dim=1)
            predicted = predicted.view(labels.shape[0])
            y_prob = outputs[:, 1].cpu().detach().numpy()

            p = list(predicted.cpu().numpy())
            all_labels.extend(p)
            # true_labels.extend(list(labels.cpu().numpy()))
            # calculate metrics
            loss += loss.data
            roc_auc += metrics.roc_auc_score(labels.cpu(), y_prob)
            acc += metrics.accuracy_score(labels.cpu(), predicted.cpu().numpy())
            f1 += metrics.f1_score(labels.cpu(), predicted.cpu().numpy())

        f1_lst.append(f1 / len(loader))
        loss_lst.append(loss / len(loader))
        roc_auc_lst.append(roc_auc / len(loader))
        acc_lst.append(acc / len(loader))

        if test:
            print_stats(loss_lst, roc_auc_lst, f1_lst, acc_lst, mode='Test')

            # write to CSV
            filenames = [x.split('_')[0] for x in loader.dataset.images]
            all_labels = [x.item() for x in all_labels]
            # true_labels = [x.item() for x in true_labels] # add to df in order to calculates ROC-AUC easilty in
            # a jupyter notebook later
            df = pd.DataFrame({'filenames': filenames,
                               'labels': all_labels})
            return df
        else:
            return loss_lst[-1], roc_auc_lst[-1], f1_lst[-1], acc_lst[-1]


if __name__ == '__main__':
    t = time.time()
    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    # Reading input folder
    files = os.listdir(args.input_folder)

    # stage 1: prep data
    print('Loading Data...')
    _, test_loader = prepare_data(path=args.input_folder, mode='test')
    print('------------------- Stage 1 completed -------------------')

    # stage 2: load model
    print('Loading Model...')
    model = torch.load('model.pkl')
    print('------------------- Stage 2 completed -------------------')

    # stage 3: evaluating
    print('Evaluating...')
    criterion = nn.NLLLoss()
    prediction_df = eval(model, test_loader, criterion, test=True)
    print('------------------- Stage 3 completed -------------------')

    # stage 4: saving csv
    print('Exporting to prediction.csv...')
    prediction_df.to_csv("prediction.csv", index=True, header=False)
    print('------------------- Stage 4 completed -------------------')

    print('Finished!')
    print(f'Total Time Taken: {time.time() - t}')
