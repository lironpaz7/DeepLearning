import pandas as pd
import time
import torch
from dataset import MasksDataset

# CONSTANTS
TRAIN_IMG_PATH = '/home/student/HW1/train'
TEST_IMG_PATH = '/home/student/HW1/test'
NUM_EPOCHS = 30


def prepare_data(path=TRAIN_IMG_PATH, batch_size=100, mode='train'):
    """
    Reads folder of images and returns MasksDataset object and DataLoader object
    :param path: Image folder path
    :param batch_size: Size of the batch - default: 100
    :param mode: Can be 'train' or 'test'
    :return: MasksDataset object and DataLoader object of the images
    """
    assert mode in {'train', 'test'}

    if mode == 'train':
        train_dataset = MasksDataset(data_folder=path, split=mode)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        return train_dataset, train_loader
    else:
        test_dataset = MasksDataset(data_folder=path, split=mode)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return test_dataset, test_loader


def print_stats(loss_lst, roc_auc_lst, f1_lst, acc_lst, test_loss_lst=None, test_roc_auc_lst=None, test_f1_lst=None,
                test_acc_lst=None, num_epochs=NUM_EPOCHS, mode='Test'):
    """
    Prints model statistics and exports to a csv
    :param loss_lst: list of train loss values
    :param roc_auc_lst: list of train roc_auc values
    :param f1_lst: list of train f1 values
    :param acc_lst: list of train accuracy values
    :param test_loss_lst: list of test loss values
    :param test_roc_auc_lst: list of test roc_auc values
    :param test_f1_lst: list of test f1 values
    :param test_acc_lst: list of test accuracy values
    :param num_epochs: number of epochs for the model
    :param mode: 'Train' or 'Test'
    """
    acc = acc_lst[-1]
    loss = loss_lst[-1]
    roc_auc = roc_auc_lst[-1]
    f1_score = f1_lst[-1]

    print(f'{mode} Accuracy: {acc}')
    print(f'{mode} Loss: {loss}')
    print(f'{mode} Roc-Auc: {roc_auc}')
    print(f'{mode} F1 Score: {f1_score}')
    if mode == 'Test':
        return
    d = {'acc': acc_lst,
         'loss': loss_lst,
         'roc_auc': roc_auc_lst,
         'f1': f1_lst,
         'epochs': range(1, num_epochs + 1)
         }
    if test_acc_lst:
        p = {'test_acc': test_acc_lst,
             'test_loss': test_loss_lst,
             'test_roc_auc': test_roc_auc_lst,
             'test_f1': test_f1_lst,
             }
        d.update(p)

    # writes output.csv
    pd.DataFrame(d).to_csv(f'plots.csv')
