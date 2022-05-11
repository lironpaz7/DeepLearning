import time
from predict import eval
from sklearn import metrics
from model import CNN
import torch
import torch.nn as nn
from utils import *


def train(dataset, model, loader, test_loader, criterion, optimizer, num_epochs=5, batch_size=100):
    """
    Trains the model and calculates loss, accuracy, f1 and Roc-Auc scores
    :param dataset: Train dataset
    :param model: Model object - for example: CNN
    :param loader: Train DataLoader object
    :param test_loader: Test DataLoader object
    :param criterion: Loss criterion
    :param optimizer: Optimizer object - for example: torch.optim.Adam()
    :param num_epochs: Number of epochs - default: 5
    :param batch_size: Size of the batch - default: 100
    """
    loss_lst, roc_auc_lst, f1_lst, acc_lst = [], [], [], []
    test_loss_lst, test_roc_auc_lst, test_f1_lst, test_acc_lst = [], [], [], []

    for epoch in range(num_epochs):
        loss, roc_auc, f1, acc = 0, 0, 0, 0
        start = time.time()
        for i, (images, labels) in enumerate(loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.topk(1, dim=1)
            predicted = predicted.view(labels.shape[0])
            y_prob = outputs[:, 1].cpu().detach().numpy()

            loss += loss.data
            roc_auc += metrics.roc_auc_score(labels.cpu(), y_prob)
            acc += metrics.accuracy_score(labels.cpu(), predicted.cpu().numpy())
            f1 += metrics.f1_score(labels.cpu(), predicted.cpu().numpy())

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Iter [{i + 1}/{len(dataset) // batch_size}] Loss: {loss.data: .4f} ')

        # calculate metrics
        f1_lst.append(f1 / len(loader))
        loss_lst.append(loss / len(loader))
        roc_auc_lst.append(roc_auc / len(loader))
        acc_lst.append(acc / len(loader))
        print(f'Epoch {epoch + 1} total time: {time.time() - start}')

        # eval on test
        test_loss, test_roc_auc, test_f1, test_acc = eval(model, test_loader, criterion, test=False)
        print(f'Test F1: {test_f1}, Test Acc: {test_acc}')
        test_loss_lst.append(test_loss)
        test_roc_auc_lst.append(test_roc_auc)
        test_f1_lst.append(test_f1)
        test_acc_lst.append(test_acc)

    print_stats(loss_lst, roc_auc_lst, f1_lst, acc_lst, test_loss_lst, test_roc_auc_lst, test_f1_lst, test_acc_lst,
                mode='Train')


if __name__ == '__main__':
    # Model Hyper Parameters
    num_epochs = NUM_EPOCHS
    batch_size = 100
    learning_rate = 0.001

    t = time.time()

    # stage 1: prep data
    print('Loading Data...')
    train_dataset, train_loader = prepare_data(path=TRAIN_IMG_PATH, batch_size=batch_size, mode='train')
    _, test_loader = prepare_data(path=TEST_IMG_PATH, batch_size=batch_size, mode='test')

    print('------------------- Stage 1 completed -------------------')

    # stage 2: build model & define CUDA device
    print('Building Model...')
    cnn = CNN()

    if torch.cuda.is_available():
        cnn = cnn.cuda()

    # convert all the weights tensors to cuda()
    # Loss and Optimizer

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    print('Number Of Parameters: ', sum(param.numel() for param in cnn.parameters()))
    print('------------------- Stage 2 completed -------------------')

    # stage 3: training
    print('Training Model...')
    train(train_dataset, cnn, train_loader, test_loader, criterion, optimizer, num_epochs, batch_size)
    print('------------------- Stage 3 completed -------------------')

    # stage 4: save model
    model_name = 'model.pkl'
    print(f'Saving {model_name}...')
    torch.save(cnn, model_name)
    print('------------------- Stage 4 completed -------------------')

    print('Finished!')
    print(f'Total Time Taken: {time.time() - t}')
