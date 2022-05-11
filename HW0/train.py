import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import time


def Sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def Sigmoid_Prime(s):
    return s * (1 - s)


def ReLU(t):
    return torch.max(t, torch.zeros_like(t))


def ReLU_Prime(t):
    return torch.where(t <= 0, torch.tensor(0), torch.tensor(1))


def LeakyReLU(t):
    return torch.max(t, 0.01 * t)


def LeakyReLU_Prime(t):
    return torch.where(t <= 0.0, torch.tensor(0.01), torch.tensor(1.0))


def softmax_no_overflow(x):
    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    exp = torch.exp(x - torch.max(x, 1).values[:, None])
    row_sum = torch.sum(exp, 1)
    return exp / row_sum[:, None]


def convert_label_to_one_hot(y):
    tensor = torch.zeros(y.size(0), 10)
    for index, value in enumerate(y):
        tensor[index][value] = 1
    return tensor


def calc_loss_and_accuracy(y_pred, y):
    true_predictions = float(torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)))
    y_pred[y_pred == 0] = 1e-12
    loss = - torch.mean(torch.sum(y * torch.log(y_pred), dim=1))
    return loss, true_predictions / y.size(0)


def graph_plot(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list):
    indices_list = [(1 + i) for i in range(len(train_accuracy_list))]
    plt.plot(indices_list, train_accuracy_list, '-', c="tab:blue", label='Train accuracy')
    plt.plot(indices_list, test_accuracy_list, '-', c="tab:orange", label='Validation accuracy')

    plt.plot(indices_list, train_accuracy_list, 'o', color='tab:blue', markersize=4)
    plt.plot(indices_list, test_accuracy_list, 'o', color='tab:orange', markersize=4)
    plt.xticks(np.arange(1, len(indices_list) + 1, step=1))
    plt.grid(linewidth=1)
    plt.title(f'Train and Validation accuracies along epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.show()


class Neural_Network:
    def __init__(self, input_size=784, hidden_size=100, output_size=10, activation_func='Sigmoid'):
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # Activation function
        if activation_func == 'Sigmoid':
            self.activation_func, self.activation_func_derivative = Sigmoid, Sigmoid_Prime
        elif activation_func == 'ReLU':
            self.activation_func, self.activation_func_derivative = ReLU, ReLU_Prime
        elif activation_func == 'LeakyReLU':
            self.activation_func, self.activation_func_derivative = LeakyReLU, LeakyReLU_Prime
        else:
            raise ValueError("Unknown activation function. choose from {'Sigmoid', 'ReLU', 'LeakyReLU'}")

        # weights initialization
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 784xhidden_size
        self.b1 = torch.zeros(self.hiddenSize)  # same size as x^T*W1 =  hidden_sizex1

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # hidden_sizeX10
        self.b2 = torch.zeros(self.outputSize)  # 10x1

    def forward(self, X):  # X is 1 batch: batch_sizex784
        self.z1 = torch.matmul(X,
                               self.W1) + self.b1  # batch_sizex784 * 784xhidden_size + 1xhidden_size= batch_sizexhidden_size + 1xhidden_size =  batch_sizexhidden_size (row-wise addition)
        self.h = self.activation_func(self.z1)  # batch_sizexhidden_size
        self.z2 = torch.matmul(self.h,
                               self.W2) + self.b2  # batch_sizexhidden_size * hidden_sizeX10 + 1x10 = batch_sizex10 + 1x10 = batch_sizex10 (row-wise addition)
        return softmax_no_overflow(self.z2)  # batch_sizex10

    def backward(self, X, y, y_hat, lr=.1):
        dl_dz2 = torch.zeros(y.size(0), y.size(1), dtype=torch.float32)
        for sample in range(dl_dz2.size(0)):  # batch_size
            for i in range(dl_dz2.size(1)):  # 10
                if y[sample][i] == 1:  # y[sample] = [0,0,0,0,1,0,0,0,0,0]
                    for k in range(dl_dz2.size(1)):  # 10
                        if k == i:
                            dl_dz2[sample][k] = y_hat[sample][k] - 1
                        else:
                            dl_dz2[sample][k] = y_hat[sample][k]

        dl_dW2 = torch.matmul(torch.t(self.h), dl_dz2)
        dl_db2 = torch.matmul(torch.t(dl_dz2), torch.ones(y.size(0)))
        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * self.activation_func_derivative(self.h)
        dl_dW1 = torch.matmul(torch.t(X), dl_dz1)
        dl_db1 = torch.matmul(torch.t(dl_dz1), torch.ones(y.size(0)))

        # update the weight (GD alg.) with the grads
        self.W1 -= lr * dl_dW1
        self.b1 -= lr * dl_db1
        self.W2 -= lr * dl_dW2
        self.b2 -= lr * dl_db2

    def train(self, X, y):
        y_pred = self.forward(X)  # y_pred is batch_sizex10
        self.backward(X, y, y_pred)  # the backward need to know also X
        return calc_loss_and_accuracy(y_pred, y)


def main():
    # hyperparameters
    batch_size = 32
    hidden_size = 100
    num_epochs = 10
    # learning_rate = 0.1

    # define our data table
    table = PrettyTable()
    table.field_names = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']

    transform = transforms.Compose([transforms.ToTensor()])  # Data is already normalized between [0,1]
    train_and_val_dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [50000, 10000])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # choose weights according to evaluation on valudation set
    train_losses, train_accuracies, val_losses, val_accuracies = list(), list(), list(), list()
    NN = Neural_Network(input_size=784, hidden_size=hidden_size, output_size=10, activation_func='Sigmoid')
    for epoch in range(num_epochs):
        # train model on train set (and evaluate on it)
        train_epoch_loss, train_epoch_accuracy = [], []
        for i, (x, y) in enumerate(train_loader):
            x = x.view(-1, 28 * 28)  # batch_size x 784 -- flatten every image 28*28 to 784 (Grayscale)
            y = convert_label_to_one_hot(y)
            loss, accuracy = NN.train(x, y)
            train_epoch_loss.append(float(loss))
            train_epoch_accuracy.append(float(accuracy))

        train_losses.append(np.mean(train_epoch_loss))
        train_accuracies.append(np.mean(train_epoch_accuracy))

        # evaluate on val set
        val_epoch_loss, val_epoch_accuracy = list(), list()
        for i, (x, y) in enumerate(val_loader):
            x = x.view(-1, 28 * 28)  # batch_size x 784 -- flatten every image 28*28 to 784 (Grayscale)
            y = convert_label_to_one_hot(y)
            y_pred = NN.forward(x)
            loss, accuracy = calc_loss_and_accuracy(y_pred, y)
            val_epoch_loss.append(float(loss))
            val_epoch_accuracy.append(float(accuracy))

        table.add_row([epoch, round(np.mean(train_epoch_loss), 6), round(np.mean(train_epoch_accuracy), 6),
                       round(np.mean(val_epoch_loss), 6), round(np.mean(val_epoch_accuracy), 6)])

        val_losses.append(np.mean(val_epoch_loss))
        val_accuracies.append(np.mean(val_epoch_accuracy))

        torch.save({"W1": NN.W1, "W2": NN.W2, "b1": NN.b1, "b2": NN.b2}, f'weights_epoch{epoch}.pkl')

    print(table)
    graph_plot(train_accuracies, train_losses, val_accuracies, val_losses)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}')
