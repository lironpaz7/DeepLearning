import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import numpy as np
import train as train_module


def evaluate_hw0():
    # hyperparameters
    batch_size = 32
    hidden_size = 100
    num_epochs = 10
    # learning_rate = 0.1

    transform = transforms.Compose([transforms.ToTensor()])  # Data is already normalized between [0,1]
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # After look at the epochs performance on the validation set, we choose weights for the test set
    choosed_epoch = 9
    pkl_weights = torch.load(f'weights_epoch{choosed_epoch}.pkl')
    NN = train_module.Neural_Network(input_size=784, hidden_size=hidden_size, output_size=10, activation_func='Sigmoid')
    NN.W1, NN.W2, NN.b1, NN.b2 = pkl_weights.values()

    # evaluate on test set
    test_loss, test_accuracy = [], []
    for i, (x, y) in enumerate(test_loader):
        x = x.view(-1, 28 * 28)
        y = train_module.convert_label_to_one_hot(y)
        y_pred = NN.forward(x)
        loss, accuracy = train_module.calc_loss_and_accuracy(y_pred, y)
        test_loss.append(float(loss))
        test_accuracy.append(float(accuracy))
    print(f'Test Loss: {round(np.mean(test_loss), 6)},\t Test Accuracy: {round(np.mean(test_accuracy), 6)}')


if __name__ == "__main__":
    evaluate_hw0()
