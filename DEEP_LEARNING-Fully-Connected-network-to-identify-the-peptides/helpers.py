import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import Subset
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
OVERSAMPLING = "oversampling"
UNDERSAMPLING = "undersampling"
BATCH_SIZE = 24
amino_binary = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
                 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

max_epochs = 50
training_rate = 0.9
learning_rate = 0.0002
sars_optimizations = 100


class DataSet(torch.utils.data.Dataset):
    def __init__(self, aminos_data):
        self.aminos_id = aminos_data

    def __len__(self):
        return len(self.aminos_id)

    def __getitem__(self, item):
        X = torch.from_numpy(self.aminos_id[item][0])
        y = torch.zeros(1) + self.aminos_id[item][1][0]
        return X, y.int()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(180, 80)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(80, 24)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(24, 18)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(18, 1)
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x.float())
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        return x


def line_to_binary(line, label):
    """
    represent 9 amino bases to "one-hot" representation
    :param line: 9 bases to translate
    :param label: '1' for pos, '0' for neg
    :return: 1X180 "one-hot" vector
    """
    parsed_line = np.zeros((9, 20))
    for i, amino in enumerate(line):
        parsed_line[i][amino_binary[amino.upper()]] = 1
    return np.array([parsed_line.flatten(), np.array([label])], dtype=object)


def get_training_data(sampling_type=OVERSAMPLING):
    """
    Extracts the pos and neg data from the given files to num representation
    :param sampling_type: oversampling/undersamplimg for handling biased data
    :return: the pos and neg data translated to num in a matrix
    """
    data_path = os.path.curdir
    neg_data_file_name = os.path.join(data_path, 'data', 'neg_A0201.txt')
    pos_data_file_name = os.path.join(data_path, 'data', 'pos_A0201.txt')

    with open(neg_data_file_name) as f:
        neg_data = np.array([line_to_binary(s, 0) for s in f.read().splitlines()])

    with open(pos_data_file_name) as f:
        pos_data = np.array([line_to_binary(s, 1) for s in f.read().splitlines()])

    # oversample pos data to deal with imbalance data
    if sampling_type == OVERSAMPLING:
        random_pos_sample = pos_data[: 1]
        while random_pos_sample.shape[0] + pos_data.shape[0] < neg_data.shape[0]:
            # sample 300 sequences in each iteration
            indexes = np.random.choice(pos_data.shape[0], size=300)
            random_pos_sample = np.vstack([random_pos_sample, pos_data[indexes]])
        pos_data = np.vstack([pos_data, random_pos_sample])
    else:
        indexes = np.random.choice(neg_data.shape[0], size=pos_data.shape[0])
        neg_data = neg_data[indexes]

    return np.vstack([pos_data, neg_data])


def get_generators(sampling_type=OVERSAMPLING):
    """
    creates data generator for train and test
    :param sampling_type: oversampling/undersampling to deal with biased data
    :return: train and test data generators
    """
    # divide the data into train&test
    dataset = get_training_data(sampling_type)
    test_size = round((1 - training_rate) * dataset.shape[0])
    train_size = dataset.shape[0] - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # create our data
    # Generators
    training_set = DataSet(train_data)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = DataSet(test_data)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return training_generator, test_generator

def calc_accuracy(expected, actual):
    """
    accuracy calculator: (true predictions) / (total predictions)
    :param expected: the result of the network (rounded)
    :param actual: actual labels
    :return: the calculated accuracy
    """
    acc = expected == actual
    acc = acc.sum()
    return acc.item() / expected.size(0)


def run_epoch(net, data_generator, optimizer, criterion, is_training=True):
    """
    runs an epoch
    :param net: nn
    :param data_generator: training ot testing
    :param optimizer: the defined optimaizer (e.g SGD)
    :param criterion: the loss function
    :param is_training: id true - train the network
    :return: the loss and the accuracy of this epoch
    """
    counter, loss_avg, accuracy_avg = 0, 0, 0
    with torch.set_grad_enabled(is_training):
        for local_batch, local_labels in data_generator:
            # Transfer to GPU
            inputs, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            if is_training:
                optimizer.zero_grad()

            outputs = net(inputs).float()
            loss = criterion(outputs, local_labels.float())

            if is_training:
                loss.backward()
                optimizer.step()
            loss_avg += loss.item()
            accuracy_avg += calc_accuracy(outputs.round(), local_labels)
            counter += 1
    return loss_avg / counter, accuracy_avg / counter


def get_matrices(net, test_generator):
    """
    extracts the total predictions and the labels of the testing data
    :param net: nn
    :param test_generator: the test data generator
    :return: predictions and the labels of the testing data
    """
    y_pred, y_labels = np.array([[0]]), np.array([[0]])
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            y_pred = np.vstack([y_pred, net(local_batch).round().numpy()])
            y_labels = np.vstack([y_labels, local_labels])
    return y_pred, y_labels


def get_sars():
    """
    extract the spike sars data and translate it to "one-hot" representation
    :return: spike sars data translated
    """
    data_path = os.path.curdir
    spike_file_name = os.path.join(data_path, 'data', 'P0DTC2.txt')
    with open(spike_file_name) as f:
        spike_parsed = f.read().splitlines()[0]
        spike_parsed = [spike_parsed[i: i+9] for i in range(len(spike_parsed) - 8)]
        spike_data = np.array([line_to_binary(s, 0)[0] for s in spike_parsed])
    return spike_parsed, torch.from_numpy(spike_data)


def plot_epoch_graph(train_data, test_data, data_type: str):
    """
    plots the epoch graph
    :param train_data: loss or accuracy of the training
    :param test_data: loss or accuracy of the testing
    :param data_type: loss or accuracy (string)
    :return: nothing, just plot
    """
    plt.xlabel('Epoch')
    plt.ylabel(data_type)
    plt.plot(train_data, label=f'train {data_type.lower()}')
    plt.plot(test_data, label=f'test {data_type.lower()}')
    plt.legend()
    plt.show()


def plot_results(train_loss, test_loss, train_acc, test_acc, test_pred, test_labels):
    """
    plots the results
    :param train_loss:
    :param test_loss:
    :param train_acc:
    :param test_acc:
    :param test_pred:
    :param test_labels:
    :return:
    """
    plot_epoch_graph(train_loss, test_loss, "Loss")
    plot_epoch_graph(train_acc, test_acc, "Accuracy")

    # plot confusion matrix
    confusion_m = confusion_matrix(test_labels, test_pred)
    plt.subplot()
    sns.heatmap(confusion_m, annot=True, fmt="g")
    plt.title('Confusion matrix')
    plt.ylabel('True labels')
    plt.xlabel('Prediction labels')
    plt.show()


def optimize_sequence(net):
    """
    Find the most detectable peptides according to your trained network by optimising the networks output score with
    respect to the input sequence.
    prints 5 most detectable peptides in the Spike sars protein
    :param net: nn
    :param optimizations_num: number of optimizations
    :return: nothing
    """
    spike_txt, sars_tensor = get_sars()
    net.requires_grad_(False)
    weights = torch.nn.Parameter(sars_tensor, requires_grad=True)
    optimizer = torch.optim.Adamax([weights], lr=learning_rate)
    bce = nn.BCELoss()
    y = torch.ones(len(spike_txt))
    for i in range(sars_optimizations):
        out = net(weights).flatten()
        loss = bce(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    outputs = net(weights).detach().numpy().flatten()
    for idx in np.argsort(outputs)[-5:][::-1]:
        print(f'{spike_txt[idx]}: {outputs[idx]}')