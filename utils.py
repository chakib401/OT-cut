import torchvision
from scipy import io
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torchvision.datasets as td


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def CA(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def CF1(y_true, y_pred, **kwargs):
    def cmat_to_psuedo_y_true_and_y_pred(cmat):
        y_true = []
        y_pred = []
        for true_class, row in enumerate(cmat):
            for pred_class, elm in enumerate(row):
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
        return y_true, y_pred

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, average='macro',
                            **kwargs)


# create graphs on image datasets
def acm():
    dataset = "data/ACM"
    data = io.loadmat('{}.mat'.format(dataset))

    A = data['PLP']

    labels = data['label']
    labels = labels.reshape(-1)

    return A, labels


def dblp():
    dataset = "data/DBLP"
    data = io.loadmat('{}.mat'.format(dataset))

    A = data['net_APTPA']

    labels = data['label']
    labels = labels.reshape(-1)

    return A, labels


def get_dataset(dname):
    download = True
    train = False
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])
    datasets = {
        'mnist': td.MNIST('data', download=download, train=train),
        'fmnist': td.FashionMNIST('data', download=download, train=train),
        'kmnist': td.KMNIST('data', download=download, train=train),
        'cifar10': td.CIFAR10('data', transform=transform, download=download, train=train),
        'cifar100': td.CIFAR10('data', transform=transform, download=download, train=train)
    }
    data = datasets[dname]

    if dname in ['cifar10']:
        x = data.data
        x = np.dot(x, [0.299, 0.587, 0.114])
    else:
        x = np.array(data.data)

    labels = np.array(data.targets)

    x = x.reshape(len(x), -1)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return x, labels
