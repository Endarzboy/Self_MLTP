"""
Python code for applying Temporally Consistent Random Geometric Transformations on a video clips

By: Maregu Assefa
October, 2020
University of Electronic Science and Technology of China, ChengDu, China
School: School of Software Engineering
2nd Year Ph.D Student.
Research Area: Computer Vision, Deep Learning (Multimedia Deep Learning).

"""
import os

import csv
import warnings
import itertools
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore', category=DeprecationWarning)


def save_csv(data, file_path, fieldnames=None):
    if fieldnames is None:
        fieldnames = ['frame_path', 'region', 'degree', 'colorChannel']
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


def save_loss_plot(train_loss, val_loss, filename):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train_loss')
    plt.plot(val_loss, color='red', label='valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)  # './figures/' + model + "_loss.jpg"
    plt.show()


def save_accuracy_plot(train_acc, val_acc, filename):
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='orange', label='train_acc')
    plt.plot(val_acc, color='red', label='valid_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename)  # './figures/' + model + "_loss.jpg"
    plt.show()


def load_checkpoint(model, name):
    print("Restoring checkpoint: {}".format(name))
    model.load_state_dict(torch.load(name))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def calculate_metrics(output, target):
    _, predicted_region = output['region'].cpu().max(1)  # ?
    gt_region = target['region_labels'].cpu()

    _, predicted_degree = output['degree'].cpu().max(1)
    gt_degree = target['degree_labels'].cpu()

    _, predicted_channel = output['channel'].cpu().max(1)
    gt_channel = target['channel_labels'].cpu()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc_region = balanced_accuracy_score(y_true=gt_region.numpy(), y_pred=predicted_region.numpy())
        acc_degree = balanced_accuracy_score(y_true=gt_degree.numpy(), y_pred=predicted_degree.numpy())
        acc_channel = balanced_accuracy_score(y_true=gt_channel.numpy(), y_pred=predicted_channel.numpy())

    return acc_region, acc_degree, acc_channel


def loss_func(outputs, targets):
    l1 = nn.CrossEntropyLoss()(outputs['region'], targets['region_labels'])
    l2 = nn.CrossEntropyLoss()(outputs['degree'], targets['degree_labels'])
    l3 = nn.CrossEntropyLoss()(outputs['channel'], targets['channel_labels'])

    loss = (l1 + l2 + l3) / 3
    return loss, dict(region_loss=l1, degree_loss=l2, channel_loss=l3)


def loss_func_single(outputs, targets):
    l1 = nn.CrossEntropyLoss()(outputs, targets)
    return l1


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# INPUTS: output have shape of [batch_size, category_count]
#    and target in the shape of [batch_size] * there is only one true class for each sample
# topk is tuple of classes to be included in the precision
# topk have to a tuple so if you are giving one number, do not forget the comma
def accuracy_(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # we do not need gradient calculation for those
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # we will use biggest k, and calculate all precisions from 0 to k
        # topk gives biggest maxk values on dimth dimension from output
        # output was [batch_size, category_count], dim=1 so we will select biggest category scores for each batch
        # input=maxk, so we will select maxk number of classes
        # so result will be [batch_size,maxk]
        # topk returns a tuple (values, indexes) of results
        # we only need indexes(pred)
        _, pred = output.topk(input=maxk, dim=1, largest=True, sorted=True)
        # then we transpose pred to be in shape of [maxk, batch_size]
        pred = pred.t()
        # we flatten target and then expand target to be like pred target [batch_size] becomes [1,batch_size] target
        # [1,batch_size] expands to be [maxk, batch_size] by repeating same correct class answer maxk times. when you
        # compare pred (indexes) with expanded target, you get 'correct' matrix in the shape of  [maxk, batch_size]
        # filled with 1 and 0 for correct and wrong class assignments
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        """ correct=([[0, 0, 1,  ..., 0, 0, 0],
            [1, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 1, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 1, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.uint8) """
        res = []
        # then we look for each k summing 1s in the correct matrix for first k element.
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def plot_confusion_matrix(cm, classes, figname, normalize=False, title='Confusion matrix', cmap='Blues'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figname)
    return plt


def cm_as_dataframe(cm, class_names):
    cm = pd.DataFrame(cm)
    cm.columns = class_names
    cm.index = class_names
    cm = cm.reset_index()

    return cm


def loss_function(net_output, ground_truth):
    region_loss = F.cross_entropy(net_output['region'], ground_truth['region_labels'])
    degree_loss = F.cross_entropy(net_output['degree'], ground_truth['degree_labels'])
    channel_loss = F.cross_entropy(net_output['channel'], ground_truth['channel_labels'])

    loss = region_loss + degree_loss + channel_loss

    return loss, {'region_loss': region_loss, 'degree_loss': degree_loss, 'channel_loss': channel_loss}


# checkpoint = {'model': FashionClassifier(),
#               'state_dict': model.state_dict(),
#               'optimizer': optimizer.state_dict()}
#
# torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint_(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):

        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print("INFO: Early stopping counter {} of {} patience.")
            if self.counter >= self.patience:
                print("INFO: Early stopping.")
                self.early_stop = True


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def save_checkpoint(args, model, name, epoch):
    f = os.path.join(name, '{}_checkpoint-{:06d}.pth'.format(args.choose_model, epoch))
    torch.save(model.state_dict(), f)
    print('Checkpoint Saved as: ', f)
