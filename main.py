
import os
import cv2
import glob
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image
from torch import nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

from models import r3d, c3d, r21d
from dataset.ucf101dataset import *

# Here is an auxiliary function for checkpoint saving.
def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, '{}_checkpoint_{:06d}.pth'.format(model.__class__.__name__, epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()

    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)

def compute_roc(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()

    ROC = []
    for i in range(y_true.shape[0]):
        ROC.append(roc_auc_score(y_true[i], y_pred[i]))
    return np.mean(ROC)

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def train(model, trainloader, optimizer, criterion, epoch, device):
    # print("Training ...")
    model.train()

    counter = 0 
    train_running_loss = 0.0
    train_running_mAP = 0.0
    train_running_ROC = 0.0

    for i, dict_sample in tqdm(enumerate(trainloader), desc='Loading Trian batches: ', total=int(len(trainloader.dataset) / trainloader.batch_size)):
        counter += 1

        imgs, targets = dict_sample['frames'].to(device), dict_sample['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(imgs) # sigmoid output
        loss = criterion(outputs, targets.type(torch.float))

        train_running_loss += loss.item()
        train_running_mAP += compute_mAP(outputs, targets)
        train_running_ROC += compute_roc(outputs, targets)

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_mAP = train_running_mAP / counter
    train_ROC = train_running_ROC / counter

    print("Epoch: {} -- Train Loss: {:.4f} -- Train mAP: {:.4f} -- Train ROC: {:.4f}".format(epoch, train_loss, train_mAP, train_ROC))

    return train_loss, train_mAP, train_ROC 


def validate(model, testloader, criterion, epoch, device):
    # print("Validating ...")
    model.eval()
    counter = 0

    val_running_loss = 0.0
    val_running_mAP = 0.0
    val_running_ROC = 0.0

    with torch.no_grad():
        model_result = []
        targets = []
        for i, samples in tqdm(enumerate(testloader), desc="Loading Test batches: ", total=int(len(testloader.dataset) / testloader.batch_size)):
            counter += 1
            imgs, target = samples['frames'].to(device), samples['labels'].to(device)
            outputs = model(imgs) # sigmoid output

            loss = criterion(outputs, target.type(torch.float)) # targets.type(torch.float)

            model_result.extend(outputs.cpu().numpy())
            targets.extend(target.cpu().numpy())

            val_running_loss += loss.item()
            val_running_mAP += compute_mAP(outputs, target)
            val_running_ROC += compute_roc(outputs, target)

        result = calculate_metrics(np.array(model_result), np.array(targets))
        val_loss = val_running_loss / counter
        val_mAP = val_running_mAP / counter
        val_ROC = val_running_ROC / counter
        print("Epoch:{:2d} Test: "
              "micro f1: {:.3f} "
              "macro f1: {:.3f} "
              "samples f1: {:.3f}".format(epoch,
                                          result['micro/f1'],
                                          result['macro/f1'],
                                          result['samples/f1']))
        # print("Epoch: {} -- Valid Loss: {:.4f} -- Valid mAP: {:.4f} -- Valid ROC: {:.4f}".format(epoch, val_loss, val_mAP, val_ROC))

    return val_loss, val_mAP, val_ROC


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="r3d", choices=['c3d', 'r3d', 'r21d'], help="Choose backbone model")
    parser.add_argument("--dataset_path", type=str, default="/media/ican/XxX/Datasets/UCF101/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="/media/ican/XxX/Datasets/UCF101/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for the pseudo-labels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="Optional path to checkpoint model")
    
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_crop = T.Compose([RandomCrop(opt.img_dim)])
    center_crop = T.Compose([CenterCrop(opt.img_dim)])
    corner_crop = T.Compose([RandomCrop(opt.img_dim)])
    crop_transforms = {0: random_crop, 1: center_crop, 2: corner_crop}

    rotate_0 = T.Compose([RandomRotation(0)])
    rotate_90 = T.Compose([RandomRotation(90)])
    rotate_180 = T.Compose([RandomRotation(180)])
    rotate_270 = T.Compose([RandomRotation(270)])
    rotation_transforms = {3: rotate_0, 4: rotate_90, 5: rotate_180, 6: rotate_270}

    # Define training set
    train_dataset = UCF101Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        sequence_length=opt.sequence_length,
        training=True,
        rotation_transform=rotation_transforms,
        crop_transform=crop_transforms
    )
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    testdataset = UCF101Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        sequence_length=opt.sequence_length,
        training=False,
        rotation_transform = rotation_transforms,
        crop_transform=crop_transforms
    )
    testloader = DataLoader(testdataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # sample_dict = next(iter(trainloader)) 
    # print("seq: {}, label: {}".format(sample_dict['frames'].size(), sample_dict['labels']))
    # single_frame = sample_dict['frames'][0].permute(1, 0, 2, 3)[0] # torch.Size([16, 1, 16, 224, 224])
    
    # transform = T.ToPILImage()
    # img = transform(single_frame)
    # img.show()

    # # Dataset statistics
    # print("\n", "+++"*66)
    # print("Number of Training dataset: \t{}".format(int(len(trainloader.dataset))))
    # print("Number of Validation dataset: \t{}".format(int(len(testloader.dataset))))
    # print("Target Pseudo-Labels: {}".format(train_dataset.idx2target))
    # print("\n", "+++"*66)

    # Binary-Cross-Entropy
    criterion = nn.BCELoss(reduce=False)

    # Model definition
    if opt.model == "c3d":
        model = c3d.C3DBN(num_classes=opt.num_classes).to(device)
    elif opt.model == "r3d":
        model = r3d.R3DNet((2, 2, 2, 2), num_classes=opt.num_classes).to(device)
    else:
        model = r21d.R2Plus1DNet((2, 2, 2, 2), num_classes=opt.num_classes).to(device)

    # print("Model: ", model)
    # print("Model Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_val_losses = []

    for epoch in range(opt.num_epochs):
        
        train_epoch_loss, train_epoch_mAP, train_epoch_ROC = train(model, trainloader, optimizer, criterion, epoch, device)
        val_epoch_loss, val_epoch_mAP, val_epoch_ROC = validate(model, testloader, criterion, epoch, device)
        
        print("Epoch:{:2d} : Train Loss: {:.4f} : Valid Loss: {:.4f}".format(epoch, train_epoch_loss, val_epoch_loss))
        
        train_val_losses.append([train_epoch_loss, train_epoch_mAP, train_epoch_ROC, val_epoch_loss, val_epoch_mAP, val_epoch_ROC])

        if epoch % 20 == 0:
            checkpoint_save(model, opt.checkpoints, epoch)


    loss = pd.DataFrame(train_val_losses, columns=['train_loss', 'train_mAP', 'train_ROC', 'val_loss', 'val_mAP', 'val_ROC'], index=None)
    loss.to_csv(os.path.join(opt.checkpoints, "train_val_losses.csv"))


if __name__ == '__main__':
    import timeit
    start_time = timeit.default_timer()
    print("\n", "\t\t", "+++"*10, " \tMulti-Label Transformation Prediction (MLTP)\t", "+++"*10)
    main()
    print("\n", "\t\t", "+++"*10, " Total time taken to end the training: {:.4f} minutes.\t".format(int(timeit.default_timer() - start_time) / 60), "+++"*10)
    print()
    