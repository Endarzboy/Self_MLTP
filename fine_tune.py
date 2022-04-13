
import os
import cv2
import json
import time
import glob
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from helper.utils import get_current_time
from helper import utils, path_finder, metrics
from dataset.dataset_for_fine_tuning import UCF101Dataset

from models import r3d, c3d, r21d
# from dataset.ucf101dataset import *


def save_checkpoint(model_name, model, name, epoch):
    f = os.path.join(name, '{}_checkpoint-{:06d}.pth'.format(model_name, epoch))
    torch.save(model.state_dict(), f)
    print('Checkpoint Saved as: ', f)


def train(model, trainloader, criterion, optimizer, epoch, device, print_freq=10):
    batch_time = metrics.AverageMeter('Time', ':6.3f')
    data_time = metrics.AverageMeter('Data', ':6.3f')

    losses = metrics.AverageMeter('Loss', ':.4e')
    top1 = metrics.AverageMeter('Acc@1', ':6.2f')
    top5 = metrics.AverageMeter('Acc@5', ':6.2f')

    progress = metrics.ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )
    # Switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(trainloader):
        data_time.update(time.time() - end)

        image = batch[0].permute(0, 2, 1, 3, 4).to(device) # torch.Size([16, 8, 3, 112, 112])
        # B, T, C, H, W = image.shape
        # image = image.permute(B, C, T, H, W).to(device)

        label = batch[1].to(device)

        # Step 1: Clear gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        output = model(image)

        # Step 3: compute loss
        loss = criterion(output, label)

        acc1, acc5 = metrics.accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # Step 4: Compute gradient
        loss.backward()

        # Step 5: Optimize the model
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(model, validloader, criterion, device, print_freq=10):
    batch_time = metrics.AverageMeter('Time', ':6.3f')
    data_time = metrics.AverageMeter('Data', ':6.3f')

    losses = metrics.AverageMeter('Loss', ':.4e')
    top1 = metrics.AverageMeter('Acc@1', ':6.2f')
    top5 = metrics.AverageMeter('Acc@5', ':6.2f')

    progress = metrics.ProgressMeter(
        len(validloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Validate: ")
    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(validloader):
            image = batch[0].permute(0, 2, 1, 3, 4).to(device) # torch.Size([16, 8, 3, 112, 112])
            # B, T, C, H, W = image.shape
            # image = image.permute(B, C, T, H, W).to(device)
            label = batch[1].to(device)

            # Compute output
            output = model(image)
            loss = criterion(output, label)

            # Measure accuracy and record loss
            acc1, acc5 = metrics.accuracy(output, label, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def main(acc_loss_file="acc_and_loss"):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="r3d", choices=['c3d', 'r3d', 'r21d'], help="Choose backbone model")
    parser.add_argument("--dataset_path", type=str, default="/media/ican/XxX/Datasets/UCF101/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="/media/ican/XxX/Datasets/UCF101/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=8, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="color channels")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for the pseudo-labels")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/checkpoint-000080.pth", help="Optional path to checkpoint model")
    
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================================== Dataloaders ==============================#
    input_shape = (opt.channels, opt.img_dim, opt.img_dim)
    train_dataset = UCF101Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=input_shape,
        sequence_length=opt.sequence_length,
        training=True
    )

    # Train Data loader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # seq, label = next(iter(train_loader))
    # print("seq: {}, label: {}".format(seq.size(), label))
    # single_frame = seq[0][0]
    
    # transform = T.ToPILImage()
    # img = transform(single_frame)
    # img.show()

    # Read test dataset
    test_dataset = UCF101Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=input_shape,
        sequence_length=opt.sequence_length,
        training=False
    )

    # Test Data loader
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)


    print("Total size of Train set: ", len(train_loader.dataset))
    print("Total size of Test set: ", len(test_loader.dataset))
    print("--" * 103)

   
    # Model definition
    if opt.model == "c3d":
        net = c3d.C3DBN(num_classes=opt.num_classes).to(device)
    elif opt.model == "r3d":
        net = r3d.R3DNet((2, 2, 2, 2), num_classes=opt.num_classes).to(device)
    else:
        net = r21d.R2Plus1DNet((2, 2, 2, 2), num_classes=opt.num_classes).to(device)

    net.load_state_dict(torch.load(opt.checkpoint_path, map_location='cpu'), strict=False)
    # print("net: ", net)

    net.conv1 = r3d.SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
    cnn_layers = list(net.children())[:-2]

    model = nn.Sequential(*cnn_layers,
                          nn.Flatten(), # torch.Size([B, 512*1*1])
                          nn.Linear(512, train_dataset.num_classes),
                          nn.Softmax(dim=-1))

    # model = nn.Sequential(*cnn_layers,
    #                       nn.Flatten(), 
    #                       nn.Linear(512, 512),
    #                       nn.BatchNorm1d(512, momentum=0.01),
    #                       nn.ReLU(),
    #                       nn.Linear(512, train_dataset.num_classes),
    #                       nn.Softmax(dim=-1))
    model = model.to(device)
    # print("model: ", model)
    # print("Parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Criterion
    learning_rate = 1e-3
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        verbose=True
    )

    # dump args
    with open(os.path.join("checkpoints", '{}_{}_fine_tuning_args.json'.format(str(get_current_time()), model.__class__.__name__)), 'w') as fid:
        json.dump(opt.__dict__, fid, indent=2)

    # =============================== Summary logger ==========================#
    log_dir = os.path.join(os.getcwd(), 'logs', get_current_time() + "_" + opt.model + "_fine_tuned_" + str(opt.num_epochs))
    save_dir = os.path.join(os.getcwd(), 'checkpoints', get_current_time() + "_fine_tuned_" + opt.model + "_"
                            + str(opt.num_epochs))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    logger = SummaryWriter(log_dir)

    info = {
        'train_losses': [],
        'valid_losses': [],
        'train_acc': [],
        'valid_acc': [],
    }

    for epoch in range(opt.num_epochs):
        print("\n[INFO:] ===> Training")
        metrics.adjust_learning_rate(optimizer, epoch, learning_rate)

        # Train for each epoch
        top1_acc, train_loss = train(model, train_loader, criterion, optimizer, epoch, device)

        # Validate on each set
        top1_val_acc, val_loss = validate(model, test_loader, criterion, device)

        scheduler.step(val_loss)

        # Track losses and accuracies
        info['train_losses'].append(train_loss)
        info['valid_losses'].append(val_loss)

        info['train_acc'].append(top1_acc.item())
        info['valid_acc'].append(top1_val_acc.item())

        # Log metrics
        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('valid_loss', val_loss, epoch)

        logger.add_scalar('train_acc', top1_acc, epoch)
        logger.add_scalar('val_acc', top1_val_acc, epoch)

        if epoch % 20 == 0:
            save_checkpoint(opt.model, model, save_dir, epoch)

    with open(
            os.path.join(os.getcwd(), acc_loss_file + "_" + get_current_time() + "_" + str(epochs) + '.csv'),
            'w') as f:
        for k in info.keys():
            f.write('%s,%s\n' % (k, info[k]))

    # Save the training and validation losses to disk
    fig_name = "./figures/" + opt.model + "_fine_tune_loss.jpg"
    fig_name2 = "./figures/" + opt.model + "_fine_tune_acc.jpg"
    utils.save_loss_plot(info['train_losses'], info['valid_losses'], fig_name)
    utils.save_accuracy_plot(info['train_acc'], info['valid_acc'], fig_name2)

if __name__ == '__main__':
    import timeit

    start = timeit.default_timer()

    # Set Parameters for main function.
    main(acc_loss_file="acc_and_loss")

    print("\nTraining completed in {:.2f} minutes.".format(int(timeit.default_timer() - start) / 60))
    print("++" * 103)
