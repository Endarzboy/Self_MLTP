
import os
import cv2
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

from models import r3d, c3d, r21d
from dataset.ucf101dataset import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="r3d", choices=['c3d', 'r3d', 'r21d'], help="Choose backbone model")
    parser.add_argument("--dataset_path", type=str, default="/media/ican/XxX/Datasets/UCF101/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="/media/ican/XxX/Datasets/UCF101/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=8, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for the pseudo-labels")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/checkpoint-000080.pth", help="Optional path to checkpoint model")
    
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = T.Compose([RandomCrop(opt.img_dim)])
    test_transforms = T.Compose([CenterCrop(opt.img_dim)])

    random_crop = T.Compose([RandomCrop(opt.img_dim)])
    center_crop = T.Compose([CenterCrop(opt.img_dim)])
    corner_crop = T.Compose([RandomCrop(opt.img_dim)])

    crop_transforms = {0: random_crop, 1: center_crop, 2: corner_crop}

    rotate_0 = T.Compose([RandomRotation(0)])
    rotate_90 = T.Compose([RandomRotation(90)])
    rotate_180 = T.Compose([RandomRotation(180)])
    rotate_270 = T.Compose([RandomRotation(270)])

    rotation_transforms = {3: rotate_0, 4: rotate_90, 5: rotate_180, 6: rotate_270}


    # Define dataset
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
   
    # Model definition
    if opt.model == "c3d":
        model = c3d.C3DBN(num_classes=opt.num_classes).to(device)
    elif opt.model == "r3d":
        model = r3d.R3DNet((2, 2, 2, 2), num_classes=opt.num_classes).to(device)
    else:
        model = r21d.R2Plus1DNet((2, 2, 2, 2), num_classes=opt.num_classes).to(device)

    model.load_state_dict(torch.load(opt.checkpoint_path, map_location='cpu'), strict=False)
    model.eval()

    # Class-Labeles
    class_labels = testdataset.idx2target
    print("class_labels: ", class_labels)

    for j, samples in tqdm(enumerate(testloader), desc="Loading Test batches: ", total=int(len(testloader.dataset) / testloader.batch_size)):
        imgs, targets = samples['frames'].to(device), samples['labels']
        # get all the index positions of each three multi-labels (where value == 1 for one-hot encoded targets) 
        target_indices = [i for i in range(len(targets[0])) if targets[0][i]==1] 
        # print("target_indices: ", target_indices)
        outputs = model(imgs) # sigmoid output

        outputs = outputs.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
        # print("sorted_indices: ", sorted_indices) # 
        best = sorted_indices[-3:]
        string_predicted = ""
        string_actual = ""
        # Save a single video frames from each batch up to three iterations
        # imgs = torch.Size([B, C, T, H, W])
        if j <= 2:
            for i in range(len(best)):
                string_predicted += f"{class_labels[int(best[i])]}\t"

            for i in range(len(target_indices)):
                string_actual += f"{class_labels[int(target_indices[i])]}\t"
                
            clip = imgs[0].permute(1, 2, 3, 0) # torch.Size([T, H, W, C])
            for idx in range(clip.size(0)):
                frame = clip[idx] # .squeeze(1)
                frame = frame.detach().cpu().numpy()
                # frame = np.transpose(frame, (1, 2, 0))
                plt.imshow((frame * 255).astype(np.uint8))
                plt.axis('off')
                plt.title(f"Predicted: {string_predicted}\nActual: {string_actual}")
                plt.savefig(f"./figures/inference_{j+idx+1}.jpg")
                plt.show()

if __name__ == '__main__':
    main()
