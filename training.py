import os 
import numpy as np
import pandas as pd 
import copy
from Bio import SeqIO
# import wget
import torch

import re

import requests
from tqdm.auto import tqdm
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Sigmoid, LogSoftmax
from torch import flatten
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss
from torch.utils.data import random_split, DataLoader 
from cnn_models import CNN2Layers
from torchsummary import summary
from torchmetrics.functional import f1_score,matthews_corrcoef
from dataset import Seq_Dataset
from dataset import Res_Dataset
from train_utils import prepare_res_features, prepare_subsets, prepare_seq_features
import time
import random
# import 
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(model, train_dataloader, opt, lossFn, trainSteps):
    # set the model in training mode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    all_preds = np.array([])
    targets = np.array([])
    # loop over the training set
    # for (x, y) in zip(X_train, y_train):
    for batch_idx, (x, y) in enumerate(train_dataloader):
        (x, y) = (x.to(device), y.to(device))
        # print(x.type())
        pred = torch.sigmoid(model(x))
        pred = pred.reshape([trainSteps])
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # print(type(pred))
        all_preds = np.concatenate( (all_preds, pred.cpu().detach().numpy()) )
        targets = np.concatenate( (targets, y.cpu().detach().numpy()) )
        pred = (pred>0.5).float()
        totalTrainLoss += loss
        trainCorrect += (np.array(pred.cpu()) == np.array(y.cpu())).astype(int).sum()
    
    return totalTrainLoss, trainCorrect, all_preds, targets


def validate(model, val_dataloader, lossFn, valSteps):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    totalValLoss = 0
    valCorrect = 0
    all_preds = np.array([])    
    targets = np.array([])
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for batch_idx, (x,y) in enumerate(val_dataloader):
            (x, y) = (x.to(device), y.to(device))
            pred = torch.sigmoid(model(x))
            pred = pred.reshape([valSteps])
            loss = lossFn(pred, y)                    
            
            all_preds = np.concatenate( (all_preds, pred.cpu().detach().numpy() ) )
            targets = np.concatenate( (targets, y.cpu().detach().numpy()) )

            pred = (pred > 0.5).float()
            totalValLoss += loss
            # calculate the number of correct predictions
            valCorrect += (np.array(pred.cpu()) == np.array(y.cpu())).astype(int).sum()
    return totalValLoss, valCorrect, all_preds, targets

def train_subset(data, model, opt, lossFn, history, trainSteps=128, valSteps=128, EPOCHS=50):
    H = history
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_size = int(len(data)*.8)
    val_size = len(data) - train_size
    print(train_size, " ", val_size)

    train_data, val_data = random_split(data, [train_size, val_size])
    train_dataloader = DataLoader(train_data, batch_size = trainSteps, shuffle = True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size = valSteps, shuffle = True, drop_last=True)
    
    scheduler = ReduceLROnPlateau(opt, 'min')
    # train_data_loader = 
    # loop over our epochs
    for e in range(0, EPOCHS):
        print("On Epoch = ",e)
        totalTrainLoss, trainCorrect, train_preds, train_targets = train(model=model, train_dataloader=train_dataloader,
                                         opt = opt, lossFn=lossFn, trainSteps=trainSteps)
        # switch off autograd for evaluation
        totalValLoss, valCorrect, val_preds, val_targets = validate(model, val_dataloader, lossFn, valSteps)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / train_size
        valCorrect = valCorrect / val_size
        #calculate f1_score and mcc
        f1_train = f1_score(train_preds, train_targets)
        mcc_train = matthews_corrcoef(train_preds, train_targets)

        f1_val = f1_score(val_preds, val_targets)
        mcc_val = matthews_corrcoef(val_preds, val_targets)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(
            avgTrainLoss, trainCorrect, f1_train, mcc_train))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(
            avgValLoss, valCorrect, f1_val, mcc_val))

        scheduler.step(totalValLoss)

def main():
    data_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Data/"
    feature_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Features/"

    print("Preparing residue level features .. ...")
    train_dataset, test_dataset, train_subsets = prepare_res_features(data_folder, feature_folder, 15)

    H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    # X_train, X_val, y_train, y_val = train_test_split(train_x_subsets[0] , train_y_subsets[0],
    #                                                     stratify=train_y_subsets[0], 
    #                                                     test_size=0.2, random_state= 10)
    # print(len(X_train), len(y_train))
    model = CNN2Layers(1024, 128, 5, 1, 2, 0.5, 256)
    # print(summary(model, (31, 256, 5, 1, 2, 0.5, 128)))
    optim = Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor(np.array([3]), dtype=float).to(device)
    lossFn = BCEWithLogitsLoss(pos_weight=pos_weight)
    train_subset(train_subsets[0], model, optim, lossFn, H, trainSteps= 256, valSteps= 256,EPOCHS= 30)

    endTime = time.time()
    

if __name__ == "__main__":
    main()