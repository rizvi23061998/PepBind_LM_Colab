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
from cnn_models import CNN2Layers
from torchsummary import summary

import time
import random
# import 
import gc


def prepare_res_features(data_folder, feature_folder):
    with open(feature_folder + "train_feature_all.pkl", "rb") as fp:
        train_features = pkl.load(fp)

    with open(feature_folder + "test_feature_all.pkl", "rb") as fp:
        test_features = pkl.load(fp)

    
    with open(data_folder + "train_sequences.pkl", "rb") as fp:
        train_sequences = pkl.load(fp)
    with open(data_folder + "test_sequences.pkl", "rb") as fp:
        test_sequences = pkl.load(fp)

    with open(data_folder + "test_labels.pkl", "rb") as fp:
        test_labels = pkl.load( fp)
    with open(data_folder + "train_labels.pkl", "rb") as fp:
        train_labels = pkl.load(fp)

    print(train_features[0].shape)

    train_features_padded = []
    test_features_padded = []
    for feature in train_features:
        new_feature = np.concatenate([np.zeros((15,1024),dtype=float),feature,np.zeros((15,1024),dtype=float)])
        train_features_padded.append(new_feature)

    for feature in test_features:
        new_feature = np.concatenate([np.zeros((15,1024),dtype=float),feature,np.zeros((15,1024),dtype=float)])
        test_features_padded.append(new_feature)

    # train_x = np.concatenate(train_features)
    # test_x = np.concatenate(test_features)


    print(len(train_features_padded))
    print(len(test_features_padded))

    with open(data_folder + "test_labels.pkl", "rb") as fp:
        test_labels = pkl.load( fp)
    with open(data_folder + "train_labels.pkl", "rb") as fp:
        train_labels = pkl.load(fp)

    # print(train_labels)

    train_labels = [" ".join(train_label) for train_label in train_labels]
    test_labels = [" ".join(test_label) for test_label in test_labels]

    # print(train_labels)

    train_y = [np.fromstring(train_label, dtype=int, sep=' ') for train_label in train_labels]
    test_y = [np.fromstring(test_label, dtype=int, sep=' ') for test_label in test_labels]

    train_data_res = []
    for chain in train_features_padded:
        for residue_idx in range(15,len(chain) - 15):
            residue_feature = chain[(residue_idx - 15): (residue_idx + 16),:]
            train_data_res.append(residue_feature)
        # print(len(chain)-30)
    print(len(train_data_res))

    test_data_res = []
    for chain in test_features_padded:
        for residue_idx in range(15,len(chain) - 15):
            residue_feature = chain[(residue_idx - 15): (residue_idx + 16),:]
            test_data_res.append(residue_feature)

    train_y_cat = np.concatenate(train_y)
    train_y = train_y_cat.tolist()
    test_y_cat = np.concatenate(test_y)
    test_y = test_y_cat.tolist()

    return train_data_res, train_y

def prepare_subsets(train_data_res, train_y, sample_reps):
    
    train_x_pos = [train_data_res[i] for i in range(len(train_data_res)) if train_y[i]==1]
    train_x_neg = [train_data_res[i] for i in range(len(train_data_res)) if train_y[i]==0]
    # train_x_neg = train_data_res[train_neg_idx]
    print("Positive  samples:", len(train_x_pos))
    print("Negative sampels: ", len(train_x_neg))
    # train_x_pos[]
    
    train_x_subsets = []
    train_y_subsets = []
    for i in range(sample_reps):
        x_subset = train_x_pos[:]
        subset_idx = random.sample(range(0,len(train_x_neg)), int(0.2*len(train_x_neg)))
        for idx in subset_idx:
            x_subset.append(train_x_neg[idx])
  
        y_subset = [1]*len(train_x_pos)
        y_subset.extend([0]*len(subset_idx))
        # print("x length:",len(x_subset),",subset: ",i)
        # print("y length:",len(y_subset),",subset: ",i)
        
        tmp = list(zip(x_subset,y_subset))
        random.shuffle(tmp)
        x_subset, y_subset = zip(*tmp)
        x_subset, y_subset = list(x_subset), list(y_subset)

        train_x_subsets.append(x_subset)
        train_y_subsets.append(y_subset)

        return train_x_subsets, train_y_subsets

def train_subset(X_train, y_train, X_val, y_val, model, optim, lossFn, history, trainSteps=128, valSteps=128, EPOCHS=20):
    # EPOCHS = 20
    # trainSteps = 256
    # valSteps = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # loop over our epochs
    for e in range(0, EPOCHS):
        print("On Epoch = ",e)
        # set the model in training mode
        model.train()
        # print(model.init_conv1.weight)
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        # for (x, y) in zip(X_train, y_train):
        for batch_idx in range(0, len(X_train), trainSteps): 
            # send the input to the device
            batch_start = batch_idx
            batch_end = min(batch_start + trainSteps, len(X_train))

            
            x_cat = np.stack(X_train[batch_start : batch_end])
            y_cat = np.stack(y_train[batch_start : batch_end])
            x = torch.Tensor( x_cat )
            y = torch.Tensor( y_cat )
            cur_batch_size = batch_end - batch_start
            x = np.reshape(x, (cur_batch_size, x_cat.shape[1], x_cat.shape[2]))
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = torch.sigmoid(model(x))
            # pred = torch.round(pred)
            pred = pred.reshape([cur_batch_size])
            # print(type(pred), type(y))
            # print(pred.shape, y.shape)
            # print(pred)
            # print(y)

            # print((pred.argmax(1) == y))
            # print(pred.get_device(), y.get_device())
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(0) == y).type(
                torch.float).sum().item()
        
            # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for batch_idx in range(0, len(X_val), valSteps):
                batch_start = batch_idx
                batch_end = min(batch_start + valSteps, len(X_val))
                x_cat = np.stack( X_val[batch_start : batch_end] )
                y_cat = np.stack( y_val[batch_start : batch_end] )
                x = torch.Tensor( x_cat )
                y = torch.Tensor( y_cat )
                cur_batch_size = batch_end - batch_start
                # send the input to the device
                x = np.reshape(x, ( cur_batch_size, x_cat.shape[1], x_cat.shape[2]))
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = torch.sigmoid(model(x))
                # pred = torch.round(pred)
                pred = pred.reshape([cur_batch_size])
                totalValLoss += lossFn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(0) == y).type(
                    torch.float).sum().item()
        
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(X_train)
        valCorrect = valCorrect / len(X_val)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

def main():
    data_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Data/"
    feature_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Features/"

    print("Preparing residue level features .. ...")
    train_data_res, train_y = prepare_res_features(data_folder, feature_folder)

    print("Preparing subsets by undersampling ... ...")
    train_x_subsets, train_y_subsets = prepare_subsets(train_data_res, train_y, 10)

    H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    X_train, X_val, y_train, y_val = train_test_split(train_x_subsets[0] , train_y_subsets[0],
                                                        stratify=train_y_subsets[0], 
                                                        test_size=0.2, random_state= 10)
    print(len(X_train), len(y_train))
    model = CNN2Layers(31, 256, 5, 1, 2, 0.5)
    print(summary(model, (31, 256, 5, 1, 2, 0.5)))
    optim = Adam(model.parameters(), lr=1e-3)
    lossFn = BCEWithLogitsLoss()
    train_subset(X_train, y_train, X_val, y_val, model, optim, lossFn, H)

    endTime = time.time()
    

if __name__ == "__main__":
    main()
