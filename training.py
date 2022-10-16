import argparse
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
from models import CNN2Layers, Logistic_Reg_model, LSTM_base
from torchsummary import summary
from torchmetrics.functional import f1_score,matthews_corrcoef
from dataset import Seq_Dataset
from dataset import Res_Dataset
from train_utils import prepare_res_features, prepare_subsets, prepare_seq_features, EarlyStopping
import time
import random
# import 
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(model, train_dataloader, opt, lossFn, trainSteps, subset_models=None):
    # set the model in training mode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    if subset_models != None:
        for subset_model in subset_models:
            subset_model.eval()
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    all_preds = torch.tensor([])
    targets = torch.tensor([], dtype=int)
    # loop over the training set
    # for (x, y) in zip(X_train, y_train):
    for batch_idx, (x, y) in enumerate(train_dataloader):
        (x, y) = (x.to(device), y.to(device))
        if subset_models != None:
            intermediate_out = torch.tensor([])
            intermediate_out = intermediate_out.to(device)
            with torch.no_grad():
                
                for subset_model in subset_models:
                    out_i = torch.sigmoid(subset_model(x))
                    out_i = out_i.reshape(( trainSteps, 1))
                    intermediate_out = torch.cat((intermediate_out,out_i), axis=1)
        else:
            intermediate_out = x

        pred = (model(intermediate_out))
        pred = pred.reshape([trainSteps])
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()


        # print(type(pred))
        pred = torch.sigmoid(pred)
        all_preds = torch.cat( (all_preds, pred.cpu().detach()) )
        targets = torch.cat( (targets, y.cpu().detach().int()) )
        pred = (pred>0.5).float()
        totalTrainLoss += loss
        trainCorrect += (np.array(pred.cpu()) == np.array(y.cpu())).astype(int).sum()
    
    return totalTrainLoss, trainCorrect, all_preds, targets


def validate(model, val_dataloader, lossFn, valSteps, subset_models=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    totalValLoss = 0
    valCorrect = 0
    all_preds = torch.tensor([])
    targets = torch.tensor([], dtype=int)

    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for batch_idx, (x,y) in enumerate(val_dataloader):
            (x, y) = (x.to(device), y.to(device))
            if subset_models != None:
                intermediate_out = torch.tensor([])
                intermediate_out = intermediate_out.to(device)
                with torch.no_grad():                
                    for subset_model in subset_models:
                        out_i = torch.sigmoid(subset_model(x))
                        out_i = out_i.reshape((valSteps, 1))
                        # print(out_i.shape)
                        intermediate_out = torch.cat((intermediate_out,out_i), axis=1)
            else:
                intermediate_out = x

            
            pred = model(intermediate_out)
            pred = pred.reshape([valSteps])
            loss = lossFn(pred, y)                    
            
            pred = torch.sigmoid(pred)
            all_preds = torch.cat( (all_preds, pred.cpu().detach()) )
            targets = torch.cat( (targets, y.cpu().detach().int()) )

            pred = (pred > 0.5).float()
            totalValLoss += loss
            # calculate the number of correct predictions
            valCorrect += (np.array(pred.cpu()) == np.array(y.cpu())).astype(int).sum()
    return totalValLoss, valCorrect, all_preds, targets

def train_subset(data, model, opt, lossFn, history, trainSteps=128, valSteps=128, EPOCHS=50, subset_models=None):
    H = history
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_size = int(len(data)*.8)
    val_size = len(data) - train_size
    print(train_size, " ", val_size)

    train_data, val_data = random_split(data, [train_size, val_size])
    train_dataloader = DataLoader(train_data, batch_size = trainSteps, shuffle = True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size = valSteps, shuffle = True, drop_last=True)
    
    scheduler = ReduceLROnPlateau(opt, 'min', patience = 3)
    early_stopping = EarlyStopping(min_delta=1e-8, patience=6)
    best_model = None
    best_mcc = -1
    best_f1 = -1
    # train_data_loader = 
    # loop over our epochs
    for e in range(0, EPOCHS):
        print("On Epoch = ",e)
        totalTrainLoss, trainCorrect, train_preds, train_targets = train(model=model, train_dataloader=train_dataloader,
                                         opt = opt, lossFn=lossFn, trainSteps=trainSteps, subset_models=subset_models)
        # switch off autograd for evaluation
        totalValLoss, valCorrect, val_preds, val_targets = validate(model, val_dataloader, lossFn, valSteps, subset_models=subset_models)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / train_size
        valCorrect = valCorrect / val_size
        #calculate f1_score and mcc
        f1_train = f1_score(train_preds, train_targets)
        mcc_train = matthews_corrcoef(train_preds, train_targets, num_classes=2)

        f1_val = f1_score(val_preds, val_targets)
        mcc_val = matthews_corrcoef(val_preds, val_targets, num_classes=2)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        if e % 1 == 0:
            print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(
                avgTrainLoss, trainCorrect, f1_train, mcc_train))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(
                avgValLoss, valCorrect, f1_val, mcc_val))

        if mcc_val>best_mcc or best_model == None:
            best_mcc = mcc_val
            best_model = copy.deepcopy(model)
            best_f1 = f1_val

        scheduler.step(totalValLoss)
        early_stopping(totalValLoss)
        if early_stopping.early_stop:
            break

    return best_model, best_mcc, best_f1




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sub', type=int, default=0)
    parser.add_argument('--pos_weight', type=int, default=16)
    parser.add_argument('--test_only', type=int, default=1)
    args = parser.parse_args()
    config = {"batch_size": 64,
          "hidden_dim": 256,
          "lstm_layers": 128,
          "emdedding_len": 1024,
          "dropout_ratio": 0.5,
          "trp": 1}

    torch.manual_seed(10)
    
    root_folder = "/content/drive/MyDrive/Masters/"
    # root_folder = "/content/drive/MyDrive/"
    data_folder = root_folder + "PepBind_LM/Data/"
    feature_folder = root_folder + "PepBind_LM/Features/"
    model_folder = root_folder + "PepBind_LM/Model/"
    print("Preparing residue level features .. ...")
    train_dataset, test_dataset, train_samples = prepare_res_features(data_folder, feature_folder, 15, trp = config["trp"])

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
    if args.train_sub == 1:
        # model = CNN2Layers(1024, 128, 5, 1, 2, 0.3, 256)
        # print(summary(model, (31, 256, 5, 1, 2, 0.5, 128)))
        # optim = Adam(model.parameters(), lr=1e-3)
        
        subset_model_list = []
        count = 0
        for sample_i in train_samples:
            # model = CNN2Layers(1024, 128, 5, 1, 2, 0.3, 256)
            model = LSTM_base(config)
            optim = Adam(model.parameters(), lr=1e-3)
            pos_weight = torch.tensor(np.array([3]), dtype=float).to(device)
            lossFn = BCEWithLogitsLoss(pos_weight=pos_weight)
        
            subset_model, mcc, f1 = train_subset(sample_i, model, optim, lossFn, H, trainSteps= config["batch_size"], valSteps= config["batch_size"],EPOCHS= 50)
            subset_model_list.append(subset_model)
            count += 1
            print("===========Model {} training finished ========== \n", format(count))
            print("MCC: ", mcc, ", F1: ", f1)
            with open(model_folder + 'subset_models_'+ str(count) + '.pkl', 'wb') as handle:
                pkl.dump(subset_model, handle)
        with open(model_folder + 'subset_models.pkl', 'wb') as handle:
            pkl.dump(subset_model_list, handle)
    if(args.test_only == 0):
        # with open(model_folder + 'subset_models.pkl', 'rb') as handle:
        #     subset_model_list = pkl.load( handle)
        # model = Logistic_Reg_model(no_input_features=10)
        model = CNN2Layers(in_channels= 1024, feature_channels= 128,kernel_size= 3,stride= 1,padding= 2,dropout= 0.7,batch_size= 512)
        # print(summary(model, (31, 256, 5, 1, 2, 0.5, 128)))
        optim = Adam(model.parameters(), lr=1e-3)
        pos_weight = torch.tensor(np.array([args.pos_weight]), dtype=float).to(device)
        lossFn = BCEWithLogitsLoss(pos_weight=pos_weight)
        
        ensemble_model, mcc, f1 = train_subset(train_dataset, model, optim, lossFn, H, trainSteps= 512, valSteps= 512,EPOCHS= 50, subset_models=None)

        print("===========Model training finished ========== \n")
        print("MCC: ", mcc, ", F1: ", f1)
    
        with open(model_folder + 'ensembler.pkl', 'wb') as handle:
            pkl.dump(ensemble_model, handle)
    else:
        with open(model_folder + 'subset_models.pkl', 'rb') as handle:
            subset_model_list = pkl.load( handle)
        
        with open(model_folder + 'ensembler.pkl', 'rb') as handle:
            ensemble_model = pkl.load(handle)
        lossFn = BCEWithLogitsLoss()
        print(test_dataset.__len__())
        test_dataloader = DataLoader(test_dataset, batch_size = 512, shuffle = True, drop_last=True)
        test_loss, test_corr, test_preds, test_targets = validate(model = ensemble_model, val_dataloader = test_dataloader, lossFn = lossFn, valSteps = 512, subset_models=subset_model_list)
        test_corr = test_corr / test_dataset.__len__()
        #calculate f1_score and mcc
        f1_test = f1_score(test_preds, test_targets)
        mcc_test = matthews_corrcoef(test_preds, test_targets, num_classes=2)
        print("Accuracy: ", test_corr, ", F1: ", f1_test, ", MCC: ", mcc_test )
    endTime = time.time()
    

if __name__ == "__main__":
    main()