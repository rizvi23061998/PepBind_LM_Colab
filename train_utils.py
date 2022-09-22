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
# from train_utils import prepare_res_features, prepare_subsets, prepare_seq_features
import time
import random
# import 
import gc


def prepare_seq_features(data_folder, feature_folder):
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
    
    max_len_train = -1
    max_len_test = -1
    train_chain_lengths = []
    test_chain_lengths = []
    for feature in train_features:
        if max_len_train < len(feature):
            max_len_train = len(feature)
        train_chain_lengths.append(len(feature))
         
    for feature in test_features:
        if max_len_test < len(feature):
            max_len_test = len(feature)
        test_chain_lengths.append(len(feature))
    
    max_len = max_len_train if (max_len_train > max_len_test) else max_len_test

    train_labels = [" ".join(train_label) for train_label in train_labels]
    test_labels = [" ".join(test_label) for test_label in test_labels]

    train_y = [np.fromstring(train_label, dtype=int, sep=' ') for train_label in train_labels]
    test_y = [np.fromstring(test_label, dtype=int, sep=' ') for test_label in test_labels]

    for feature in train_features:
        new_feature = np.concatenate([feature,np.zeros((max_len-len(feature),1024),dtype=float)])
        train_features_padded.append(np.transpose(new_feature))

    for feature in test_features:
        new_feature = np.concatenate([feature,np.zeros((max_len - len(feature),1024),dtype=float)])
        test_features_padded.append(np.transpose(new_feature))

    train_dataset = Seq_Dataset(features=train_features_padded, targets = train_y, seq_lens = train_chain_lengths)
    test_dataset = Seq_Dataset(features=test_features_padded, targets = test_y, seq_lens = test_chain_lengths)    




def prepare_res_features(data_folder, feature_folder, pad_len):
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
        new_feature = np.concatenate([np.zeros((pad_len,1024),dtype=float),feature,np.zeros((pad_len,1024),dtype=float)])
        train_features_padded.append((new_feature))

    for feature in test_features:
        new_feature = np.concatenate([np.zeros((pad_len,1024),dtype=float),feature,np.zeros((pad_len,1024),dtype=float)])
        test_features_padded.append((new_feature))

    train_labels = [" ".join(train_label) for train_label in train_labels]
    test_labels = [" ".join(test_label) for test_label in test_labels]

    train_y = [np.fromstring(train_label, dtype=int, sep=' ') for train_label in train_labels]
    test_y = [np.fromstring(test_label, dtype=int, sep=' ') for test_label in test_labels]

    train_data_res = []
    for chain in train_features_padded:
        for residue_idx in range(pad_len,len(chain) - pad_len):
            residue_feature = chain[(residue_idx - pad_len): (residue_idx + pad_len + 1),:]
            train_data_res.append(np.transpose(residue_feature))
        # print(len(chain)-30)
    print(len(train_data_res))

    test_data_res = []
    for chain in test_features_padded:
        for residue_idx in range(pad_len,len(chain) - pad_len):
            residue_feature = chain[(residue_idx - pad_len): (residue_idx + pad_len + 1),:]
            test_data_res.append(np.transpose(residue_feature))

    train_y_cat = np.concatenate(train_y)
    train_y = train_y_cat.tolist()
    test_y_cat = np.concatenate(test_y)
    test_y = test_y_cat.tolist()

    train_dataset = Res_Dataset(features = train_data_res, targets = train_y)
    test_dataset = Res_Dataset(features = test_data_res, targets = test_y)
    train_subsets = prepare_subsets(train_data_res, train_y, 10)
    return train_dataset, test_dataset, train_subsets

def prepare_subsets(train_data_res, train_y, sample_reps):
    
    train_x_pos = [train_data_res[i] for i in range(len(train_data_res)) if train_y[i]==1]
    train_x_neg = [train_data_res[i] for i in range(len(train_data_res)) if train_y[i]==0]
    # train_x_neg = train_data_res[train_neg_idx]
    print("Positive  samples:", len(train_x_pos))
    print("Negative sampels: ", len(train_x_neg))
    # train_x_pos[]
    
    train_subsets = []
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

        data = Res_Dataset(features = x_subset, targets = y_subset)
        train_subsets.append(data)
        # train_x_subsets.append(x_subset)
        # train_y_subsets.append(y_subset)


    return train_subsets
