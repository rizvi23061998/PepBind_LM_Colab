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

    print(train_labels)

    train_labels = [" ".join(train_label) for train_label in train_labels]
    test_labels = [" ".join(test_label) for test_label in test_labels]

    print(train_labels)

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

def main():
    data_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Data/"
    feature_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Features/"

    print("Preparing residue level features .. ...")
    train_data_res, train_y = prepare_res_features(data_folder, feature_folder)

    print("Preparing subsets by undersampling ... ...")
    prepare_subsets(train_data_res, train_y)




if __name__ == "__main__":
    main()
