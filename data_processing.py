# import libraries

import os 
import numpy as np
import pandas as pd 
import copy
from Bio import SeqIO
# import wget
import torch
# from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import BertModel, BertTokenizer
import re

import requests
from tqdm.auto import tqdm
import pickle as pkl
import subprocess


# initialization
data_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Data/"
train_data = data_folder + "Train.txt"
test_data = data_folder + "Test.txt"

test_df = pd.read_csv(test_data, lineterminator='>', delimiter='\n', header = None, names=['id', 'sequence', 'label', 'tmp'])
train_df = pd.read_csv(train_data, lineterminator='>', delimiter='\n', header = None, names=['id', 'sequence', 'label', 'tmp'])

#Convert to list and Dump as Pickle
train_sequences = train_df['sequence'].tolist()
with open(data_folder + "train_sequences.pkl", "wb") as fp:
  pkl.dump(train_sequences, fp)

train_sequences = [" ".join(sequence) for sequence in train_sequences]
train_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in train_sequences]
print(len(train_sequences))

test_sequences = test_df['sequence'].tolist()
with open(data_folder + "test_sequences.pkl", "wb") as fp:
  pkl.dump(test_sequences, fp)

test_sequences = [" ".join(sequence) for sequence in test_sequences]
test_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in test_sequences]
print(len(test_sequences))


train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()
with open(data_folder + "test_labels.pkl", "wb") as fp:
  pkl.dump(test_labels, fp)
with open(data_folder + "train_labels.pkl", "wb") as fp:
  pkl.dump(train_labels, fp)




