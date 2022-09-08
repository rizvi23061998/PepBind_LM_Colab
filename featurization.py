import os 
import numpy as np
import pandas as pd 
import copy
from Bio import SeqIO
import torch
from transformers import BertModel, BertTokenizer
import re

import requests
from tqdm.auto import tqdm
import pickle as pkl
import subprocess

def batch_featurization(sequences, device, model,tokenizer):
  ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  # print(get_gpu_memory_map())

  with torch.no_grad():
      embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
  # print(get_gpu_memory_map())

  embedding = embedding.cpu().numpy()
  features = [] 
  for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][1:seq_len-1]
    features.append(seq_emd)
  
  del embedding
  del input_ids
  del attention_mask
  del model
  torch.cuda.empty_cache()


def featurization(sequences, device, model, tokenizer, batch_size, output_folder):
  n = len(sequences)
  features = []
  
  for i in range(int(n/batch_size)+1):
    low = int(batch_size)*i
    high = min(int(batch_size)*(i+1), n)
    print("batch " + str(i) + " starting. Low :",low, ",High:", high)
    feature = batch_featurization(sequences[low:high], device, model, tokenizer)
    print("batch " + str(i) + " features generated.")
    features.extend(feature)

    print("batch " + str(i) + " done.")
    print("Current feature length:", len(features))

    
  return features

def main():
    #Initialize model 
    model_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Model/protbert_bfd/"
    tokenizer = BertTokenizer.from_pretrained(model_folder, do_lower_case=False)
    model = BertModel.from_pretrained(model_folder)

    #Load the model to the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    # get_gpu_memory_map()

    out_folder = "/content/drive/MyDrive/Masters/PepBind_LM/Features/"
    train_features = featurization(train_sequences,device, model,tokenizer, 32, out_folder)
    with open(out_folder + 'train_feature_all'+ '.pkl', 'wb') as handle:
      pkl.dump(train_features, handle)
    
    test_features = featurization(test_sequences,device, model,tokenizer, 32, out_folder)
    with open(out_folder + 'test_feature_all'+ '.pkl', 'wb') as handle:
        pkl.dump(test_features, handle)



if name == "__main__":
    main()