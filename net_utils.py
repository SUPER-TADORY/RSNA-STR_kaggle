import enum
import os
import sys
import numpy as np
from numpy.lib import index_tricks
import pandas as pd
from argparse import ArgumentParser
import glob
from pandas.io.parsers import PythonParser
import torch
import torch.nn as nn
from torchvision import models,transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
#%matplotlib inline
from matplotlib import pyplot as plt
import random
import pickle
import json
from PIL import Image

def train_val_split(input_list,label_list):
    train_input=[]
    val_input=[]
    train_label=[]
    val_label=[]

    for tensor,label in zip(input_list,label_list):
        r=random.random()
        if r>=0.3:
            train_input.append(tensor)
            train_label.append(label)
        else:
            val_input.append(tensor)
            val_label.append(label)
    
    return train_input,val_input,train_label,val_label


class Dataset1(data.Dataset):
    def __init__(self,input_list,label_list,seq_len,phase="train"):
        self.input_list=input_list
        self.label_list=label_list
        self.seq_len=seq_len
        self.phase=phase
    
    def __len__(self):
        return len(self.file_list)

    #患者(index)一人のimagesから指定したseq_len枚をランダムに取り出す
    def __getitem__(self,index):
        i_all_seq=self.input_list[index]
        i_all_label=self.label_list[index]

        if not len(i_all_seq)==len(i_all_label):
            print('Error: labelがずれています', file=sys.stderr)
            sys.exit(1)

        r=random.randint(0,len(i_all_seq)-self.seq_len)
        i_input=i_all_seq[r:r+self.seq_len]
        i_label=i_all_label[r:r+self.seq_len]

        return i_input,i_label

class Dataset2(data.Dataset):
    def __init__(self,input_list,label_list,seq_len,phase="train"):
        self.input_list=input_list
        self.label_list=label_list
        self.seq_len=seq_len
        self.phase=phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        i_all_seq=self.input_list[index]
        i_all_label=self.label_list[index]

        if not len(i_all_seq)==len(i_all_label):
            print('Error: labelがずれています', file=sys.stderr)
            sys.exit(1)
        
        return i_all_seq,i_all_label

class LSTM_Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,output_size_final,seq_size):
        super(LSTM_Net,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.seq_size=seq_size

        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc1=nn.Linear(hidden_size,output_size)
        self.fc2=nn.Linear(hidden_size,output_size_final)

    
    def forward(self,x,h0,c0,mode="continue"):
        #h0=torch.zeros(batchsize,self.seq_size,self.hidden_size).permute(1, 0, 2)
        #c0=torch.zeros(batchsize,self.seq_size,self.hidden_size).permute(1, 0, 2)
        output,(h_n,c_n)=self.lstm(x,(h0,c0))
        #↓全結合層は引数で指定したtのところだけ取り出す
        out=torch.tensor([])
        if mode=="continue":
            for h_t in h_n:
                out=torch.cat((out,self.fc1(h_t)),0)
        elif mode=="final":
            for h_t in h_n:
                out=torch.cat((out,self.fc2(h_t)),0)

        return out,h_n[-1],c_n[-1]

#↓残りがseq_len以下になったseqがあれば、Falseを返す
def choice(data,label,i,seq_len):
    out_l=[]
    out_label=[]
    t_f=True
    for seq,lab in zip(data,label):
        if len(seq)>=i+seq_len:
            out_l.append(seq[i:i+seq_len])
            out_label.append(lab[i:i+seq_len])
        else:
            t_f=False

    out_l=torch.stack(out_l)
    out_label=torch.stack(out_label)

    return out_l,out_label,t_f