import os
import sys
import numpy as np
import pandas as import pd
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
%matplotlib inline
from matplotlib import pyplot as plt

#↓ボトルネックデータの中でデータ区分してpathを辞書に入れておく
def mk_data_dict(train,basepath):
    data_dict={"train":{},"test":{},"val":{}}
    for category in os.listdir(basepath):
        category_path=os.path.join(basepath,category)
        filelist_in_category=[]
        for _ , _ , files in os.walk(category_path):
            for f in files:
                filelist_in_category.append(os.path.basename(f))
        
        for filename in filelist_in_category:
            third_ID=filename.split(".")[0]
            first_ID=train.loc[train.SOPInstanceUID==third_ID].StudyInstanceUID
            index=train.loc[train.StudyInstanceUID==first_ID].query('SOPInstanceUID==third_ID').index
            if not first_ID in data_dict[category].keys():
                data_dict[category][first_ID]={index:[os.path.join(basepath,category,filename),train.loc[train.SOPInstanceUID==third_ID]]}
            else:
                data_dict[category][first_ID][index]=[os.path.join(basepath,category,filename),train.loc[train.SOPInstanceUID==third_ID]]

    return data_dict

class Net(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.block_a=nn.LSTMCell(input_size,hidden_size,batch_first=True)
        self.block_b=nn.LSTMCell(hidden_size,hidden_size,batch_first=True)
    
    def forward(self,x,hx_a0,cx_a0,hx_b0,cx_b0):
        hx_a1,cx_a1=self.block_a(x,)(hx_a0,cx_a0))
        hx_b1,cx_b1=self.block_a(hx_a1,)(hx_b0,cx_b0))

        return hx_a1,cx_a1,hx_b1,cx_b1


def main():
    basepath="/home/fmhc006/kaggle_RSNA/bottle_neck"
    train=pd.read_csv("/home/mshirota/kaggle/RSNA-STR/train.csv")

    data_dict=mk_data_dict(train,basepath)
    
    net=Net()
    hx_a=torch.randn(batch_size,hidden_size)
    cx_a=torch.randn(batch_size,hidden_size)
    hx_b=torch.randn(batch_size,hidden_size)
    cx_b=torch.randn(batch_size,hidden_size)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)

    num_epochs=50
    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]

    for epoch in range(num_epochs):
        train_loss=0
        train_acc=0
        val_loss=0
        val_acc=0

        net.train()
        
        


if __name__=="__main__":
    main()
