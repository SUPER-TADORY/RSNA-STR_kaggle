import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torch
import torch.nn as nn
#import torchvision
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as transforms
#%matplotlib inline
from matplotlib import pyplot as plt
import random


#↓ボトルネックデータの中でデータ区分してpathを辞書に入れておく
def mk_data_dict(train,basepath):
    data_dict={"train":{},"test":{},"val":{}}
    for category in os.listdir(basepath):
        print(category)
        category_path=os.path.join(basepath,category)
        filelist_in_category=[]
        for _ , _ , files in os.walk(category_path):
            for f in files:
                filelist_in_category.append(os.path.basename(f))
        
        for filename in filelist_in_category:
            third_ID=filename.split(".")[0]
            print(third_ID)
            ######↓あくまでも、SOPInstanceUIDがthird_IDである行を取り出していることに要注意！！！#######
            first_ID=train.loc[train.SOPInstanceUID==third_ID].StudyInstanceUID.values[0]
            print(first_ID)
            index=train.loc[train.StudyInstanceUID==first_ID].query('SOPInstanceUID=="{}"'.format(third_ID)).index[0]
            print(index)
            if not first_ID in data_dict[category].keys():
                ind=train.loc[train.SOPInstanceUID==third_ID]
                #df_=[ind.negative_exam_for_pe,ind.indeterminate,ind.chronic_pe,ind.acute_and_chronic_pe,ind.central_pe,ind.leftsided_pe,ind.rightsided_pe,ind.rv_lv_ratio_gte_1,ind.rv_lv_ratio_lt_1,ind.pe_present_on_image]
                #l_=df_.values.tolist()
                df_=ind.pe_present_on_image.values
                data_dict[category][first_ID]={index:[os.path.join(basepath,category,filename),df_]}
            else:
                ind=train.loc[train.SOPInstanceUID==third_ID]
                #df_=[ind.negative_exam_for_pe,ind.indeterminate,ind.chronic_pe,ind.acute_and_chronic_pe,ind.central_pe,ind.leftsided_pe,ind.rightsided_pe,ind.rv_lv_ratio_gte_1,ind.rv_lv_ratio_lt_1,ind.pe_present_on_image]
                #l_=df_.values.tolist()
                df_=ind.pe_present_on_image.values
                data_dict[category][first_ID][index]=[os.path.join(basepath,category,filename),df_]

    return data_dict


class LSTM_Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(model,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)

    
    def forward(self,x,h0,c0):
        output,(h_n,c_n)=self.lstm(x,(h0,c0))
        #↓全結合層は引数で指定したtのところだけ取り出す
        out=torch.tensor([])
        for h_t in h_n:
            out=torch.cat((out,self.fc(h_t)),0)

        return out

class Attention_for_exam(nn.Module):
    def __init__(self):
        pass


#↓各Seriesの中から一定長のテンソルを抽出する(バッチ内で長さを揃えるため)
def batch_seq_select(bottleneck_tensor,seq_size):
    r=random.randint(0,len(bottleneck_tensor)-seq_size)
    
    return bottleneck_tensor[r:r+seq_size]


#↓データをバッチ化する
def data_batch(data_dict,seq_size):
    train_Xy_l=[[],[]]
    val_Xy_l=[[],[]]
    test_Xy_l=[[],[]]
   
    for category,series_dict in data_dict.items():
        series_index=0
        for series,indivisual_dict in series_dict.items():
            if category=="train":
                train_Xy_l[0].append([])
                train_Xy_l[1].append([])
            elif category=="test":
                test_Xy_l[0].append([])
                test_Xy_l[1].append([])
            elif category=="val":
                val_Xy_l[0].append([])
                val_Xy_l[1].append([])

            key_list=list(indivisual_dict.keys()).sort()
            for key in key_list:
                l=indivisual_dict[key]
                f=open(l[0],'r', encoding='UTF-8' )
                bottleneck_tensor_str=f.read()
                bottleneck_tensor=bottleneck_tensor_str.split(",")
                bottleneck_tensor=batch_seq_select(bottleneck_tensor)
                if category=="train":
                    train_Xy_l[0][series_index].append(bottleneck_tensor)
                    train_Xy_l[1][series_index].append(bottleneck_tensor[1])
                elif category=="test":
                    test_Xy_l[0][series_index].append(bottleneck_tensor)
                    test_Xy_l[1][series_index].append(bottleneck_tensor[1])
                elif category=="val":
                    val_Xy_l[0][series_index].append(bottleneck_tensor)
                    val_Xy_l[1][series_index].append(l[1])
    
    return torch.tensor(train_Xy_l),torch.tensor(val_Xy_l),torch.tensor(test_Xy_l)
            

def main():
    basepath="/home/fmhc006/kaggle_RSNA/bottleneck"
    train_=pd.read_csv("/home/mshirota/kaggle/RSNA-STR/train.csv")

    data_dict=mk_data_dict(train_,basepath)

    hidden_size=5
    input_size=2048
    output_size=1
    seq_size=50
    
    net=LSTM_Net(input_size,hidden_size,output_size)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)

    num_epochs=50
    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]

    for epoch in range(num_epochs):
        #↓データをinputできる形に
        train,val,test=data_batch(data_dict,seq_size)

        train_loss=0
        train_acc=0
        val_loss=0
        val_acc=0

        net.train()
        for i,(bottleneck_tensor,labels) in enumerate(train):
            optimizer.zero_grad()
            outputs=net(bottleneck_tensor,t)
            loss=criterion(outputs,labels)
            train_loss+=loss.item()
            loss.backward()
            optim.step()

        avg_train_loss=train_loss/len(train)
        print('Epoch[{}/{}],Loss:{loss:.4f},Perplexity:{perp:5.2f}'.format(epoch+1,num_epochs,loss=avg_train_loss,perp=np.exp(avg_train_loss)))
        
        
if __name__=="__main__":
    main()
