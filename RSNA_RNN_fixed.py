import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torchvision import models,transforms
import torch.optim as optim
import torch.nn.functional as F
#%matplotlib inline
from matplotlib import pyplot as plt
import random
import pickle
import json
from PIL import Image
import filecmp

import bottleneck_utils as bu
import net_utils as nu


def main(args):
#####転移学習によりボトルネック作成#####
    if args.bottleneck:
        bottleneck_net=torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)

        resize=299
        ######↓今回の画像データはグレースケールなのでチャネル数が一個であることに要注意#####
        mean=[0.485]
        std=[0.229]

        inputs,img_num,indivisual_count_list,label_list=bu.make_inputs(resize,mean,std)
        print("inputサイズ：",inputs.shape)
        ###################↓グレースケールの画像の場合、チャンネル数の次元をunsqueezeで足さないといけない#####
        out=bottleneck_net(inputs)[0]

        try:
            print("try実行中.....")
            with open("file.binaryfile","rb") as f:
                bottleneck_data=pickle.load(f)
                print("bottleneckは既に存在")
                if len(bottleneck_data)>=img_num:
                    with open("file_sub.binaryfile","wb") as f:
                        pickle.dump(out,f)
                        print("file_sub.binaryfileに一応保存")
                else:
                    print("bottleneckを破棄します")
            
        except:
            with open("file.binaryfile","wb") as f:
                pickle.dump(out,f)
            print("bottleneckを新たに保存")

#####bottleneck_dataのデータ構造を整理する#####
    if args.cleaning:
        #seq_len=5

        input_list,label_tensor=bu.separate_bottleneck(out,label_list,indivisual_count_list)
        train_input,val_input,train_label,val_label=nu.train_val_split(input_list,label_tensor)

        train_dataset=nu.Dataset2(train_input,train_label,seq_len,phase="train")
        val_dataset=nu.Dataset2(val_input,val_label,seq_len,phase="val")

       
#####LSTMモデル作成#####
    if args.learning:
        #↓一度に考慮する患者数
        batch_size=30
        #train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        #val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

        #dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

        input_size=2048
        hidden_size=10
        #↓pe_on_imageのための出力
        output_size=1
        output_size_final=9
        seq_size=10
        net=nu.LSTM_Net(input_size,hidden_size,output_size,output_size_final,seq_size)
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(net.parameters(),lr=0.01)

        n=0

        while 1:
            train=train_input[n:n+batch_size]
            label=train_label[n:n+batch_size]
            h0=torch.zeros(batch_size,seq_size,hidden_size).permute(1, 0, 2)
            c0=torch.zeros(batch_size,seq_size,hidden_size).permute(1, 0, 2)
            t_f=True

            i=0

        ####十個ずつバッチでRNNに入れるが、最後はseq_lenが不揃いなため、seq一個ずつ学習させないといけない
            while t_f:
                X_train,y_train,t_f=nu.choice(train,train_label,i,seq_size)
                train_loss=0
                net.train()
                optimizer.zero_grad()

                outputs,h0,c0=net(X_train,h0,c0,mode="continue")
                
                for label in y_train:
                    try:
                        loss+=criterion(outputs,label[0])
                    except:
                        loss=criterion(outputs,label[0])
                train_loss+=loss.item()

                loss.backward(retain_graph=True)
                optimizer.step()

                i+=1

        ####↓imageレベル、examレベルのラベルを同時に出力しないといけない
            index=0
            for seq_,label_ in zip(train,train_label):
                seq=seq_[i+seq_size]
                label=label_[i+seq_size]

                outputs,h0,c0=net(seq,h0[index],c0[index],mode="final")
                
                for label in y_train:
                    try:
                        loss+=criterion(outputs,label)
                    except:
                        loss=criterion(outputs,label)
                train_loss+=loss.item()

                loss.backward(retain_graph=True)
                optimizer.step()





if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("-bottleneck","--bottleneck",action="store_true")
    parser.add_argument("-cleaning","--cleaning",action="store_true")
    parser.add_argument("-learning","--learning",action="store_true")
    args=parser.parse_args()

    main(args)