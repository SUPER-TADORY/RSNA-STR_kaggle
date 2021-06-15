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

def make_datapath_list(phase="train"):
    img_file_path="/home/mshirota/kaggle/RSNA-STR/jpeg"
    train_csv_path="/home/mshirota/kaggle/RSNA-STR/train.csv"
    test_csv_path="/home/mshirota/kaggle/RSNA-STR/test.csv"
    file_list=[]
    label_list=[]
    #↓それぞれの患者について、何枚の写真が用意されているか{first_id:枚数}
    indivisual_count_list={}

    train_csv=pd.read_csv(train_csv_path)
    first_id_list = train_csv.StudyInstanceUID
    second_id_list = train_csv.SeriesInstanceUID
    third_id_list=train_csv.SOPInstanceUID
    n=len(third_id_list)
    #for id in third_id_list:
    for count, ( first_id, second_id, third_id ) in enumerate( zip( first_id_list, second_id_list, third_id_list ) ):
        path=os.path.join(img_file_path, first_id, second_id, third_id + '.jpeg' )
        #target_path=os.path.join(img_file_path+f"/**/**/{id}.jpeg")
        #print(glob.glob(target_path))
        if os.path.exists(path):
            #################↓globで検索すると、該当パスのリストで返されるので要注意######
            #path=glob.glob(target_path)[0]
            file_list.append(path)
            #first_id=path.split("/")[-3]
            ind=train_csv.loc[train_csv.SOPInstanceUID==third_id]
            label=torch.tensor([ind.negative_exam_for_pe.values[0],ind.indeterminate.values[0],ind.chronic_pe.values[0],ind.acute_and_chronic_pe.values[0],ind.central_pe.values[0],ind.leftsided_pe.values[0],ind.rightsided_pe.values[0],ind.rv_lv_ratio_gte_1.values[0],ind.rv_lv_ratio_lt_1.values[0],ind.pe_present_on_image.values[0]])
            #l_=df_.values.tolist()
            #print("label is",label)
            label_list.append(label)
            indivisual_count_list[first_id]=indivisual_count_list.get(first_id,0)+1

        else:
            print(f"{path}は存在しないです")
        
        if count%100==0:
            print("making datapath list",count*100/n,"%")

        count+=1
        
        #print("indivisual_count_list is",indivisual_count_list)

        
        if count==200:
            break
            

    return file_list,label_list,n,indivisual_count_list


class BaseTransform():
    def __init__(self,resize,mean,std):
########↓データ前処理はtrainとvalで異なる場合は辞書方にして分ける#####
        self.base_transform={
            "train":transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
        #################↓グレースケールからRGBへの変換はtransform.Grayscale()で変換できる#####
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ]),
            "val":transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        }
    
    def __call__(self,img,phase="train"):
        return self.base_transform[phase](img)

def make_inputs(resize,mean,std,phase="train"):
    transform=BaseTransform(resize,mean,std)
    file_list,label_list,n,indivisual_count_list=make_datapath_list()

    inputs=torch.tensor([])
    n=len(file_list)
    count=0

    #print("file list is",file_list)
    for path in file_list:
        img=Image.open(path)
        img_transformed=transform(img,phase)
        img_transformed=torch.unsqueeze(img_transformed,0)
        inputs=torch.cat((inputs,img_transformed),0)

        if count%100==0:
            print("making inputs",count*100/n,"%")
        count+=1
    
    return inputs,n,indivisual_count_list,label_list


#bottleneckを患者毎に分ける
def separate_bottleneck(bottleneck_tensor,label_list,indivisual_count_list):
    train_csv_path="/home/mshirota/kaggle/RSNA-STR/train.csv"
    test_csv_path="/home/mshirota/kaggle/RSNA-STR/test.csv"
    train_csv=pd.read_csv(train_csv_path)

    input_list=[]
    label_tensor=[]
    count=0
    n=len(bottleneck_tensor)
    for id,value in indivisual_count_list.items():
        i_list=[]
        l_i_list=[]
        for _ in range(value):
            i_bottleneck=next(iter(bottleneck_tensor))
            i_label=next(iter(label_list))
            i_list.append(i_bottleneck)
            l_i_list.append(i_label)
            count+=1
        
        i_list=torch.stack(i_list)
        l_i_list=torch.stack(l_i_list)
        input_list.append(i_list)
        label_tensor.append(l_i_list)

        if count%100==0:
                print("separating bottleneck",count*100/n,"%")
    """
    first_id_series=train_csv.StudyInstanceUID.unique()
    index=0
    i_list=[]
    id=next(iter(first_id_series))
    for row in train_csv:
        if row.StudyInstanceUID==id:
            i_list.append(bottleneck_tensor[index])
        else:
            input_list.append(i_list)
            i_list=[]
            id=next(iter(first_id_series))
            i_list.append(bottleneck_tensor[index])
        index+=1
    """

    return input_list,label_tensor




