import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import pydicom
import shutil
import cv2
import random
from argparse import ArgumentParser

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


#↓PE_TRUEの分布関数作成
def location(input_list,which_dir,tr,te,val):
    N=len(input_list)
    train_distribution_list=tr
    test_distribution_list=te
    val_distribution_list=val

    for i in range(N):
        if which_dir=="train":
            train_distribution_list[int((i/N)//0.05) if not i==N else 19]+=input_list.iloc[i]
        elif which_dir=="test":
            test_distribution_list[int((i/N)//0.05) if not i==N else 19]+=input_list.iloc[i]
        elif which_dir=="val":
            val_distribution_list[int((i/N)//0.05) if not i==N else 19]+=input_list.iloc[i]
    
    return train_distribution_list,test_distribution_list,val_distribution_list


#↓分布リストから、累積分布関数を作る
def accumulate(distribution_list):
    sum=0
    l=distribution_list.copy()
    for i,o in enumerate(distribution_list):
        sum+=o
        l[i]=sum
    
    #l=l.map(lambda x:x/max(l))
    l=[x/max(l) for x in l]

    return l


def which_x(accum_list,x_list):
    r=random.random()
    #####↓indexは0~1で標準化しないといけないのに、そのまま使ってしまった。
    index=which_bin(accum_list,r)/20
    print(index)
    #print(x_list)
    x_l=[x for x in x_list if x>=index-0.05 and x<index+0.05]
    try:
        one_value=random.choice(x_l)
        index__=x_list.index(one_value)
        return x_list[index__]
    except:
        return False

def which_bin(accum_list,num):
    l_=accum_list
    out_index=0

    #↓二分探索法
    while 1:
        n=len(l_)
        if n==1:
            break
        if l_[(n//2)-1]>num:
            l_=l_[:(n//2)]
        else:
            l_=l_[n//2:]
    
    if num>l_[0]:
        out_index=accum_list.index(l_[0])+1
    elif num<=l_[0]:
        out_index=accum_list.index(l_[0])
    
    return out_index


def processing(inputfile,input_dir,output_dir,jobid,maxid):
    first_ID_series=inputfile.StudyInstanceUID.unique()

    train_distribution_list=[0 for _ in range(20)]
    val_distribution_list=[0 for _ in range(20)]
    test_distribution_list=[0 for _ in range(20)]

    train_pe_count=0
    test_pe_count=0
    val_pe_count=0

    false_list={}
    all_false_list={}
    input_out={}

    train_false_count=0
    test_false_count=0
    val_false_count=0

    for i,first in enumerate(first_ID_series):
        """
        if i%maxid != jobid:
            #####↓continueするときも、indexは更新しないといけないことに要注意!!!!!#####
            print("skip")
            continue
        """

        individual_list=inputfile.loc[(inputfile.StudyInstanceUID==first) & (inputfile.indeterminate==0)]
        pe_or_not_list=individual_list.pe_present_on_image
        second_ID_list=individual_list.SeriesInstanceUID
        third_ID_list=individual_list.SOPInstanceUID


        r = random.random()
        if r < 0.15:
            odir = "val"
        elif r>=0.15 and r<0.80:
            odir = "train"
        else:
            odir = "test"

        train_distribution_list,test_distribution_list,val_distribution_list=location(pe_or_not_list,odir,train_distribution_list,test_distribution_list,val_distribution_list)
        print("頻度",train_distribution_list)

        
        for id in range(len(second_ID_list)):
            """
            if list_[id]==0:
                continue
            """
            oodir="PE_TRUE" if pe_or_not_list.iloc[id]==1 else "PE_FALSE"

            if oodir=="PE_TRUE":
                if odir=="train":
                    train_pe_count+=1
                elif odir=="test":
                    test_pe_count+=1
                elif odir=="val":
                    val_pe_count+=1

                if i%100:
                    print("訓練",train_pe_count,"テスト",test_pe_count,val_pe_count)
            
                """
                dcm_sample = pydicom.dcmread(os.path.join(input_dir,first,second_ID_list[id],f"{third_ID_list[id]}.dcm"))
                dcm_img = dcm_sample.pixel_array 
                cv2.imwrite(os.path.join(output_dir,odir,oodir,third_ID_list[id])+".jpeg", dcm_img)
                """
                input_out[os.path.join(input_dir,first,second_ID_list.iloc[id],f"{third_ID_list.iloc[id]}.dcm")]=[os.path.join(output_dir,odir,oodir,third_ID_list.iloc[id])+".jpeg","True"]
            
            else:
                #false_list[f"{first}/{second_ID_list[id]}"]=false_list.get(f"{first}/{second_ID_list[id]}",[]).append({id/len(second_ID_list):[odir,oodir,third_ID_list[id]])
                #false_list[odir]=false_list.get(odir,{})[f"{first}/{second_ID_list[id]}"].get(f"{first}/{second_ID_list[id]}",{})[id/len(second_ID_list)]=[odir,oodir,third_ID_list[id]]
                if not odir in false_list.keys():
                    false_list[odir]={}
                if not f"{first}/{second_ID_list.iloc[id]}" in false_list[odir].keys():
                    false_list[odir][f"{first}/{second_ID_list.iloc[id]}"]={}
    
                false_list[odir][f"{first}/{second_ID_list.iloc[id]}"][id/len(second_ID_list)]=[odir,oodir,third_ID_list.iloc[id]]
                p=os.path.join(input_dir,f"{first}/{second_ID_list.iloc[id]}",third_ID_list.iloc[id]+".dcm") in input_out.keys()
                if odir=="train":
                    train_false_count+=1
                elif odir=="test":
                    test_false_count+=1
                elif odir=="val":
                    val_false_count+=1
                #print(p)
                #print(false_list[odir])

        print("complete:",(i/1800000)*100,"%")
        #if i==100:
        #   break
    
    print("true:false=",train_pe_count,":",train_false_count)
    print(len(false_list["train"]))

    for key,dict_ in false_list.items():
        if key=="train":
            l=accumulate(train_distribution_list)
            c=train_pe_count
        elif key=="test":
            l=accumulate(test_distribution_list)
            c=test_pe_count
        elif key=="val":
            l=accumulate(val_distribution_list)
            c=val_pe_count

        print(l)
        num=0
        while num<=c:
        #for key2,dict2 in dict_.items():
            #print(dict_.keys())
            key2=random.choice(list(dict_.keys()))
            #print(key2)
            
            x_list=list(dict_[key2].keys())
            if which_x(l,x_list)==False:
                print("該当数なし")
                continue
            path_list=dict_[key2][which_x(l,x_list)]
            """
            key3=random.choice(list(dict_[key2].keys()))
            path_list=dict_[key2][key3]
            """
            #print(path_list[2])
            if os.path.join(input_dir,key2,f"{path_list[2]}.dcm") in input_out.keys():
                #print("重複あり")
                continue
            """
            dcm_sample = pydicom.dcmread(os.path.join(input_dir,key2,f"{path_list[2]}.dcm"))
            dcm_img = dcm_sample.pixel_array
            cv2.imwrite(os.path.join(output_dir,path_list[0],path_list[1],path_list[2])+".jpeg", dcm_img)
            """
            input_out[os.path.join(input_dir,key2,f"{path_list[2]}.dcm")]=[os.path.join(output_dir,path_list[0],path_list[1],path_list[2])+".jpeg","False"]
            num+=1
            print(num,"足した")

    print(1)
    print(len(input_out.keys()))

    count=0
    a=0
    b=0
    for key,value in input_out.items():
        """
        if count%maxid != jobid:
            count+=1
            continue
        """
        
        try:
            dcm_sample = pydicom.dcmread(key)
            dcm_img = dcm_sample.pixel_array
            cv2.imwrite(value[0], dcm_img)
            if value[1]=="True":
                a+=1
            else:
                b+=1
            print("True:",a,"False",b,"成功")
            count+=1
        except:
            count+=1
            print(b,"失敗")
            continue


def main():
    """
    l=["splitted_train","splitted_trainpe_true" ,"splitted_validpe_false","test","testpe_true","splitted_trainpe_false","splitted_valid","splitted_validpe_true","testpe_false"]
    for i in l:
        if os.path.exists(os.path.join("/home/fmhc006/kaggle_RSNA",i)):
            shutil.rmtree(os.path.join("/home/fmhc006/kaggle_RSNA",i))
    """

    #↓１:trainをtrainとvalに分ける
    basepath="/home/mshirota/kaggle/RSNA-STR/"
    to_basepath="/home/fmhc006/kaggle_RSNA/input_data_fixed/"
    train=pd.read_csv(basepath+"train.csv")
    #test=pd.read_csv(basepath+"test.csv")
    
    
    if os.path.exists(to_basepath):
        shutil.rmtree(to_basepath)
    

    #pe_or_not_list_test=test.pe_present_on_image

    #↓必要なディレクトリを作成する
    create_directory(to_basepath+"train")
    create_directory(to_basepath+"val")
    create_directory(to_basepath+"test")
    for p in os.listdir(to_basepath):
        #print(p)
        create_directory(to_basepath+p+"/PE_TRUE")
        create_directory(to_basepath+p+"/PE_FALSE")

    input_dir_train=os.path.join(basepath+"train")
    #input_dir_test=os.path.join(basepath+"test")
    output_dir=os.path.join(to_basepath)

    processing(train,input_dir_train,output_dir,args.jobid,args.maxid)
    #trian_val_split(inputLS_dir_test,output_dir,pe_or_not_list_test,True)


if __name__=="__main__":
    
    parser=ArgumentParser()
    parser.add_argument("jobid",type=int,help="JOB ID")
    parser.add_argument("--maxid",type=int,help="MAX JOB ID",default=10)
    args=parser.parse_args()
    
    main()