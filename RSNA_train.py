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

        
def processing(inputfile,input_dir,output_dir,jobid,maxid):
    pe_or_not_list_train=inputfile.pe_present_on_image
    first_ID_list=inputfile.StudyInstanceUID
    second_ID_list=inputfile.SeriesInstanceUID
    third_ID_list=inputfile.SOPInstanceUID

    #print(pe_or_not_list_train)

    index=0
    for i,first in enumerate(first_ID_list):
        if i%maxid != jobid:
            #####↓continueするときも、indexは更新しないといけないことに要注意!!!!!#####
            index+=1
            print("skip")
            continue
        r = random.random()
        oodir="pe_true" if pe_or_not_list_train.loc[index]==1 else "pe_false"
        if r < 0.15:
            odir = "splitted_train"
        elif r>=0.15 and r<0.80:
            odir = "splitted_valid"
        else:
            odir = "test"

        second=second_ID_list[index]
        third=third_ID_list[index]

        index+=1

        try:
            dcm_sample = pydicom.dcmread(os.path.join(input_dir,first,second,f"{third}.dcm"))
            dcm_img = dcm_sample.pixel_array 
            cv2.imwrite(os.path.join(output_dir,odir,oodir,third)+".jpeg", dcm_img)
        except:
            print("skip__")
            continue


        print("complete:",(index/1800000)*100,"%")


    """
    else:
        index=0
        for fn in os.walk( input_dir )[2]:
            r = random.random()
            oodir="pe_true" if list_.loc[index] else "pe_false"
            odir = "test"
            
            dcm_sample = pydicom.dcmread(fn)
            dcm_img = dcm_sample.pixel_array 
            cv2.imwrite(os.path.join(output_dir,odir,oodir)+".jpeg", dcm_img)
    """


#まず、train,val,testに分けて指定したディレクトリに入れる
#次に、train,valのなかで、pe,peでないの２分類をしないといけない
#最後に、dcmファイルをjpegにしないといけない
def main():
    """
    l=["splitted_train","splitted_trainpe_true" ,"splitted_validpe_false","test","testpe_true","splitted_trainpe_false","splitted_valid","splitted_validpe_true","testpe_false"]
    for i in l:
        if os.path.exists(os.path.join("/home/fmhc006/kaggle_RSNA",i)):
            shutil.rmtree(os.path.join("/home/fmhc006/kaggle_RSNA",i))
    """

    #↓１:trainをtrainとvalに分ける
    basepath="/home/mshirota/kaggle/RSNA-STR/"
    to_basepath="/home/fmhc006/kaggle_RSNA/input_data/"
    train=pd.read_csv(basepath+"train.csv")
    #test=pd.read_csv(basepath+"test.csv")
    
    
    if os.path.exists(to_basepath):
        shutil.rmtree(to_basepath)
    

    #pe_or_not_list_test=test.pe_present_on_image

    #↓必要なディレクトリを作成する
    create_directory(to_basepath+"splitted_train")
    create_directory(to_basepath+"splitted_valid")
    create_directory(to_basepath+"test")
    for p in os.listdir(to_basepath):
        #print(p)
        create_directory(to_basepath+p+"/pe_true")
        create_directory(to_basepath+p+"/pe_false")

    input_dir_train=os.path.join(basepath+"train")
    #input_dir_test=os.path.join(basepath+"test")
    output_dir=os.path.join(to_basepath)
    processing(train,input_dir_train,output_dir,args.jobid,args.maxid)
    #trian_val_split(input_dir_test,output_dir,pe_or_not_list_test,True)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("jobid",type=int,help="JOB ID")
    parser.add_argument("--maxid",type=int,help="MAX JOB ID",default=10)
    args=parser.parse_args()

    main()