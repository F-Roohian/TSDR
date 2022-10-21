#preproces - create input dataset for neural network
import pandas as pd
import csv
import numpy as np
import Hellper as h
import random
from random import Random

#path saving
path_saving=[]
path_saving+=['Input_TSDR_drug_dis.data']
path_saving+=['Input_TSDR_drug_pubchem.data']
path_saving+=['Input_TSDR_drug_target_domain.data']
path_saving+=['Input_TSDR_drug_target_go.data']
path_saving+=['Input_TSDR_drugs_BiowordVec.data'] 

#path dataset
path_drug=[]
path_drug+=['drug_dis.txt']
path_drug+=['drug_pubchem.txt']
path_drug+=['drug_target_domain.txt']
path_drug+=['drug_target_go.txt']
path_drug+=['BioWordVec_features-763.csv'] 
path_disease_feature='disease_integrated_similarity.txt'
path_lable='dis_drug.txt'

#pairs disease-drug
whole_positive_index , negative_sample_index = h.pairs_disease_drug()

#create input dataset
for index_feture in range(5):
    
    path_drug_feature=path_drug[index_feture]
    path_lable='dis_drug.txt'
    path_pre_data=path_saving[index_feture]
    
    if index_feture==4:
        drug_feture=h.load(path_drug_feature) #for  BioWordVec_features
    else:
        drug_feture=pd.read_csv(path_drug_feature, header=None, sep='\t', encoding='utf-8')
    lable = pd.read_csv(path_lable, header=None, sep='\t', encoding='utf-8').to_numpy()
    disease_feature=pd.read_csv(path_disease_feature, header=None, sep='\t', encoding='utf-8').to_numpy()

    #if no BioWordVec features
    if index_feture!=4:
        drug_feture=pd.read_csv(path_drug_feature, header=None, sep='\t', encoding='utf-8')
        drug_feture=drug_feture.drop(0, axis=0)
        drug_feture=drug_feture.drop(columns=0)
        #convert to numpy
        drug_feture=drug_feture.to_numpy()
        drug_feture=drug_feture.astype(float)
    pos_data_set=[]
    for i in whole_positive_index :
                d_row=[]
                d_row=np.concatenate((disease_feature[i[0]], drug_feture[i[1]], [lable[i[0],i[1]]]))
                pos_data_set.append(d_row)
    neg_data_set=[]
    for i in negative_sample_index :
                d_row=[]
                d_row=np.concatenate((disease_feature[i[0]], drug_feture[i[1]], [lable[i[0],i[1]]]))
                neg_data_set.append(d_row)
    #training dataset   
    SEED=4
    dataset=pos_data_set+neg_data_set
    Random(SEED).shuffle(dataset)

    h.save(path_saving[index_feture],dataset)
    print(f'creating  input dataset-{index_feture+1}')