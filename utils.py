import pandas as pd
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn import metrics
import numpy as np

def expandData(data,res):
    for key in res.keys():
        data[key]=res[key]

def transform(seq):
    mol=Chem.MolFromSequence(seq)
    smi=Chem.MolToSmiles(mol)
    return smi
def smiIsLegal(smi):
    mol=Chem.MolFromSmiles(smi)
    if (str(mol)!='None'):return True
    return False
def seqIsLegal(seq):
    if '-' in seq:return False
    mol=Chem.MolFromSequence(seq)
    if (str(mol)!='None'):return True
    return False
    
def filterData(df,smiField="SMILES",seqField="SEQUENCE"):
    smi_legal_index=[]
    for index,smi in enumerate(df[smiField]):
        if (smiIsLegal(smi)):
            smi_legal_index.append(index)
    print("{}合法的数据有{}个,非法数据如下===============".format(smiField,len(smi_legal_index)))
    for index,smi in enumerate(df[smiField]):
        if index not in smi_legal_index:
            print(smi)
            print('--------------------------------------------')
            
    print("=======================================================")
    

    seq_legal_index=[]
    for index,seq in enumerate(df[seqField]):
        if (seqIsLegal(seq)):
            seq_legal_index.append(index)
    print("{}合法的数据有{}个,非法数据如下===============".format(seqField,len(seq_legal_index)))
    
    for index,seq in enumerate(df[seqField]):
        if index not in seq_legal_index:
            print(seq)
            print('--------------------------------------------')
            
    print("=======================================================")
    legal_index=[index for index in smi_legal_index if index in seq_legal_index]

    print("{}合法及{}合法的数据有{}个,共删除{}个数据".format(smiField,seqField,len(legal_index),len(df)-len(legal_index)))

    return legal_index
    
def seqToUpper(seqList):
    for i in range(len(seqList)):
        seqList[i]=seqList[i].upper()
    return seqList


def myMetrics(y_true, y_pred):
    y_true = np.squeeze(np.asarray(y_true))
    y_pred = np.squeeze(np.asarray(y_pred))
    print(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    mse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=True)
    
    r2 = metrics.r2_score(y_true, y_pred)
    
    r = pearsonr(y_true, y_pred)[0]
    
    return r, r2, mae, mse, rmse

def hasNull(df):
    for line in range(df.shape[0]):
        if pd.isnull(df.iloc[line,:]).any():
            print(df.iloc[line,:])