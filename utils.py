# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:36:23 2019
@author: puranam
"""

import os
import sys
import time
import numpy as np
import pickle
import gensim
from . import data_helpers

def create_cross_validation_fold(datapath, filename, n_splits, downsample=True):
   
    import pandas as pd
    from sklearn.model_selection import KFold 

    df = pd.read_csv(datapath + '/' + filename)
    df = df.rename(columns={"photoornot": "label"})
    
    #Downsample majority class
    if downsample==True:
        positive = df[df["label"] == 1]
        negative = df[df["label"] == 0].sample(positive.shape[0])
        df = pd.concat([positive, negative])

    #train-test split
    mask = np.random.rand(df.shape[0]) < 0.8

    train = df.loc[mask]
    train.reset_index(drop=True, inplace=True)

    test = df.loc[~mask]
    test.reset_index(drop=True, inplace=True)

    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    kf.get_n_splits(train)

    for fold, (train_index, val_index) in enumerate(kf.split(train)):
        train.loc[train_index].to_csv(datapath + '/' + 'train_fold' + str(fold) + '.csv', index=False)
        train.loc[val_index].to_csv(datapath + '/' + 'val_fold' + str(fold) + '.csv', index=False)
    
    test.to_csv(datapath + '/test.csv', index=False)    

    return

def customize_embeddings_from_pretrained_googlenews_w2v(args, vocabulary, vocabulary_inv_list, model):

    vocabulary_inv = {rank: word for rank, word in enumerate(vocabulary_inv_list)}
    embedding_weights = {}

    words = []
    for id, word in vocabulary_inv.items():

        words.append(word)

        if word in model.vocab:
            embedding_weights[id] = model.word_vec(word)
        else:
            embedding_weights[id] = np.random.uniform(-0.001, 0.001, args['embedding_dim'])

    return np.array(list(embedding_weights.values())), words

def load_pretrained_embeddings():
    print("loading vectors")
    
    pretrained_fpath_saved = os.path.expanduser("models/googlenews_extracted-python{}.pl".format(sys.version_info.major))
    if os.path.exists(pretrained_fpath_saved):
        with open(pretrained_fpath_saved, 'rb') as f:
            embedding_weights = pickle.load(f)
    else:
        print('- Error: file not found : {}\n'.format(pretrained_fpath_saved))
        print('- Please run the code "python utils.py" to generate the file first\n\n')
        sys.exit()

    # embedding_weights is a dictionary {word_index:numpy_array_of_300_dim}
    out = np.array(list(embedding_weights.values())) # added list() to convert dict_values to a list for use in python 3
    #np.random.shuffle(out)

    print('embedding_weights shape:', out.shape)
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    return out

def label_swap(x):
    """ use this to swap the label to make the minority class 1, x should be int"""
    if x==1:
        return 0
    else:
        return 1
    
def check_dir(directory):
    """ does directory exist if not make directory"""
    import os
    #directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_test_perf(d,args,dep):
    import json
    adict={}
    for kv in d.items():
        adict[kv[0]]=int(kv[1])                
    with open('Transformer_Output/'+dep+"_"+args['model_name']+"_test_perf.json", 'w') as f:
        json.dump(adict, f)
        f.close()

def import_prep(args, prefix, fname, _name, data_dir, swap_label):

    import pandas as pa

    try:
        print("Verbose Data directory: ", prefix, fname)
        train_df = pa.read_csv(prefix + fname)
        print("Positive label: ", train_df['label'].sum())

    except:
        print("importing",args['topic'],prefix+fname ,"failed")
        return

    if swap_label:
        train_df['label']= train_df['label'].map(lambda x: label_swap(x))    

    train_df.to_csv(data_dir+_name+'.csv', sep=',', index=False, header=True)
    print(data_dir+_name+'.csv')
