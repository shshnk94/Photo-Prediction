# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:34:35 2019
@author: puranam
"""

import torch, torch.nn as nn
import pandas as pa
import gc
import shutil
import os
import time
import gensim
from train import train, eval_dev
from BestModel import get_best_epoch

np=pa.np
pa.np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

def run_cross_val(args, device):

    embeddings = None
    if args['use_pretrained_embeddings']:
        tic = time.time()
        print("time start: ",tic)
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(args['pretrained_embedding_fpath'], binary=True)
        print('Please wait ... (it could take a while to load the file : {})'.format(args['pretrained_embedding_fpath']))
        print('Done.  (time used: {:.1f}s)\n'.format(time.time()-tic))
    
    cv_results=[]
    cvmax = args['n_splits'] if args['cross_val'] else 1

    learn_epochs=[]    
    ep_tracker=[]
    restart=0
    count=0

    for cv in range(cvmax):

        adict, sentence_vector, model, args, vocabulary, vocabulary_inv, epoch_tracker, epoch_ = train(cv, args, embeddings, True)

        learn_epochs.append(epoch_)
        ep_tracker.append(epoch_tracker)
        adict.update(args)
        cv_results.append(adict)
                    
        if args['do_test']:

            if args['cross_val']:
                num_epochs,adict = get_best_epoch(ep_tracker)
                adict.update(args)
                cv_results.append(adict)
                
            model, args, vocabulary, vocabulary_inv = train(cv, args, embeddings, False) 
            criterion = nn.CrossEntropyLoss()
            eval_acc, loss_test, sentence_vector, adict = eval_dev(model, None, args, criterion, vocabulary, vocabulary_inv)
            adict.update(args)
                
        cv_results.append(adict)
        torch.cuda.empty_cache()
        gc.collect()
            
    return cv_results
