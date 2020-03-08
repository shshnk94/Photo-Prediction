# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:52:10 2019
@author: puranam
"""

import torch
import pandas as pa
import torch.nn as nn
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from .sampler import get_sampler
from .cnn_model import CNN
import time
import gc
from .utils import  customize_embeddings_from_pretrained_googlenews_w2v
from .evaluate import evaluate
from . import data_helpers

np=pa.np
pa.np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

def train(cv, args, embeddings=None, should_eval=True):

    from .best_model import BestModel

    x_train, y_train,vocabulary, vocabulary_inv_list = data_helpers.load_train(args['datapath'], cv)
    args['vocab_size'] = len(vocabulary_inv_list)
    args['sentence_len'] = x_train.shape[1]
    args['num_classes'] = int(max(y_train)) +1 

    if args['use_pretrained_embeddings']:
        pretrained_embeddings, words = customize_embeddings_from_pretrained_googlenews_w2v(args, vocabulary, vocabulary_inv_list, embeddings)
    else:
        pretrained_embeddings = np.random.uniform(-0.01, -0.01, size=(args['vocab_size'], args['embedding_dim']))

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    print("Training set size: ", x_train.shape)
    dataset_train = TensorDataset(x_train, y_train)
    train_labels=[t[-1] for t in dataset_train]
    train_sampler = get_sampler(dataset_train, train_labels)
    train_loader = DataLoader(dataset_train, sampler=train_sampler,batch_size=args['batch_size'], num_workers=4, pin_memory=False)
    
    #Read the validation data here to avoid reading every epoch.
    x_test, y_test, vocabulary, vocabulary_inv_list = data_helpers.load_data(args['datapath'],
                                                                             cv,
                                                                             vocabulary, 
                                                                             vocabulary_inv_list, 
                                                                             args['sentence_len'])
   
    y_test = torch.from_numpy(y_test).long()
    x_test = torch.from_numpy(x_test).long()
    
    model = CNN(pretrained_embeddings,args)
    if args['use_cuda']:
        model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    from .early_stopping import EarlyStopping
    es = EarlyStopping(min_delta=0.0001, patience=args['earlystopping_patience'])

    bm = BestModel()
    epoch_tracker=[]

    for epoch in range(args['num_train_epochs']):

        tic = time.time()
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs), Variable(labels)
            if args['use_cuda']:
                inputs, labels = inputs.cuda(), labels.cuda()

            preds, _ = model(inputs)
            if args['use_cuda']:
                preds = preds.cuda()
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 0: # this does not improve the performance (even worse) (it was used in Kim's original paper)
                constrained_norm = 1 # 3 original parameter
                if model.fc.weight.norm().data[0] > constrained_norm:
                    model.fc.weight.data = model.fc.weight.data * constrained_norm / model.fc.weight.data.norm()
        
        sentence_vector = None
        if should_eval:
 
            eval_acc, loss_test, sentence_vector, adict = eval_dev(model, x_test, y_test, cv, args, criterion)
            adict['val_epoch'] = epoch
            bm.step(adict,epoch)
            adict.update(args)
            epoch_tracker.append(adict)
    
            with open('validation_result_lr' + str(args['learning_rate']) + '_cv' + str(cv) + '.csv', 'a') as handle:
                handle.write(','.join([str(adict['Precision']), str(adict['F1']), str(adict['Accuracy'])]) + '\n')
    
    epoch_tracker = pa.DataFrame.from_dict(epoch_tracker)
    
    if should_eval:
        return bm.adict, sentence_vector, model, args, vocabulary, vocabulary_inv_list, epoch_tracker, bm.best_epoch

    return model, args, vocabulary, vocabulary_inv_list
    
def eval_dev(model, x_test, y_test, cv, args, criterion):
    
    if args['use_cuda']:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    
    eval_acc, loss_test,sentence_vector, adict = evaluate(model, x_test, y_test, args, criterion, 'test' if cv is None else 'val')
    
    return eval_acc,  loss_test,sentence_vector, adict
