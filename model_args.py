# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:00:39 2019

@author: puranam
"""

args={
      'datapath':'./data',
      'sentence_len':50,
      'num_classes':2,
      'n_splits':5, 
      'vocab_size':5000,
      'kernel_sizes': [3,4,5], 
      'num_filters':100,
      'embedding_dim': 300,
      'batch_size':64,
      'use_cuda':True,
      'use_pretrained_embeddings':True,
      'mode': 'nonstatic',# 'nonstatic'
      'ConvMethod':"in_channel__is_embedding_dim",# "in_channel__is_embedding_dim",#"in_channel__is_1"
      'learning_rate':1e-4,
      'cross_val': True,
      'pretrained_embedding_fpath': "./data/GoogleNews-vectors-negative300.bin.gz"
      }
