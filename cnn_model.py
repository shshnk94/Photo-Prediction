# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:30:44 2019
@author: puranam
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, pretrained_embeddings, args):

        super(CNN, self).__init__()
        
        self.use_cuda = args['use_cuda']
        self.kernel_sizes = args['kernel_sizes']
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(pretrained_embeddings))
        self.ConvMethod =  args['ConvMethod']

        # Cannot load the entire embedding matrix to GPU as the matrix is possibly (3000000000, 300)
        if  args['use_cuda']:
            self.embedding = self.embedding.cuda(1)

        conv_blocks = []
        for kernel_size in args['kernel_sizes']:

            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size =  args['sentence_len'] - kernel_size +1

            if  args['ConvMethod'] == "in_channel__is_embedding_dim":
                conv1d = nn.Conv1d(in_channels = args['embedding_dim'], out_channels = args['num_filters'], kernel_size = kernel_size, stride = 1)
            else:
                conv1d = nn.Conv1d(in_channels = 1, out_channels = args['num_filters'], kernel_size = kernel_size*args['embedding_dim'], stride = args['embedding_dim'])

            component = nn.Sequential(conv1d,
                                      nn.ReLU(),
                                      nn.MaxPool1d(kernel_size = maxpool_kernel_size))

            if args['use_cuda']:
                component = component.cuda()

            conv_blocks.append(component)

            if 0:
                conv_blocks.append(
                nn.Sequential(
                    conv1d,
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size = maxpool_kernel_size)
                ).cuda()
                )


        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(args['num_filters']*len(args['kernel_sizes']), args['num_classes'])



    def forward(self, x):       # x: (batch, sentence_len)

        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)

        if self.ConvMethod == "in_channel__is_embedding_dim":

            #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        else:

            #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)

        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)

        return F.softmax(self.fc(out), dim=1), feature_extracted

