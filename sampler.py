# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:31:06 2019
@author: puranam
"""

def get_sampler(trainset, labels):
    from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset,WeightedRandomSampler)
    import torch
    import pandas as pa

    np=pa.np
    pa.np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    
    targets = labels
    class_count = pa.np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler
