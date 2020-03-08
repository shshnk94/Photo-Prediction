# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:05:47 2019

@author: puranam
"""

import torch


from torch.autograd import Variable
def val_performance(y_test,y_pred,y_probs,key='UNK',eval_type='val', verbose= False):
    adict={}
    import pandas as pa
    y_probs=y_probs[:,1]
    #print(len(y_test), len(y_pred))
    from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,precision_recall_curve ,roc_curve,auc
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    area = auc(recall, precision)
  
    adict['PR AUC'] = area
    precision, recall, thresholds = roc_curve(y_test, y_probs)
    area = auc(precision, recall)
    adict['ROC AUC'] = area
    a = accuracy_score( y_test ,y_pred) 
    p = precision_score( y_test ,y_pred) 
    r = recall_score( y_test ,y_pred) 
    f1 = f1_score( y_test ,y_pred)
    m=  matthews_corrcoef( y_test ,y_pred, sample_weight=None)
    cv=0
    baseline = max(pa.np.mean(y_test), 1-pa.np.mean(y_test))
    
    adict['Baseline']= baseline
    adict['Accuracy'] = a  /(cv+1)   
    adict['CV Accuracy'] = float(a/(cv+1))
    adict['Precision'] = p/(cv+1)
    adict['Recall'] = r/(cv+1)
    adict['F1'] = f1/(cv+1)
    adict['M'] = m/(cv+1)
    adict['Eval_Type'] = eval_type
    if verbose:
        print(classification_report(y_test, y_pred))
        print(eval_type)
        print(key,' : cv ',float(a/(cv+1)),baseline/(cv+1))#model.save(name_weights)
        print(key,' : p ',float(p/(cv+1)),baseline/(cv+1))
        print(key,' : r ',float(r/(cv+1)),baseline/(cv+1))
        print(key,' : m ',float(m/(cv+1)),baseline/(cv+1))
        print(key,' : f1 ',float(f1/(cv+1)),baseline/(cv+1))
    return adict
        

def evaluate(model, x_test, y_test,args,loss_fn,eval_type='val',verbose= False):
    model.eval()
    inputs = Variable(x_test)
    print(inputs.shape)
    preds, vector = model(inputs)
    
    preds_c = torch.max(preds, 1)[1]
    import torch.nn.functional as F
    
    if args['use_cuda']:
        preds_c = preds_c.cuda()
        preds = preds.cuda()
        adict=val_performance(y_test.detach().cpu().numpy(), preds_c.detach().cpu().numpy(), preds.detach().cpu().numpy(), '',eval_type,verbose= False)
    else:
        adict=val_performance(y_test.detach().numpy(), preds_c.detach().numpy(), preds.detach().numpy())

    return adict['F1'],loss_fn(preds, y_test).cpu().data.numpy(), vector.cpu().data.numpy(),adict
