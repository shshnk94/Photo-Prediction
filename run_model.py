import os
from .model_args import args
from .cross_val import run_cross_val
import torch
import gensim
import pandas as pa

np=pa.np
pa.np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_name = 'CNN'
args['do_train']= True
args['do_test'] = True
args['do_eval'] = True
args['model_name']= model_name    
args['num_train_epochs'] = 100
args['learning_rate']=1e-5
args['earlystopping_patience']=5

embeddings = None
if args['use_pretrained_embeddings']:
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(args['pretrained_embedding_fpath'], binary=True)
    print('Please wait ... (it could take a while to load the file : {})'.format(args['pretrained_embedding_fpath']))

if __name__=='__main__':
    
    results=[]
    
    for lr in [5e-3,4e-3,3e-3,2e-3,1e-3,9e-4, 8e-4, 7e-4,6e-4,5e-4,4e-4,3e-4,2e-4,1e-4,
               9e-5, 8e-5, 7e-5,6e-5,5e-5,4e-5,3e-5,2e-5,1e-5,
               9e-6, 8e-6, 7e-6,6e-6,5e-6,4e-6,3e-6,2e-6,1e-6,
               9e-7, 8e-7, 7e-7,6e-7,5e-7,4e-7,3e-7,2e-7,1e-7
               ]:

        args['learning_rate'] = lr
        cv_results = run_cross_val(args, embeddings, device)
        results.append(cv_results)
   
    x = [pa.DataFrame.from_dict(ix) for ix in results]
    df = pa.concat(x)
    df.to_csv('results.csv')
