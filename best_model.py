class BestModel(object):
    def __init__(self ):
        self.adict = {}
        self.best_epoch = 0

    def step(self, adict,epoch):
        if epoch==0:
            self.adict = adict
            self.best_epoch=0
        else:
            if adict['F1']>=self.adict['F1']:
                self.adict = adict
                self.best_epoch=epoch
                
                
def get_best_epoch(ep_tracker):

    import pandas as pa
    ep_tracker= pa.concat(ep_tracker)
    
    metrics={}
    for x in ['ROC AUC','PR AUC','Baseline','Accuracy','CV Accuracy','Precision','Recall','F1','M']:
        metrics[x] = pa.np.mean
    
    df = ep_tracker.groupby('val_epoch',as_index=False).agg(metrics)
    df.sort_values(by=['F1'],inplace=True, ascending=False)
    
    records = df.to_dict(orient='records')
    return list(df['val_epoch'])[0] , records[0]
