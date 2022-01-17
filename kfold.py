#https://pyclustering.github.io/docs/0.9.0/html/d1/d6b/kmedoids_8py_source.html
#pyclustering kmedoids example
import pandas as pd 
import numpy as np
import random
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold

def kfold_split(kfold,repitions):
    kf=StratifiedKFold(n_splits=kfold)
    seed_list=list(range(1,repitions+1))
    for seed in seed_list:
        for i in range(1,28):
            df=pd.read_csv('data/simulated_set_{}_seed{}.csv'.format(i,seed))
            X=df[['String']].values
            y=df[['ClusterLabel']].values
            exp=0
            for train_index, test_index in kf.split(X,y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                df_train=np.concatenate((X_train,y_train),axis=1)
                df_test=np.concatenate((X_test,y_test),axis=1)
                df_train=pd.DataFrame(df_train, columns = ['String','ClusterLabel'])
                df_test=pd.DataFrame(df_test, columns = ['String','ClusterLabel'])
                #df_train.to_csv('kfold/df_train_simulated_{}_truek{}_exp{}_seed{}.csv'.format(method,sets,exp,seed),index=False)
                #df_test.to_csv('kfold/df_test_simulated_{}_truek{}_exp{}_seed{}.csv'.format(method,sets,exp,seed),index=False)
                
                df_train.to_csv('kfold/df_train_sim{}_seed{}_exp{}.csv'.format(i,seed,exp),index=False)
                df_test.to_csv('kfold/df_test_sim{}_seed{}_exp{}.csv'.format(i,seed,exp),index=False)
                exp+=1 
    print('Kfold finished.')


