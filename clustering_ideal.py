import pandas as pd 
import numpy as np
import random
import pickle  

import pyclustering
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
import sys
import warnings
warnings.filterwarnings("ignore") 

from evaluation import split, dist_matrix

import nltk
from nltk.metrics.distance import edit_distance as l2

max_k=11
klist=list(range(2,max_k))
#dist='jaccard' 
dist_list=['jaccard','edit']
kfold=3
trest=['train','test']

#if you're moving the medoids to the top of the next list, then the medoid index should aways remain 0,1...0,1,2 etc

seed_list=[1,2,3,4,5]   #currently no real seed, but run the experiments 5 times....datasets will be at the very least a different order (no shuffle situation) but will be different based on the random samples in the shuffle sequences at the most.

for seed in seed_list:
    for dist in dist_list:
        for t in trest:
            for i in range(1,28):
                for exp in range(kfold):
                    df=pd.read_csv('kfold/df_{}_sim{}_seed{}_exp{}_plus_medoidideal.csv'.format(t,i,seed,exp,dist))
                    
                    df=df.reset_index(drop=True)
                    df['true_index']=df.index
                    df2=df['String'].values
                    l=(len(df),len(df))
                    Matrix=np.zeros(l,dtype=np.float)
                    dist_matrix(df2,Matrix,dist,1,sim='True')                    
                    np.save("distance_matrices_{}/{}_exp{}_simulated_set{}_seed{}.npy".format(dist,t,exp,i,seed),Matrix)

                    for k in klist:
                        data=np.load('distance_matrices_{}/{}_exp{}_simulated_set{}_seed{}.npy'.format(dist,t,exp,i,seed))
                        initial_medoids=list(range(0,k))
                        kmedoids_instance=kmedoids(data,initial_medoids,data_type='distance_matrix')
                        kmedoids_instance.process()
                        clusters=kmedoids_instance.get_clusters()
                        medoids=kmedoids_instance.get_medoids()
                        lengthy=len(clusters)
                        lengthy2=len(initial_medoids)
                        
                        #store new precomputed matrix without medoids.
                        #data_precomputed=np.delete(data,medoids,axis=0)
                        #data_precomputed2=np.delete(data_precomputed,medoids,axis=1)
                        #np.save("distance_matrices_train/{}_exp{}_simulated_set{}_seed{}_precomputed.npy".format(dist,exp,i,seed),data_precomputed2)
                        
                        df['cluster']=99
                        for cluster in range(len(clusters)):
                            for row in range(len(df)):
                                if df.at[row,'true_index'] in (clusters[cluster]):
                                    df.at[row,'cluster']=cluster
                        df.to_csv('clustering_{}/{}/cluster_ids_{}_exp{}_k{}_simulated_set{}_seed{}.csv'.format(t,dist,dist,exp,k,i,seed),sep=',',index=False)
                        
                        med_obs=df.loc[df['true_index'].isin(medoids)]
                        med_obs.to_csv('clustering_{}/{}/final_medoids_exp{}_k{}_simulated_set{}_seed{}.csv'.format(t,dist,exp,k,i,seed),sep=',',index=False)




for dist in dist_list:
    for seed in seed_list:
        for i in range(1,28):
            for k in klist:
                for exp in range(kfold):
                    test=pd.read_csv('clustering_test/{}/cluster_ids_{}_exp{}_k{}_simulated_set{}_seed{}.csv'.format(dist,dist,exp,k,i,seed))
                    test2=test
                    medoids=pd.read_csv('clustering_train/{}/final_medoids_exp{}_k{}_simulated_set{}_seed{}.csv'.format(dist,exp,k,i,seed))
                    medoids2=medoids

                    for row in range(len(test2)):
                        dissim_list=[]
                        for j in range(len(medoids2)):
                            medoid_clust_id=medoids2['cluster'].values
                            medoid=medoids2['String'].values
                            medoid=medoid[j]
                            test3=test2['String'].values
                            test4=test3[row]
                            if dist=='jaccard':
                                dissim=nltk.jaccard_distance(set(split(test4)),set(split(medoid)))
                            elif dist=='masi':
                                dissim=nltk.masi_distance(set(split(test4)),set(split(medoid)))
                            elif dist=='length_norm_edit':
                                dissim=l2(test4,medoid)/max(len(test4),len(medoid))
                            else:
                                dissim=l2(test4,medoid)
                            
                            dissim_list.append(dissim)
                        
                        clust=np.argmin(dissim_list)
                        clusts=medoid_clust_id[clust]
                        test2.at[row,'cluster']=clusts
                    
                    test2.to_csv('pred_clust/{}/pred_exp{}_k{}_simulated_set{}_seed{}.csv'.format(dist,exp,k,i,seed),sep=',',index=False)
