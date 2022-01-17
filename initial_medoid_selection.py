


#https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
import pandas as pd
import numpy as np
import random as rd
import nltk
from nltk.metrics.distance import edit_distance as l2
from nltk.metrics.distance import jaccard_distance as j2
from nltk.metrics.distance import masi_distance as m2

from evaluation import split




def initial_medoid_selection(init_list,max_k,kfold,repititions,dist_list,rtol1,rtol2,rtol3):
    seed_list=list(range(1,repititions+1))
    trest=['train','test']
    for dist in dist_list:
        for init in init_list:
            for seed in seed_list:            
                for t in trest:
                    print("On {} out of {} for the {}ing set of {}, initial medoid selection strategy '{}'.".format(seed,repititions,t, dist,init))
                    for exp in range(0,kfold):
                        for dfi in range(1,28):                            
                            df=pd.read_csv('kfold/df_{}_sim{}_seed{}_exp{}.csv'.format(t,dfi,seed,exp))
                            X = df['String'].values
                            pick=len(X)-1
                            i=rd.randint(0,pick)
                            Centroid=X[i]
                            Centroid_list=[]
                            Centroid_list.append(Centroid)
                            #print(len(Centroid_list))
                            element=0
                            for k in range(1,max_k):                    
                                D=[]
                                for x in X:
                                    if dist=='jaccard':
                                        D=np.append(D,j2(set(split(Centroid_list[element])),set(split(x))))
                                    elif dist=='edit':                                    
                                    #D.append(nltk.jaccard_distance(set(split(Centroid_list[element])),set(split(x))))
                                        D=np.append(D,l2(Centroid_list[element],x,substitution_cost=1))
                                if k==1:
                                    D2=D
                                else:
                                    if init in ['orig','orig.5']:
                                        D2=D
                                    else:
                                        D2=D+D2
                                D_=np.sort(D2)        
                                denom=np.sum(D_)
                                prob=D_/denom   
                                cummulative_prob=np.cumsum(prob)
                                #print('cumprob_min={}'.format(cummulative_prob[0]))   
                                #print('cumprob_max={}'.format(cummulative_prob[-1]))                                
                                while True:
                                    try:
                                        if init in ['orig','SDK']:
                                            r=round(rd.uniform(0.0,1.0),3)
                                        else:
                                            r=round(rd.uniform(0.5,1.0),3)
                                        #print('r={}'.format(r))
                                        indices = np.where(np.isclose(cummulative_prob,r,rtol=rtol1))                                
                                        #index=indices[0]
                                        index=rd.sample(indices,1)
                                        edistnorm=prob[index]
                                        edist=edistnorm*denom
                                        #print('edist={}'.format(edist))
                                        edist=edist[0]
                                        break
                                    except IndexError:
                                        if init in ['orig','SDK']:
                                            r=round(rd.uniform(0.0,1.0),3)
                                        else:
                                            r=round(rd.uniform(0.5,1.0),3)
                                        indices = np.where(np.isclose(cummulative_prob,r,rtol=rtol2))
                                        index=rd.sample(indices,1)
                                        edistnorm=prob[index]
                                        edist=edistnorm*denom
                                        try:
                                            edist=edist[0]
                                        except IndexError:
                                            if init in ['orig','SDK']:
                                                r=round(rd.uniform(0.0,1.0),3)
                                            else:
                                                r=round(rd.uniform(0.5,1.0),3)
                                            indices = np.where(np.isclose(cummulative_prob,r,rtol=rtol3))
                                            index=rd.sample(indices,1)
                                            edistnorm=prob[index]
                                            edist=edistnorm*denom
                                        
                                #edist=round(edist,3)
                                #print('edist={}'.format(edist))
                                while True:
                                    try:
                                        positions=np.where(np.isclose(D2,edist,rtol=rtol1))
                                        #print('positions={}'.format(positions))
                                        position=positions[0][0]
                                        #print('position={}'.format(position))
                                        break
                                    except IndexError:
                                        print('excepted')
                                        edist=round(edist-1)
                                        positions=np.where(np.isclose(D2,edist,rtol=rtol2))
                                        position=positions[0][0]                            
                                
                                center=X[position]
                                Centroid_list.append(center)    
                                element+=1

                            for i in range(len(Centroid_list)):
                                new_medoid=df.loc[df['String']==Centroid_list[i]]
                                new_medoid=new_medoid.head(1)
                            
                                if i==0:
                                    medoid=new_medoid
                                else:
                                    medoid=pd.concat([new_medoid,medoid])

                            df2=df.loc[~df.index.isin(medoid.index)]
                            #df3=pd.concat([df_top,df_repeated,df2])
                            df3=pd.concat([medoid,df2])
                            df3=df3.reset_index()
                            del df3['index']
                            df3.to_csv('kfold/df_{}_sim{}_seed{}_exp{}_plus_medoid{}_{}.csv'.format(t,dfi,seed,exp,dist,init),index=False)
    print("Initial medoid selection finished.")

                