import pandas as pd
import numpy as np
import sys
import sqlite3
from sklearn import preprocessing
import re
import random
import warnings; warnings.filterwarnings('ignore')

import nltk
from nltk.metrics.distance import edit_distance as l2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import pyclustering

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

def FindMaxLength(lst):
    indexmaxlength=max((len(l), i) for i, l in enumerate(lst))[1]
    return indexmaxlength

def split(word):
    return [char for char in word]

#https://stackoverflow.com/questions/2460177/edit-distance-in-python
def levenshteinDistance(s1, s2):
    path_length=0
    
    if len(s1) > len(s2):        
        #weight+=len(s1)+len(s2)
        s1, s2 = s2, s1 #If the length of s1 is greater than the length of s2, switch the names? 
                                            #This keeps the shortest one always called s1 and
                                            #the longest one always called s2.

    distances = range(len(s1) + 1)  #is this the length of the edit path?????? is it always the length of the shortest plus 1?
    #print(distances)
    for i2, c2 in enumerate(s2):
        #print(12,c2)
        distances_ = [i2+1]
        
        for i1, c1 in enumerate(s1):
            if c1 == c2:                                
                distances_.append(distances[i1])
                print('equal {}'.format(distances_))
                #print(i2,i1)
                path_length+=1                
            else: 
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1]))) 
                #path_length+=1                                                      
                print('not equal {}'.format(distances_))
        distances = distances_
        print(len(distances_))
    distances=distances[-1]
    return distances, path_length
    

def sim_matrix(data,Matrix,distance_calc, denom, sim='True'):    
    for k in range(0,len(data)):
        for m in range(0,len(data)):

            if distance_calc=='jaccard':
                if sim=='True':
                    Matrix[m,k]=1-nltk.jaccard_distance(set(split(data[m])),set(split(data[k])))
                
            elif distance_calc=='masi':
                if sim=='True':
                    Matrix[m,k]=1-nltk.masi_distance(set(split(data[m])),set(split(data[k])))
                         
    return Matrix

def dist_matrix(data,Matrix,distance_calc, denom, sim='True'):   
    #print(distance_calc) 
    for k in range(0,len(data)):
        for m in range(0,len(data)):
            #if k>=4:
                #print(data[k],data[m])
            if distance_calc=='jaccard':
                if sim=='True':
                    Matrix[m,k]=nltk.jaccard_distance(set(split(data[m])),set(split(data[k])))
                else:
                    Matrix[m,k]=nltk.jaccard_distance(set(data[m]),set(data[k]))
            elif distance_calc=='masi' or distance_calc=='weighted_jaccard':
                
                if sim=='True':
                    Matrix[m,k]=nltk.masi_distance(set(split(data[m])),set(split(data[k])))
                else:
                    Matrix[m,k]=nltk.masi_distance(set(data[m]),set(data[k]))
            elif distance_calc=='length_norm_edit':
                Matrix[m,k]=l2(data[m],data[k],substitution_cost=1)/max(len(data[m]),len(data[k]))
            elif distance_calc=='weight_norm_edit':
                Matrix[m,k]=levenshteinDistance(data[m],data[k])
                #if k>=4:
                    #print('edit={}'.format(Matrix[m,k]))
            elif distance_calc=='edit':
                Matrix[m,k]=l2(data[m],data[k],substitution_cost=1)
    return Matrix

def get_prediction_strength2(method_,datatype,distance_calc,k,pred_clust_label, x_train, x_test, test_labels):
    n_test=len(x_test)

    #populate the centroid-predicted co-membership matrix (use "p" and "pred_clust_labels")
    D = np.zeros(shape=(n_test,n_test))
    for p1, c1 in zip(pred_clust_label,list(range(n_test))):
        x=p1
        for p2, c2 in zip(pred_clust_label, list(range(n_test))):
            y=p2
            if c1 != c2:
                if p1==p2:
                    D[c1,c2] = 1.0
    #populate the TEST CLUSTERING co-membership matrix (use "t" and "test_labels")
    D_ = np.zeros(shape=(n_test, n_test))
    for t1, c1 in zip(test_labels, list(range(n_test))):
        for t2, c2 in zip(test_labels, list(range(n_test))):
            if c1 != c2:
                if t1 == t2:
                    D_[c1,c2]=1.0

    bias=D_-D
    bias[bias>0]=0
    bias[bias<0]=1

    variance=D_-D
    variance[variance<0]=0

    ss=[]
    bs=[]
    vs=[]

    for j in range(k):
        s = 0
        b = 0
        v = 0
        examples_j = x_test[test_labels==j].tolist()
        n_examples_j = len(examples_j)

        for p1, t1, c1 in zip(pred_clust_label, test_labels, list(range(n_test))):
            for p2, t2, c2 in zip(pred_clust_label, test_labels, list(range(n_test))):
                if c1 != c2 and t1==t2 and t1==j:
                    s += D[c1,c2]
                    v += variance[c1,c2]
                elif c1 != c2 and p1==p2 and t1 != t2 and t1 == j:
                    b += bias[c1,c2]
        
        if n_examples_j>1:
            ss.append(s / (n_examples_j * (n_examples_j - 1)))
            vs.append(v / (n_examples_j * (n_examples_j - 1)))
            bs.append((b / (n_examples_j * (n_examples_j - 1)))/(j + 1))
        
        else:
            ss.append(0)
            vs.append(0)
            bs.append(0)
    
    prediction_strength = min(ss)
    variance_ = max(vs)
    bias_ = max(bs)

    return prediction_strength, variance_, bias_