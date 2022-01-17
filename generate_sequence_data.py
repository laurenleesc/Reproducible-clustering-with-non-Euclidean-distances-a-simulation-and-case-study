#https://www.educative.io/edpresso/how-to-generate-a-random-string-in-python

import random
from random import randint
import reprlib
import string
import numpy as np
import pandas as pd


def generate_sequence_data(repititions, string_list, gamma, num_char,x_list,w_list,OI_list,m_list,max_truek,n):
    seed_list=list(range(1,repititions+1)) 
    for seed in seed_list:
        index = 0
        for truek in range(2,max_truek):
            step=0
            b=round(n/truek)+randint(0, 10)

            for m in m_list:
                for OI in OI_list:
                    for w in w_list:
                        if w==0:                   
                            index+=1
                            end=step+max_truek
                            print(index,truek,m,OI,w,"NA")
                            options=string_list[step:end]

                            #for cluster,avg_length in zip(range(truek),len_list):
                            for cluster in range(truek):
                                clust=[]
                                letters_=options[cluster]
                                letters_2=letters_                                                  
                                
                                for obs in range(b):
                                    #start=random.randint(0,5)
                                    start=0
                                    #x1=avg_length-2 #maybe later add some variation to length based on cluster
                                    #length=random.randint(x1, avg_length)
                                    length=len(letters_)
                                    
                                    #clust.append(''.join(letters_2[start:length]))
                                    sub=''.join(letters_2[start:length])
                                    r=round(random.random(),3)
                                    if r<=gamma:
                                        inds = [i for i,_ in enumerate(sub)]
                                        letts =  iter(random.sample(string.ascii_uppercase, num_char))
                                        lst = list(sub)
                                        sam = random.sample(inds, num_char)
                                        for ind in sam:
                                            lst[ind] = next(letts)
                                        sub=''.join(lst)
                                    clust.append(sub)

                                
                                #dollars = np.random.normal(base_price*(cluster+1), 10, b)
                                df = pd.DataFrame(clust,columns=['String'])
                                df['ClusterLabel']=cluster
                                #df['dollars']=dollars

                                if cluster==0:
                                    df2=df
                                else:
                                    df2=pd.concat([df2,df])
                                
                            df3=df2.sample(frac=1).reset_index(drop=True)
                                
                            df3.to_csv('data/simulated_set_{}_seed{}.csv'.format(index,seed),index=False) 

                        else:
                            for x in x_list:
                                index+=1
                                end=step+max_truek
                                print(index,truek,m,OI,w,x)
                                    
                                options=string_list[step:end]

                                #for cluster,avg_length in zip(range(truek),len_list):
                                for cluster in range(truek):
                                    clust=[]
                                    letters_=options[cluster]
                                    length=len(letters_)
                                    z=length-(x-1)
                                    #print('z={}'.format(z))
                                            
                                    for obs in range(b):
                                        if length==8 and x==2:
                                            c=[''.join(random.sample(letters_[i:i+2],2)) for i in range(0,8,2)]
                                            c2=''.join(c)
                                            c3=''.join(random.sample(c2[2:4],2))
                                            c3b=''.join(random.sample(c2[6:8],2))
                                            c4=c2[0:2]+c3+c2[4:6]+c3b
                                            c5=''.join(c4)
                                        
                                        elif length==8 and x==4:
                                            c=[''.join(random.sample(letters_[i:i+4],4)) for i in range(0,8,4)]
                                            c2=''.join(c)
                                            #print('c2={}'.format(len(c2)))
                                            
                                            c3=''.join(random.sample(c2[2:6],4))
                                            #c3b=''.join(random.sample(c2[6:8],2))
                                            c4=c2[0:2]+c3+c2[6:8]
                                            c5=''.join(c4) 
                                            #print('c5={}'.format(len(c5))) 
                                            #print(c5)   
                                            #letters_2=c5
                                        elif length==12 and x==2:
                                            c=[''.join(random.sample(letters_[i:i+2],2)) for i in range(0,12,2)]
                                            c2=''.join(c)
                                            c3=''.join(random.sample(c2[2:4],2))
                                            c3b=''.join(random.sample(c2[6:8],2))
                                            c3c=''.join(random.sample(c2[10:12],2))
                                            c4=c2[0:2]+c3+c2[4:6]+c3b+c2[8:10]+c3c
                                            c5=''.join(c4)   
                                        elif length==12 and x==4:   
                                            c=[''.join(random.sample(letters_[i:i+4],4)) for i in range(0,12,4)]
                                            c2=''.join(c)
                                            #print('c2={}'.format(len(c2)))                                        
                                            c3=''.join(random.sample(c2[2:6],4))
                                            c3b=''.join(random.sample(c2[8:12],4))
                                            c4=c2[0:2]+c3+c2[6:8]+c3b
                                            c5=''.join(c4)
                                        elif length==16 and x==2:
                                            c=[''.join(random.sample(letters_[i:i+2],2)) for i in range(0,16,2)]
                                            c2=''.join(c)
                                            c3=''.join(random.sample(c2[2:4],2))
                                            c3b=''.join(random.sample(c2[6:8],2))
                                            c3c=''.join(random.sample(c2[10:12],2))
                                            c3d=''.join(random.sample(c2[12:14],2))
                                            c4=c2[0:2]+c3+c2[4:6]+c3b+c2[8:10]+c3c+c2[10:12]+c3d+c2[14:16]
                                            c5=''.join(c4)   
                                        elif length==16 and x==4:   
                                            c=[''.join(random.sample(letters_[i:i+4],4)) for i in range(0,16,4)]
                                            c2=''.join(c)
                                            #print('c2={}'.format(len(c2)))                                        
                                            c3=''.join(random.sample(c2[2:6],4))
                                            c3b=''.join(random.sample(c2[8:12],4))
                                            c3c=''.join(random.sample(c2[10:14],4))
                                            c4=c2[0:2]+c3+c2[6:8]+c3b+c2[8:10]+c3c+c2[14:16]
                                            c5=''.join(c4)                                                   
                                        letters_2=c5
                                        check=len(letters_2)
                                        #print(length,check)
                                        #clust.append(''.join(letters_2[start:length]))
                                        sub=''.join(letters_2)
                                        r=round(random.random(),3)
                                        if r<=gamma:
                                            inds = [i for i,_ in enumerate(sub)]
                                            letts =  iter(random.sample(string.ascii_uppercase, num_char))
                                            lst = list(sub)
                                            sam = random.sample(inds, num_char)
                                            for ind in sam:
                                                lst[ind] = next(letts)
                                            sub=''.join(lst)
                                        clust.append(sub)

                                    #dollars = np.random.normal(base_price*(cluster+1), 10, b)
                                    df = pd.DataFrame(clust,columns=['String'])
                                    df['ClusterLabel']=cluster
                                    #df['dollars']=dollars

                                    if cluster==0:
                                        df2=df
                                    else:
                                        df2=pd.concat([df2,df])
                                
                                df3=df2.sample(frac=1).reset_index(drop=True)    
                                df3.to_csv('data/simulated_set_{}_seed{}.csv'.format(index,seed),index=False) 

                                    
                    step+=max_truek