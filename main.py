from generate_sequence_data import generate_sequence_data
from kfold import kfold_split
from initial_medoid_selection import initial_medoid_selection
from clustering import clustering

n=1000                  #observations 
max_truek=11            #clusters, can be anywhere between 2 and 12
m_list=[8]              #sequence length
OI_list=[0.5]           #the number of overlap of levels between clusters...this is currently hardcoded
w_list=[0,1]            #does order matter within clusters or not....later, within a sliding window of size x?
x_list=[2,4]            #sliding window size
gamma=0.25              #percent of observations in cluster with "mutations"
num_char=1              #number of characters replaced with noise (analagous to percent of string that has noise)
repititions=10          #number of times to randomly create datasets
kfold=5
max_k=11
#dist_list=['jaccard','edit']
dist_list=['jaccard']
init_list=['orig','orig.5','SDK','SDK.5']

rtol1=0.01
rtol2=0.05
rtol3=0.1

set3=[
'ABCDEFGH'
,'CDCDEFGHIJIJ'
,'FGHIJKLM'
,'HIHIJKLMNONO'
,'KLMNOPQR'
,'MNMNOPQRSTST'
,'PQRSTUVW'
,'RSRSTUVWXYXY'
,'UVWXYZ12'
,'WXWXYZ123434'] 

if __name__=="__main__":
    #generate_sequence_data(repititions, set3, gamma, num_char,x_list,w_list,OI_list,m_list,max_truek,n)
    #kfold_split(kfold,repititions)
    #initial_medoid_selection(init_list,max_k,kfold,repititions,dist_list,rtol1,rtol2,rtol3)
    clustering(init_list,max_k,kfold,repititions,dist_list)