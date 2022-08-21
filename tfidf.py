from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np

def Inverse_Doc_Freq(corpus, unique_words):
   idf_dict={}
   N=len(corpus)
   for i in unique_words:
     count=0
     for sen in corpus:
       if i in sen.split():
         count=count+1
       idf_dict[i]=(math.log((1+N)/(count+1)))+1
   return idf_dict


def feature(whole_data):
    unique_words = set()
    if isinstance(whole_data, (list,)):
      for x in whole_data:
        for y in x.split():
          if len(y)<2:
            continue
          unique_words.add(y)
          
      unique_words = sorted(list(unique_words))
      vocab = {j:i for i,j in enumerate(unique_words)}
      
      Idf_values=Inverse_Doc_Freq(whole_data,unique_words)
    return vocab, Idf_values
   
def matrix(dataset,vocabulary,idf_values):
     sparse_matrix= csr_matrix( (len(dataset), len(vocabulary)), dtype=np.float64)
     for row  in range(0,len(dataset)):
       number_of_words_in_sentence=Counter(dataset[row].split())
       for word in dataset[row].split():
           if word in  list(vocabulary.keys()):
               tf_idf_value=(number_of_words_in_sentence[word]/len(dataset[row].split()))*(idf_values[word])
               sparse_matrix[row,vocabulary[word]]=tf_idf_value
     output =normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
     return output

   
def feature_extraction(corpus):
   Vocabulary, idf_of_vocabulary=feature(corpus)
   final_output=matrix(corpus,Vocabulary,idf_of_vocabulary)
   final_output=final_output.toarray()

   return final_output
