# Note: This is python code

import pubchempy as pm
import pandas as pd

aid = "1494158" # change AID
f = 0

cids = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/cids/AID_"+aid+".txt", delimiter="\t")
cids2 = cids.values
cids2 = cids2[:,0]
cids3 = cids2.tolist()
cids4=list(map(int, cids3))

seq = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/seq/AID_"+aid+"cidssequence.txt", delimiter="\t")
seq2 = seq.values
seq2 = seq2[:,0]
seq3 = seq2.tolist()
seq4=list(map(int, seq3))

seq5 = seq4[f:seq.shape[0]]

for i in seq5:
  c = cids4[i:i+60]
  f=f+1
  path = '/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/SDF/AID_'+aid+'/AID_'+aid+'_'+str(f)+'.sdf'
  pm.download('SDF', path, c)
  print(f)
  print ('%'+str(round((float(f)/seq.shape[0])*float(100))))

