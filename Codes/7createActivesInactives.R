setwd("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/")

smiles = read.table("aid/AID_485314_sample_smiles.txt", header = T, sep = "\t")
head(smiles)
dim(smiles)
colnames(smiles) = c("PUBCHEM_CID","SMILES")

aid = read.csv("aid/AID_485314_sample_datatable_all.csv")
head(aid)
aid2 = aid[,c("PUBCHEM_CID","PUBCHEM_ACTIVITY_OUTCOME")]

table(aid2$PUBCHEM_ACTIVITY_OUTCOME)

aid2[aid2$PUBCHEM_ACTIVITY_OUTCOME == "Inconclusive",] = NA

head(aid2)
dim(aid2)

aid3=aid2[complete.cases(aid2),]
head(aid3)
dim(aid3)
table(aid3$PUBCHEM_ACTIVITY_OUTCOME)


library(dplyr)
library(readr)

dataset <- left_join(aid3,smiles,by = c("PUBCHEM_CID"))
head(dataset)
dim(dataset)

s = split(dataset,dataset$PUBCHEM_ACTIVITY_OUTCOME)

write.table(s$Active[,c(1,3)], "smiles/actives.txt", row.names = F, col.names = F, quote = F, sep = "\t")
write.table(s$Inactive[,c(1,3)], "smiles/decoys.txt", row.names = F, col.names = F, quote = F, sep = "\t")


  

