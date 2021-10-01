#### Get the dataset ####

aid = "AID_485314" # change AID
data = read.csv("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/aid/AID_485314_datatable_all.csv")
dim(data)

cid = data.frame(data$PUBCHEM_CID[complete.cases(data$PUBCHEM_CID)])
colnames(cid) = "cids"
head(cid)

write.table(cid,paste0("~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/cids/",aid,".txt"),
            quote = F, row.names = F)

s = c(seq(0,nrow(cid),60), nrow(cid))
write.table(s, paste0('~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/seq/',aid,'cidssequence.txt'), quote = F, row.names = F)


length(unique(cid[,1]))

length((data$PUBCHEM_ACTIVITY_OUTCOME))
table((data$PUBCHEM_ACTIVITY_OUTCOME))


########### Download SDF files #####

