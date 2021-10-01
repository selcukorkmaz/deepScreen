library(data.table)
aid = "AID_485314"
path = paste0("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/dataset/",aid,"/afterPreprocess/",aid,".txt")
sampleData = fread(path, sep = "\t")
sampleData[1:4,1:4]
dim(sampleData)

path = paste0("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/aid/",aid,
              "_datatable_all.csv")
comps = read.csv(path, header = T, stringsAsFactors = F)
dim(comps)
table(comps$PUBCHEM_ACTIVITY_OUTCOME)

sampleComps = comps[comps$PUBCHEM_CID %in% sampleData$Name,]
dim(sampleComps)
sampleComps2 = sampleComps[!duplicated(sampleComps$PUBCHEM_CID),]
dim(sampleComps2)
write.csv(sampleComps2, paste0("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/aid/",aid,"_sample_datatable_all.csv"))


path = paste0("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/aid/",aid,
              "_smiles.txt")
smiles = read.table(path, header = F, sep = "\t")
head(smiles)
sampleSmiles = smiles[smiles$V1 %in% sampleData$Name,]
dim(sampleSmiles)
sampleComps2 = sampleSmiles[!duplicated(sampleSmiles$V1),]
dim(sampleComps2)
head(sampleComps2)
write.table(sampleComps2, paste0("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/aid/",aid,"_sample_smiles.txt"),quote = F,
            row.names = F, sep = "\t")

