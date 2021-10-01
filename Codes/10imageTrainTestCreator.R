library(filesstrings)

setwd("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/")
setwd("/Users/selcukkorkmaz/")
files = list.files()
length(files)
#move actives 
activeFiles = files[grep("_active.png", files)]
length(activeFiles)
file.move(activeFiles,"all/zactive/")

#move decoys 
decoyFiles = files[grep("_decoys.png", files)]
length(decoyFiles)
file.move(decoyFiles,"all/decoy/")


### numerical dataset
library(caret)
set.seed(1234)

data = read.table("~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/dataset/AID_485314/afterPreprocess/AID_485314.txt",
                  sep = "\t", header = T)

data[1:4,1:4]
dim(data)

data2 = data[!duplicated(data$Name),]
data2[1:4,1:4]
dim(data2)

table(data2$PUBCHEM_ACTIVITY_OUTCOME)

files = c(list.files("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/all/zactive/"),
          list.files("~/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/all/decoy/"))

files2 = gsub("_active.png","",files)
files3 = gsub("_decoys.png","",files2)
length(files3)

data2$Name = as.character(data2$Name)

data3 = data2[data2$Name %in% files3 ,]
data3[1:4,1:4]
dim(data3)

as.character(data3$Name)

table(data3$PUBCHEM_ACTIVITY_OUTCOME)
length(files3)
rownames(data3)= data3$Name

set.seed(123)
trainIndex <- createDataPartition(data3$PUBCHEM_ACTIVITY_OUTCOME, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

train <- data3[ trainIndex,]
test  <- data3[-trainIndex,]

dim(train)

X_train = train[!(colnames(train) %in% c("Name","PUBCHEM_ACTIVITY_OUTCOME"))]
X_train[1:3,1:4]
dim(X_train)
write.table(X_train,"~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/dataset/AID_485314/afterPreprocess/train/X_train.txt",
            quote = F, sep = "\t", row.names = T)

Y_train = train["PUBCHEM_ACTIVITY_OUTCOME"]
Y_train
dim(Y_train)
write.table(Y_train,"~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/dataset/AID_485314/afterPreprocess/train/Y_train.txt",
            quote = F, sep = "\t", row.names = T)

X_test = test[!(colnames(test) %in% c("Name","PUBCHEM_ACTIVITY_OUTCOME"))]
X_test[1:3,1:4]
dim(X_test)
write.table(X_test,"~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/dataset/AID_485314/afterPreprocess/test/X_test.txt",
            quote = F, sep = "\t", row.names = T)

Y_test = test["PUBCHEM_ACTIVITY_OUTCOME"]
Y_test
dim(Y_test)
write.table(Y_test,"~/Documents/Studies/DeepCNNandMLP/exercise2/numericalData/dataset/AID_485314/afterPreprocess/test/Y_test.txt",
            quote = F, sep = "\t", row.names = T)


########## image test/train
library(filesstrings)

trainImage = train[,c("Name","PUBCHEM_ACTIVITY_OUTCOME")]
testImage = test[,c("Name","PUBCHEM_ACTIVITY_OUTCOME")]


## train
trainDecoy = paste0(trainImage[trainImage$PUBCHEM_ACTIVITY_OUTCOME==0,1], "_decoys.png")
setwd("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/all/decoy/")
decoyPath="/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/train/decoy/"
file.move(trainDecoy, decoyPath)

trainActive = paste0(trainImage[trainImage$PUBCHEM_ACTIVITY_OUTCOME==1,1], "_active.png")
setwd("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/all/zactive/")
activePath="/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/train/zactive/"
file.move(trainActive, activePath)


## test
testDecoy = paste0(testImage[testImage$PUBCHEM_ACTIVITY_OUTCOME==0,1], "_decoys.png")
setwd("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/all/decoy/")
decoyPath="/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/test/decoy/"
file.move(testDecoy, decoyPath)

testActive = paste0(testImage[testImage$PUBCHEM_ACTIVITY_OUTCOME==1,1], "_active.png")
setwd("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/all/zactive/")
activePath="/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/pose/test/zactive/"
file.move(testActive, activePath)



