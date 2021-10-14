library(data.table)
aid="AID_485314"

path = paste0("/home/vmplatin/deepDrug/deepScreen/numericalData/dataset/",aid,"/afterPreprocess/",aid,".txt")
data = fread(path, sep = "\t")
dim(data)
set.seed(1234)
probs <- c(0.70,0.30)
grp_1Row_prob <- probs / table(data$PUBCHEM_ACTIVITY_OUTCOME)
row_probs <- rep(grp_1Row_prob, times = table(data$PUBCHEM_ACTIVITY_OUTCOME))
row_probs
sampled_rows <- sample(1:NROW(data), size = 1000, prob = row_probs, replace = TRUE)

sampledData=data[sampled_rows, ]
table(sampledData$PUBCHEM_ACTIVITY_OUTCOME)

path2 = "/home/vmplatin/deepDrug/AID_485314/sampledData.txt"
fwrite(sampledData,path2, sep = "\t")


library(caret)
set.seed(1234)
trainIndex <- createDataPartition(sampledData$PUBCHEM_ACTIVITY_OUTCOME, p = .8,
                                  list = FALSE,
                                  times = 1)
head(trainIndex)

train <- sampledData[ trainIndex,]
train = data.frame(train)
dim(train)
table(train$PUBCHEM_ACTIVITY_OUTCOME)

test  <- sampledData[-trainIndex,]
test = data.frame(test)
dim(test)
table(test$PUBCHEM_ACTIVITY_OUTCOME)


X_train = train[!(colnames(train) %in% c("Name","PUBCHEM_ACTIVITY_OUTCOME"))]
X_train[1:3,1:4]
dim(X_train)
write.table(X_train,paste0("/home/vmplatin/deepDrug/deepScreen/numericalData/dataset/",aid,"/train/X_train_sample.txt"),
            quote = F, sep = "\t", row.names = T)

Y_train = train["PUBCHEM_ACTIVITY_OUTCOME"]
Y_train
dim(Y_train)
write.table(Y_train, paste0("/home/vmplatin/deepDrug/deepScreen/numericalData/dataset/",aid,"/train/Y_train_sample.txt"),
            quote = F, sep = "\t", row.names = T)

X_test = test[!(colnames(test) %in% c("Name","PUBCHEM_ACTIVITY_OUTCOME"))]
X_test[1:3,1:4]
dim(X_test)
write.table(X_test, paste0("/home/vmplatin/deepDrug/deepScreen/numericalData/dataset/",aid,"/test/X_test_sample.txt"),
            quote = F, sep = "\t", row.names = T)

Y_test = test["PUBCHEM_ACTIVITY_OUTCOME"]
Y_test
dim(Y_test)
write.table(Y_test, paste0("/home/vmplatin/deepDrug/deepScreen/numericalData/dataset/",aid,"/test/Y_test_sample.txt"),
            quote = F, sep = "\t", row.names = T)





