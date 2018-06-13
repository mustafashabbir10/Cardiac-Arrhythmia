#Reading the data file:
setwd("C:/Users/siddh/Desktop/ISEN 613/Cardiac
        arrhythmia")
getwd()
df <- read.csv("arrhythmia.csv")

#Data Cleaning:
df[df == "?"] <- NA
df$Y[df$Y!=1] <- "arrhythmia"
df$Y[df$Y==1] <- "normal"
table(df$Y) ## 245 - normal cases, 206 - cardiac

#arrhythmia conditions
## identifying dataframe columns with just one value and removing them

unique_vect <- sapply(df, function(x) length(unique(x)))
col_rm <- sort(unique_vect, decreasing = FALSE)
## Removing all columns with just one unique value
ind_lim <- sum(col_rm<2)
list_rm <- names(col_rm)[1:ind_lim]
ix <- which(colnames(df) %in% list_rm)
df <- df[,-ix]

## identifying missing values from dataframe
na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
miss_count = sort(na_count,decreasing = TRUE)

## If there are more than 10 missing values, remove the columns
ind_rmv <- sum(miss_count>10)
list_miss <- names(miss_count)[1:ind_rmv]
iz <- which(colnames(df) %in% list_miss)
df <- df[,-iz]

## converting factor variables to numeric
type_list <- sapply(df,class)

## converting factor variables to numeric
type_list <- sapply(df,class)
bool_convert <- (type_list!="numeric")+(type_list!="integer")

## Convert if not numeric or integer
ind_convert <- which(bool_convert==2)
ind_convert <- ind_convert[1:(length(ind_convert)-1)]
df[,ind_convert] <- sapply(df[,ind_convert],function(x)as.numeric(x))

## Imputing missing values
k <- sapply(df,function(x) which(is.na(x)))
for (i in 1:length(k)){
  if (length(k[[i]]) < 1) {
    k[[i]] <- FALSE
  } else {
    k[[i]] <- TRUE
  } }
miss_colind <- which(unlist(k)==TRUE) 
## Finding index of missing values
print(miss_colind) ##Tells which columns have missing values

miss_index_qrst <- which(is.na(df$QRST_angle))
miss_index_t <- which(is.na(df$T_angle))
miss_index_heart <- which(is.na(df$Heart))

## Imputing missing value with class mean
df$QRST_angle[is.na(df$QRST_angle)] = mean(df$QRST_angle,na.rm=TRUE)
df$T_angle[is.na(df$T_angle)] = mean(df$T_angle,na.rm=TRUE)
df$Heart[is.na(df$Heart)] = mean(df$T_angle, na.rm=TRUE)

## converting Y into factor
df$Y <- as.factor(df$Y)

## Check sum(is.na(df))
## Scaling the dataframe
df.scaled <- as.data.frame(scale(df[1:(ncol(df)-1)]))
df.scaled['Y'] <- df$Y

## Feature selection
library(caret)
set.seed(100)
test <- sample(nrow(df),floor(0.75*nrow(df)),replace = FALSE)
df_train <- df.scaled[test,]
train <- setdiff(c(1:nrow(df)),test)
df_test <- df.scaled[train,]

## Using boruta
library(Boruta)
set.seed(100)
boruta.train <- Boruta(Y~., data = df_train, doTrace =2,maxRuns = 120)
print(boruta.train)
var_boruta <- getSelectedAttributes(boruta.train,withTentative = TRUE)


#Sampling:
#Creating training and test sets
df_train.boruta = df_train[,var_boruta]
df_train.boruta['Y']=df_train$Y
df_test.boruta = df_test[,var_boruta]
df_test.boruta['Y']=df_test$Y

#Logistic Regression with Ridge Regularization:
library(glmnet)
library(boot)
library(glmnetUtils)
set.seed(10)
glm.ridge <- cv.glmnet(Y~.,data = df_train.boruta, family =
                         "binomial", type.measure = "auc", alpha = 0)
plot(glm.ridge)
print(max(glm.ridge$cvm)) ## Highest value for AUC
lambda.ridge <- glm.ridge$lambda.min

## Refitting the model with the best lambda
ridge.mod <- glmnet(Y~.,data = df_train.boruta,family =
                      "binomial",lambda = lambda.ridge, alpha = 0)

ridge.predict <- predict(ridge.mod,df_test.boruta,type =
                           "response")
ridge.class <- ifelse(ridge.predict>0.5,2,1)

## computing accuracy
confusionMatrix(as.numeric(ridge.class),as.numeric((df_test.b
                                                    oruta$Y)))

#Logistic Regression with Lasso Regularization:
set.seed(10)
glm.lasso <- cv.glmnet(Y~.,data = df_train.boruta, family =
                         "binomial", type.measure = "auc", alpha = 1)
plot(glm.lasso)
print(max(glm.lasso$cvm)) ## Highest value for AUC 0.8234

lambda.lasso <- glm.lasso$lambda.min ## Lambda value-0.01819

## Refitting the model with the best lambda
lasso.mod <- glmnet(Y~.,data = df_train.boruta,family ="binomial",lambda = lambda.lasso, alpha = 1)
lasso.predict <- predict(lasso.mod,s = lambda.lasso,df_test.boruta,type = "response") ## Probability
lasso.class <- ifelse(lasso.predict>0.5,2,1) ## actual classes

## computing accuracy
confusionMatrix(as.numeric(lasso.class),as.numeric(df_test.bo
                                                   ruta$Y))
#K-Nearest Neighbour:
set.seed(1)
KNN.Control <- trainControl(method = "cv", number =
                              5,classProbs = TRUE,summaryFunction = twoClassSummary)
fit <- train(Y ~ .,method = "knn",tuneGrid = expand.grid(k =
                                                           1:50),trControl = KNN.Control,
             metric = "ROC",data = df_train.boruta)
print(fit) ## The highest ROC value was observed at k = 12
trctrl_Knn=trainControl(method='repeatedcv',number=5,repe
                        ats = 5)
knn.prob <- predict(fit,df_test.boruta,type = "prob")
knn.predict <- predict(fit,df_test.boruta)
confusionMatrix(knn.predict,df_test.boruta$Y)

#Random Forest:
library(randomForest)
library(pROC)
# Fitting model
y = list()
roc = list()
#for (i in seq(from = 15,to = length(var_boruta),by = 2)){
set.seed(123)
fit <-rfcv(df_train.boruta[,var_boruta],df_train.boruta[,'Y'],step =0.95,cv.fold = 10)
print(min(fit$error.cv)) ## The min value of error.cv is for 21 predictors
plot(fit$n.var,fit$error.cv)

## Checking best value by using test dataset
for (i in seq(from = 15,to = length(var_boruta),by = 2)){
  set.seed(123)
  fit <-
    randomForest(df_train.boruta[,var_boruta],df_train.boruta[,'Y
                                                              '],mtry = i,importance = TRUE)
  predicted= predict(fit,df_test.boruta)
  y[[i]] <- mean(as.numeric(df_test$Y)==as.numeric(predicted))
  roc[[i]] <- roc(as.numeric(df_test$Y),as.numeric(predicted))
  print(auc(roc))
  print(y[[i]])
  print(i)}

## Best accuracy
print(max(unlist(y)))## For 17 and 25 predictors
set.seed(123)
rf.fit <- randomForest(df_train.boruta[,var_boruta],df_train.boruta[,'Y
                                                            '],mtry = 25,importance = TRUE)
predicted.rf= predict(rf.fit,df_test.boruta)
print("The best model accuracy is")
print(mean(as.numeric(df_test.boruta$Y)==as.numeric(predicted.rf))) ## accuracy
rf.prob <- predict(rf.fit,df_test.boruta,type = "prob")
confusionMatrix(predicted.rf,df_test.boruta$Y)

#Gradient Boosting:
fitControl <- trainControl(method = "repeatedcv",number = 5, classProbs = TRUE, repeats = 2,search='random')
gbmGrid <- expand.grid(interaction.depth = c(1,2,4), n.trees =c(500,2500,5000), shrinkage = c(0.001,0.01,0.1,1,4),
                       n.minobsinnode = 10)
gbmFit <- train(Y~., data = df_train.boruta, method = "gbm",
                trControl = fitControl, verbose = TRUE, tuneGrid = gbmGrid)
predict.gbm <- predict(gbmFit,df_test.boruta,type = "raw")
gbm.probs <- predict(gbmFit,df_test.boruta,type = "prob")[2]
confusionMatrix(predict.gbm,df_test.boruta$Y)

#Decision Tree:
library(tree)
set.seed(1)
tree.train_boruta <- tree(Y~.,df_train.boruta)
cv.train_boruta <- cv.tree(tree.train_boruta,FUN =
                             prune.misclass)
print(cv.train_boruta)
par(mfrow = c(1,2))
plot(cv.train_boruta$size,cv.train_boruta$dev,type = "b")
plot(cv.train_boruta$k,cv.train_boruta$dev,type = "b")

## The lowest misclassification is for depth = 9
prune.train_boruta <- prune.misclass(tree.train_boruta,best =
                                       9)
plot(prune.train_boruta)
text(prune.train_boruta,pretty = 0)
tree.pred <- predict(prune.train_boruta,df_test.boruta,type =
                       "class")
confusionMatrix(tree.pred,df_test.boruta$Y)

#Support Vector Machine:
set.seed(123)
svm.model <- train(Y~., data=df_train.boruta, method = 'svmRadial', tuneGrid = expand.grid(C=2^(-5:15),sigma=2^(-15:3)), trControl = trainControl(method ='repeatedcv',
                                                                                                                 number=5,repeats = 5,classProbs = TRUE))
svm.model
# From results,
sigma = 0.00390625
C = 0.5
set.seed(123)
svm.model.final <- train(Y~., data=df_train.boruta, method =
                           'svmRadial', tuneGrid = expand.grid(C=0.5,sigma=0.00390625),trControl = trainControl(method ='repeatedcv',
                                                                                                                number=5,repeats = 5,classProbs = TRUE))
pred_SVM <- predict(svm.model,df_test.boruta[,-62])
SVM.prob <- predict(svm.model,df)
pred_SVM_numeric <-
  as.numeric(predict(svm.model,df_test.boruta[,-62]))
CM_SVM <- confusionMatrix(pred_SVM,df_test.boruta$Y)

#Support Vector Classifier:
library(e1071)
set.seed(123)
svc <- svm(Y~.,data = df_train.boruta,kernel = "linear",cost = 1)
tune.out <- tune(svm,Y~.,data = df_train.boruta,ranges = list(cost= c(0.001,0.05,0.01,0.1,1,5,10,100)), kernel = "linear")
bestsvc <- tune.out$best.model
summary(bestsvc)
svcpred <- predict(bestsvc,df_test.boruta)
print(mean(svcpred==df_test.boruta$Y))
confusionMatrix(svcpred,df_test.boruta$Y)

# Neural Network:
library(nnet)
library(caret)
for (i in seq(from = 1,to = 12,by = 1))
  nn <- nnet(Y~.,data=df_train.boruta,size=3,rang=0.07,Hess=FALSE,decay=15e-4,maxit=250)
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(3,4,5, 6, 7))
nn.fit <- train(Y~., data = df_train.boruta, method = "nnet",
                maxit = 1000, tuneGrid = my.grid, trace = F,
                trControl=trainControl(method='repeatedcv',number=5,repeat
                  s = 5,classProbs = TRUE))
nn.predict <- predict(nn.fit,df_test.boruta)
confusionMatrix(nn.predict,df_test.boruta$Y)
