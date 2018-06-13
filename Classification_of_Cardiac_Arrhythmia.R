#Reading the data file:
df <- read.csv("arrhythmia.data.csv")
df[df == "?"] <- NA

#Data Cleaning:
rare_lbl <- c(7,8,14,15,16)
iy <- which(df$Y %in% rare_lbl )
df <- df[-iy,]

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

## If there are more than 15 missing values, remove the columns
ind_rmv <- sum(miss_count>15)
list_miss <- names(miss_count)[1:ind_rmv]
iz <- which(colnames(df) %in% list_miss)
df <- df[,-iz]

## converting factor variables to numeric
type_list <- sapply(df,class)
bool_convert <- (type_list!="numeric")+(type_list!="integer")

## Convert if not numeric or integer
ind_convert <- which(bool_convert==2)
df[,ind_convert] <- sapply(df[,ind_convert],function(x)
  as.numeric(x))

## Imputing missing values (change incase of low accuracy, sub with mean)
k <- sapply(df,function(x) which(is.na(x)))
for (i in 1:length(k)){
  if (length(k[[i]]) < 1) {
    k[[i]] <- FALSE
  } else {
    k[[i]] <- TRUE
  } }

miss_colind <- which(unlist(k)==TRUE) ## Finding index of missing values
print(miss_colind) ##Tells which columns have missing values

## Only use columns from miss_colind
miss_index_qrst <- which(is.na(df$QRST_angle))
miss_index_t <- which(is.na(df$T_angle))
miss_index_p <- which(is.na(df$P_angle))

#Merging classes 3-4
df$Y[which(df$Y==4)]<-3

#Merging classes 5-6
df$Y[which(df$Y==5)]<-4
df$Y[which(df$Y==6)]<-4

#Merging classes 9-10
df$Y[which(df$Y==9)]<-5
df$Y[which(df$Y==10)]<-5

## converting Y into factor
df$Y <- as.factor(df$Y)

## Imputing missing value with class mean

df$QRST_angle[is.na(df$QRST_angle)] = mean(df$QRST_angle,na.rm=TRUE)
df$T_angle[is.na(df$T_angle)] = mean(df$T_angle,na.rm=TRUE)
df$P_angle[is.na(df$P_angle)] = mean(df$P_angle,na.rm=TRUE)

## Check sum(is.na(df))
## Scaling the dataframe
df.scaled <- as.data.frame(scale(df[1:(ncol(df)-1)]))
df.scaled['Y'] <- df$Y

#Feature selection
library(caret)
#library(fscaret)
#levels(df.scaled$Y) <- make.names(levels(factor(df.scaled$Y)))
set.seed(123)
test <- sample(nrow(df),floor(0.75*nrow(df)),replace = FALSE)
df_train <- df.scaled[test,]
train <- setdiff(c(1:nrow(df)),test)
df_test <- df.scaled[train,]

library(Boruta)
set.seed(123)
boruta.train <- Boruta(Y~., data = df_train, doTrace = 2,maxRuns = 120)
print(boruta.train)
var_boruta <- getSelectedAttributes(boruta.train,withTentative = TRUE)

#Sampling:
#Creating training and test sets
df_train.boruta = df_train[,var_boruta]
df_train.boruta['Y']=df_train$Y
df_test.boruta = df_test[,var_boruta]
df_test.boruta['Y']=df_test$Y

## Scaled df with boruta features
df.scaled.boruta=df.scaled[,var_boruta]

#Linear Discriminant Analysis:
set.seed(123)
lda.model = train(Y ~ ., data=df_train.boruta, method="lda", trControl = trainControl(method = "cv"))
pred_lda=predict(lda.model,df_test.boruta)
pred_lda_numeric <- as.numeric(predict(lda.model,df_test.boruta))
auc_lda=multiclass.roc(response = df_test.boruta$Y, predictor= pred_lda_numeric)


#Random Forest:
fitControl <- trainControl(method = "repeatedcv",number = 5, ## repeated 5 times
                             repeats = 5, search='random')
grid_rf <- expand.grid(mtry = seq(4,16,4), ntree = c(700,1000,2000))

#Training Random Forest
set.seed(123)
RF=train(Y~., data=df_train.boruta, method='rf',trControl=fitControl,tuneLength=20)
pred_RF=predict(RF,df_test.boruta)
mean(pred_RF==df_test.boruta$Y)

#Support Vector Machines:
svm.model <- train(Y~., data=df_train.boruta, method =
                       'svmRadial', tuneGrid=expand.grid(C=2^(-5:5), sigma=2^(-
                                                                                7:4)), trControl=trainControl(method='repeatedcv', number=5,
                                                                                                              repeats = 5))
pred_svt=predict(svm.model,df_test.boruta)
pred_svt_numeric <-  as.numeric(predict(svm.model,df_test.boruta))
library(pROC)
auc_svmrbf <- multiclass.roc(df_test.boruta$Y,pred_svt_numeric)
print(auc_svmrbf$auc)
##############SVM LINEAR#####################
# Fit the model on the training set
set.seed(123)
model.svmLinear <- train( Y ~., data = df_train.boruta, method= "svmLinear", trControl = trainControl("repeatedcv", number= 5,repeats=5), tuneGrid=expand.grid(C=2^(-5:15)) )
pred_svmlinear=predict(model.svmLinear,df_test.boruta)

#Decision Trees:
trctrl <- trainControl(method = "repeatedcv", number = 5,repeats = 5)
set.seed(123)
dtree_fit <- train(Y ~., data = df_train.boruta, method ="rpart", parms = list(split = "information"), trControl=trctrl, tuneLength = 40)
pred_dtree=predict(dtree_fit,df_test.boruta)
mean(pred_dtree==df_test.boruta$Y)
library(rpart.plot)
prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)

#XG-Boost:
library(mlr)
library(xgboost)
df_train.boruta.1 = df_train[,var_boruta]
ytrain.boruta.1 = as.numeric(df_train$Y)
df_test.boruta.1 = df_test[,var_boruta]
ytest.boruta.1 = as.numeric(df_test$Y)
#Change level to start from 0 for xgboost model
for (k in 1:6){
  ytrain.boruta.1[which(ytrain.boruta.1==k)] <- k-1
  ytest.boruta.1[which(ytest.boruta.1==k)] <- k-1
}
ytrain.boruta.1 = as.factor(ytrain.boruta.1)
ytest.boruta.1 = as.factor(ytest.boruta.1)
df_train.boruta.1[,'Y'] = ytrain.boruta.1
df_test.boruta.1[,'Y'] = ytest.boruta.1

#create tasks
train.task <- makeClassifTask(data = df_train.boruta.1,target ="Y")
test.task <- makeClassifTask(data = df_test.boruta.1,target ="Y")
xgb.learner <- makeLearner("classif.xgboost", predict.type ="response")

# set of fixed parameters
xgb.learner$par.vals <- list(booster = "gbtree", objective ="multi:softmax", eval_metric = "merror",early_stopping_rounds = 50, verbose = 0, nthread = 4)

# paramaters to be tuned
params <- makeParamSet(makeNumericParam("eta", lower = 0.01, upper = 0.3),
makeIntegerParam("max_depth",lower = 3L, upper = 10L),
makeIntegerParam("nrounds", lower = 3L, upper = 20L),
makeNumericParam("min_child_weight",lower = 1L, upper =5L),
makeNumericParam("subsample",lower = 0.5, upper = 1),
makeNumericParam("colsample_bytree",lower = 0.5, upper =1),
makeNumericParam("lambda", lower = 0, upper = 2),
makeDiscreteParam("gamma", values = c(0)))

# resampling
res.desc <- makeResampleDesc("CV", stratify = T, iters=5L)

# search
ctrl <- makeTuneControlRandom(maxit = 5L)

# parameter tuning
set.seed(123)
xgb.tune <- tuneParams(learner = xgb.learner, task =train.task, resampling = res.desc, measures = mmce, par.set =params, control = ctrl, show.info = T)

#set hyperparameters
set.seed(123)
xgb.learner_tune <- setHyperPars(xgb.learner, par.vals =xgb.tune$x)

#train model
set.seed(125)
xgb.model <- mlr::train(learner = xgb.learner_tune, task =train.task)

#predict model
xgb.pred <- predict(xgb.model,test.task)
confusionMatrix(xgb.pred$data$response,xgb.pred$data$truth)

#KNN:
library(caret)
set.seed(123)
trctrl_Knn <-  trainControl(method='repeatedcv',number=5,repeats = 5)
knn.model <- train(Y~.,data=df_train.boruta, method='knn',trControl=trctrl_Knn,tuneLength=20,preProcess = 'ignore')
pred_knn <- predict(knn.model,df_test.boruta)
knn.model
plot(knn.model)
confusionMatrix(pred_knn, df_test.boruta$Y)
pred_knn_numeric <- as.numeric(pred_knn)
library(pROC)
auc_rf <- multiclass.roc(df_test.boruta$Y, pred_knn_numeric)
print(auc_rf$auc)