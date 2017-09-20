
setwd("C:/Users/droyf/Desktop/UPGRAD/PA/classfication/SVM/casestudy")


#Data understanding 

# Number of Attributes: 785 (784 continuous, 1 nominal class label 0-9)

#Data preparation

#Necessary Packages

install.packages("caret")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("caTools")
install.packages("DEoptimR")
install.packages(e1071)
install.packages("ROCR")

library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
library(caTools)
library(DEoptimR)
library(e1071)
library(ROCR)

#Loading files

minst_train <- read.csv ("mnist_train.csv", stringsAsFactors = F,header=F)
minst_test <- read.csv ("mnist_test.csv", stringsAsFactors = F,header=F)


#Understanding Dimensions
dim(minst_train)
dim(minst_test)

#Structure of the dataset
str(minst_train)
str(minst_test)


#Exploring the data
summary(minst_train)
summary(minst_test)

#missing value

sapply(minst_train, function(x) sum(is.na(x)))
sapply(minst_test, function(x) sum(is.na(x)))

#Blank value
sapply(minst_train, function(x) length(which(x == "")))
sapply(minst_test, function(x) length(which(x == "")))

#renaming the column names
colnames(minst_train)[1]<-"value"
colnames(minst_test)[1]<-"value"

# Changing output variable "value" to factor type 

minst_train$value <- as.factor(minst_train$value)
minst_test$value <- as.factor(minst_test$value)

#Sampling of train data

set.seed(100)

indices = sample.split(minst_train$value, SplitRatio = 0.02)

train = minst_train[indices,]

Ovalid = minst_train[!(indices),]



#########get optimum PCA -giving direction and results in reducing dimension###################
prin_comp <- prcomp(train, scale. = T)
names(prin_comp)
prin_comp$center
prin_comp$scale
prin_comp$rotation
dim(prin_comp$x)
biplot(prin_comp, scale = 0)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlab = "Principal Component",
             ylab = "Proportion of Variance Explained",
             type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component",
              ylab = "Cumulative Proportion of Variance Explained",
              type = "b")

################################################PCA##################


X<-train[,-1]
Y<-train[,1]

trainLabel<-Y
Xreduced<-X/255
Xcov<-cov(Xreduced)
Xpca<-prcomp(Xcov)

trainLabel<-as.factor(trainLabel)
Xfinal<-as.matrix(Xreduced) %*% Xpca$rotation[,1:450]

test<-minst_test
testLabel<-as.factor(test[,1])
test<-test[,-1]
testreduced<-test/255
testfinal<-as.matrix(testreduced) %*% Xpca$rotation[,1:450]

##################################PCA end###############################

model_svm<-svm(Xfinal, trainLabel, kernel="radial")

Eval_RBF<-predict(model_svm, testfinal, type="class")

confusionMatrix(Eval_RBF,minst_test$value)

#accuracy:.4886

################cross validation###################################

#train control

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies  Evaluation metric is Accuracy

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(100)

grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.


fit.svm1<-train(Xfinal, trainLabel, method="svmRadial", metric=metric, tuneGrid=grid, trControl=trainControl)

# Printing cross validation result

print(fit.svm1)

#best tune parameter

fit.svm1$bestTune

#sigma = 0.01
#C(cost)=1

# Plotting model results

plot(fit.svm1)



# Checking overfitting - Non-Linear - SVM


# Validating the model results on test data
evaluate_non_linear<-predict(fit.svm1, testfinal)
confusionMatrix(evaluate_non_linear, minst_test$value)
#Accuracy : .1135

















