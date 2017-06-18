set.seed(7)
library(DMwR)
library(XLConnect)
library(e1071)
library(randomForest)
library(mlbench)
library(csvread) 
library(lubridate)
library(ggplot2)
library(plyr)
library(dplyr)
library(stringr)
library(caret)
library(rpart)
library(rattle)
library(ROSE)
library(ROCR)
library(MASS)
library(ipred)
library(rpart.plot)
library(fastAdaboost)
library(rpart.plot)
library(pROC)

options(digits=8, scipen = 50)

# Load and Process the data
excel.file <- file.path("/Users/Akshay/Data/ABI/Project/FinalAttributes.xlsx")
finalData <- readWorksheetFromFile(excel.file, sheet="FinalAttributes")
finalData <- na.omit(finalData)
numeric_cols <- sapply(finalData, is.numeric)
finalData$int_rate <- str_replace_all(finalData$int_rate, "[%]", "")
finalData$int_rate <- as.numeric(finalData$int_rate)
finalData$issue_d <- as.Date(finalData$issue_d, format = "%d/%m/%Y")
finalData$earliest_cr_line <- as.Date(finalData$earliest_cr_line, "%d/%m/%y")
finalData$emp_length <- as.factor(finalData$emp_length)
finalData$sub_grade <- as.factor(finalData$sub_grade)
finalData$grade <- as.factor(finalData$grade)
finalData$term <- as.factor(finalData$term)
finalData$verification_status <- as.factor(finalData$verification_status)
finalData$home_ownership <- as.factor(finalData$home_ownership)

table(finalData$application_type)

# Take the difference between first credit time and issue time as a variable
finalData$time_since_first_credit <- finalData$issue_d - finalData$earliest_cr_line
finalData$time_since_first_credit <-as.numeric(finalData$time_since_first_credit)
finalData$earliest_cr_line <- unclass(finalData$earliest_cr_line)
finalData$issue_d <- unclass(finalData$issue_d)
plot(finalData$earliest_cr_line)

# Group status
good_status = c("Fully Paid")
finalData$status_group = ifelse(finalData$loan_status %in% good_status,"Good","Bad")
finalData$status_group = factor(finalData$status_group)
finalData$loan_status <- NULL
finalData$time_since_first_credit[which(finalData$earliest_cr_line >= finalData$issue_d)] <- 0


summary(finalData)
dim(finalData)

# Find high correlation
correlationMatrix <- cor(finalData[sapply(finalData,is.numeric)])  
correlationMatrix
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75, names = TRUE)  
print(highlyCorrelated)

finalData$earliest_cr_line <- NULL
finalData$installment <- NULL
finalData$application_type <- NULL


str(finalData)
# Sample for training and testing data
samp <- sample(nrow(finalData), 0.75 * nrow(finalData))
train <- finalData[samp, ]
test <- finalData[-samp, ]
str(train)


NewData <- SMOTE(status_group~., train, perc.over = 1000, perc.under = 110)
str(NewData)
table(NewData[,24])



# ==========================Train the model=============================
# - Random Forest
ran_m <- randomForest(status_group ~ ., data = train)
pred_ran=predict(ran_m,test)
confusionMatrix(pred_ran, test$status_group)
roc.curve(test$status_group, pred_ran, plot = TRUE)

# ------- SMOTE data --------
ran_ms <- randomForest(status_group ~ ., data = NewData)
pred_rans=predict(ran_ms,test)
confusionMatrix(pred_rans, test$status_group)
roc.curve(test$status_group, pred_rans, plot = TRUE)


# - Rpart classification tree
rp_m <- rpart(status_group ~ . , data = train, control=rpart.control(minsplit=15, cp=0.0006))
fancyRpartPlot(rp_m)
pred_rp <- (predict(rp_m, test, type = "class"))
confusionMatrix(pred_rp, test$status_group)
roc.curve(test$status_group, pred_rp, plot = TRUE)

# ------- SMOTE data --------
rp_ms <- rpart(status_group ~ . , data = NewData, control=rpart.control(minsplit=25, cp=0.05))
pred_rps <- (predict(rp_ms, test, type = "class"))
confusionMatrix(pred_rps, test$status_group)
roc.curve(test$status_group, pred_rps, plot = TRUE)


# - Naive-Bayes
nB_m<-naiveBayes(status_group~., data = train)
pred_nB <- predict(nB_m,test,type="class")
confusionMatrix(pred_nB, test$status_group)
roc.curve(test$status_group, pred_nB, plot = TRUE)

# ------- SMOTE data --------
nB_ms<-naiveBayes(status_group~., data = NewData)
pred_nBs <- predict(nB_ms,test,type="class")
confusionMatrix(pred_nBs, test$status_group)
roc.curve(test$status_group, pred_nBs, plot = TRUE)


# - SVM (Support Vector Machine)
svm_m <- svm(status_group~., data = train, kernel="radial", cost=60, gamma=0.04)
pred_svm=predict(svm_m, test)
summary(svm_m)
confusionMatrix(pred_svm, test$status_group)
roc.curve(test$status_group, pred_svm, plot = TRUE)

# ------- SMOTE data --------
svm_ms <- svm(status_group~., data = NewData, kernel="radial", cost=60, gamma=0.04)
pred_svms=predict(svm_ms, test)
confusionMatrix(pred_svms, test$status_group)
roc.curve(test$status_group, pred_svms, plot = TRUE)


# - Logistic linear Regression
logit_m<-glm(status_group~., data = train, family = binomial)
test_prob <-predict(logit_m, test, type='response')
pred.logit <- rep('Good',length(test_prob))
pred.logit[test_prob>=0.5] <- 'Bad'
confusionMatrix(pred.logit, test$status_group)
roc.curve(test$status_group, pred.logit, plot = TRUE)

# ------- SMOTE data --------
logit_ms<-glm(status_group~., data = NewData, family = binomial)
test_probs <-predict(logit_ms, test, type='response')
pred.logits <- rep('Good',length(test_probs))
pred.logits[test_probs>=0.5] <- 'Bad'
confusionMatrix(pred.logits, test$status_group)
roc.curve(test$status_group, pred.logits, plot = TRUE)


# - Adaboost
ada_m <- adaboost(status_group~., data = train, 37)
pred_ada=predict(ada_m, test)
confusionMatrix(pred_ada$class, test$status_group)
roc.curve(test$status_group, pred_ada$class, plot = TRUE)

# ------- SMOTE data --------
ada_ms <- adaboost(status_group~., data = NewData, 37)
pred_adas=predict(ada_ms, test)
confusionMatrix(pred_adas$class, test$status_group)
roc.curve(test$status_group, pred_adas$class, plot = TRUE)