## Lets CLear the Environment First
rm(list = ls(all=T))

## Setting up the Working Directory
setwd("E:/DataScienceEdwisor/Rscripts")

## Lets Check Our Present working directory
getwd()

## Lets Load Required Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information", 
      "MASS", "rpart", "gbm", "ROSE", 'DataCombine','sampling','pROC','ROCR')

## Lets Load the Packages
lapply(x, require, character.only = TRUE)
rm(x)

## Lets Read The Datasets Train & Test
train_actual = read.csv("~/Train_data.csv", header = T, na.strings = c(" "," ", "NA"))
test_actual  = read.csv("~/Test_data.csv",header = T, na.strings = c(" ", " ","NA"))

# Lets Create another instances copy of our Train&Test datasets on which we are going to work
train = train_actual
test = test_actual

# Lets Combine the train&test datasets to check the total observations of the combined dataset
data = rbind(train, test)

#*******************************************************************************************************#

#************************************** EXPLORATORY DATA ANALYSIS***************************************#

#******************************************************************************************************#

# Lets View the Combined 'data' dataset and its total observations
View(data) # Consists 5000 observations & 21 Features || Here Last Column var is CHURN-> False. & True.

# Lets Check the Dimension of the dataset
dim(data)

# Lets View the data
View(train) # Consists 3333 observations & 21 Features || Here Last Column var is CHURN-> False. & True.

# Dimensions of the train data
dim(train)

# Lets Check the Structure of the train data
str(train)

# Lets CHeck the summary of thetrain data
summary(train) # Gives us back a brief summary stats of all numeric cols 

# Lets Check the column names of train data
colnames(train) # As we can see that in excel there was space in between col names but in r synatx has provided dot(.) in between 

# Lets check the class of train dataset
class(train)

# Now lets Check the Unique Values of each count
apply(train, 2,function(x) length(table(x)))

# lets drop phone number as we see it is showing 3333 entries and their are no unique values in it
# Lets drop in both train & test datasets
train$phone.number = NULL
test$phone.number = NULL

# Lets first change factor column variables to category in train dataset
train$state = as.factor(train$state)
train$area.code = as.factor(train$area.code)
train$international.plan = as.factor(train$international.plan)
train$voice.mail.plan = as.factor(train$voice.mail.plan)
train$Churn = as.factor(train$Churn)

# Lets also do the same for our test dataset also
test$state = as.factor(test$state)
test$area.code = as.factor(test$area.code)
test$international.plan = as.factor(test$international.plan)
test$voice.mail.plan = as.factor(test$voice.mail.plan)
test$Churn = as.factor(test$Churn)

# Lets the rate of churn count & in precent
table(train$Churn)

# False.   True. 
# 2850     483 
prop.table(table(train$Churn))
# False.     True. 
# 85.50  14.49 

#*******************************************************************************************************#

#**************************************MISSING VALUE ANALYSIS *****************************************#

#******************************************************************************************************#
# Lets Check missing values in our Data dataset so that we dont need to check seperately in both train&test 
# here 2 indicates that is is column & 1 means row
apply(data, 2, function(x) {sum(is.na(x))}) # As we can see that there are no missing values in our dataset


#*******************************************************************************************************#

#***************************ANALYZING DATA THROUGH VISUALIZATION ***************************************#

#**************************************** BAR PLOT ANALYSIS ********************************************#

# Lets Plot Bar plots using ggplot2 library 
# Count of Churn in False and True
ggplot(data=train, aes(Churn, fill=Churn)) + geom_bar(stat='count',fill='DarkSlateBlue') +
  labs(x='Churn or Not Churn', y='Count of Churn') + ggtitle("Customer Churn")

# Lets see State wise churn of customers  
ggplot(train, aes(state, fill = Churn)) + geom_bar(position = "fill") + 
  labs(title = "State Wise Churn of customers") 

# Lets see Area code wise churn of customers  
ggplot(train, aes(area.code, fill = Churn)) + geom_bar(position = "fill") + 
  labs(title = "Area Code Wise Churn of customers") 

# Lets see Voice Mail Plan wise churn of customers  
ggplot(train, aes(voice.mail.plan, fill = Churn)) + geom_bar(position = "fill") + 
  labs(title = "Voice Mail Plan Wise Churn of customers") 

# Lets see Voice Mail Plan wise churn of customers  
ggplot(train, aes(international.plan, fill = Churn)) + geom_bar(position = "fill") + 
  labs(title = "International Plan Wise Churn of customers")  

# Lets see Voice Mail Plan wise churn of customers  
ggplot(train, aes(number.customer.service.calls, fill = Churn)) + geom_bar(position = "fill") + 
  labs(title = "Customer Service Calls Wise Churn of customers") 

#**************************************** DENSITY PLOT ANALYSIS ********************************************#

# Lets see the density plot for account length & number.voice.mail messages For Checking Churn Predicatability from these features

accntlngth <- ggplot(train, aes(account.length, fill = Churn)) + geom_density(alpha = 0.7) +
  theme(legend.position = "null")
nvmailmsgs <- ggplot(train, aes(number.vmail.messages, fill = Churn)) + geom_density(alpha = 0.7) + 
  theme(legend.position = "null")

gridExtra:: grid.arrange(accntlngth, nvmailmsgs, ncol = 2, nrow = 1)

# Lets see the density plot for internationalPlan & Voice Mail Plan For Checking Churn Predicatability from these features

intlplan  <- ggplot(train, aes(international.plan, fill = Churn)) + geom_density(alpha = 0.7) +
  theme(legend.position = "null")
vmailplan <- ggplot(train, aes(voice.mail.plan, fill = Churn)) + geom_density(alpha = 0.7) + 
  theme(legend.position = "null")

gridExtra:: grid.arrange(intlplan, vmailplan, ncol = 2, nrow = 1)


# Lets see the density plot for arecode & customerservicecalls For Checking Churn Predicatability from these features

areacode     <- ggplot(train, aes(area.code, fill = Churn)) + geom_density(alpha = 0.7) +
  theme(legend.position = "null")
custmrsrvcls <- ggplot(train, aes(number.customer.service.calls, fill = Churn)) + geom_density(alpha = 0.7) + 
  theme(legend.position = "null")

gridExtra:: grid.arrange(areacode, custmrsrvcls, ncol = 2, nrow = 1)

#*******************************************************************************************************#

#********************************************* OUTLIER ANALYSIS ****************************************#

#******************************************************************************************************#

# Before we do outlier analysis lets save all the continous variables seperately
# As boxplot analysis is only applicable on continous variable
numeric_index = sapply(train, is.numeric) # selects only numeric
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data) 

# Loop For Detecting Outliers
#for (i in 1:length(cnames)) {
#assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(train))+ 
#       stat_boxplot(geom = "errorbar", width = 0.5) +
#       geom_boxplot(outlier.colour="red", fill = "skyblue" ,outlier.shape=18,
#                       outlier.size=1, notch=FALSE) +
#       labs(y=cnames[i],x="Churn")+
#       ggtitle(paste("Box plot of Churn for",cnames[i])))
#}

## Now Lets Plot, plots togeather
#gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
#gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
#gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
#gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
#gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)

## Lets Remove Outliers & then Replace all outliers with NA and impute with KNN Method
# Loop to remove outliers from the whole train data
#for(i in cnames){
#   print(i)
#   oa = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
#   print(length(oa))
#   train[,i][train[,i] %in% oa] = NA
#}

# Lets Check the count of missing values in our data now
#sum(is.na(train)) # 682 missing values after dropping outliers
# Now lets impute values with KNN method
#train = knnImputation(train, k = 3)

## Now Lets Plot again to see, Again as we have dropped outliers
#gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
#gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
#gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
#gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
#gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)

# Againg after plotting the grids here we can observe that most of the outliers have been removed except from 
# number.vmail.messges may be not considering for removing as outliers.
# So as there is already target class imbalance so i am going to skip Outlier Analysis for further model development

#*******************************************************************************************************#

#********************************************* FEATURE SELECTION ***************************************#

#*******************************************************************************************************# 

# Lets do correlation analysis using corrgram library
corrgram(train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, 
         main = "Checking Correlation Between Features")

# Dark blue color indicated that these variables are highly correlated with each

# heatmap plot for numerical features using corrplot library for checking collinearity
library(corrplot)
corrplot(cor(train[sapply(train, is.numeric)]))

# From Correlation plot here we can clearly observe that total- day,evening,night&international-charge are 
#highly correlated with total-day,evening,night&international-minutes, so here i am going to drop charges variables


# Lets save categorical columns seperately 
cat_names = c('state','area.code','international.plan','voice.mail.plan')

# Lets Do Chi2-Square Test of Independence by using for loop
for (i in cat_names)
{
  print(i)
  print(chisq.test(table(train$Churn,train[,i])))
}

# As we can see that "area.code" value i grater than p =0.05, i.e; p-value = 0.9151

#*******************************************************************************************************#

#********************************************* DIMENSION REDUCTION *************************************#

#*******************************************************************************************************#
# Lets drop the fetaures highly correlated and the features which are not contributing much information
# From both train & test datasets

train = subset(train,select= -c(state,area.code,total.day.charge,total.eve.charge,
                                total.night.charge,total.intl.charge))

test = subset(test,select= -c(state,area.code,total.day.charge,total.eve.charge,
                              total.night.charge,total.intl.charge))

# Lets assign levels to categorical columns for bot train&test datasets
# After assigning the levels here 'No'-> 0 & 'Yes'-> 1
# Same as above 'False.' -> 0 & 'True.'-> 1
cat_names = c('international.plan','voice.mail.plan','Churn')
for (i in cat_names){
  levels(train[,i]) <- c(0,1)
  levels(test[,i]) <- c(0,1)
}

# Lets Update our continous variables as we have dropped some of the columns
numeric_index = sapply(train, is.numeric) # selects only numeric
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data) 


# Lets Do Normality Check First Using Histogram plot for all the continous variablecolumns 
hist(train$account.length)
hist(train$number.vmail.messages)
hist(train$total.day.minutes)
hist(train$total.day.calls)
hist(train$total.eve.minutes)
hist(train$total.eve.calls)
hist(train$total.night.minutes)
hist(train$total.night.calls)
hist(train$total.intl.minutes)
hist(train$total.intl.calls)
hist(train$number.customer.service.calls)
# As we can observe that most of the data distibution here is uniformly distributed is it is better to go for 
# Standardization/Z-Score Method for Feature Scaling of features/Predictors
#*******************************************************************************************************#

#********************************************* FEATURE SCALING *****************************************#

#************************************STANDARDIZTION/ Z-SCORE METHOD*************************************#
# Lets aplly for both train & test datasets at a time itself
for (i in cnames) {
  print(i)
  train[,i] = (train[,i] - mean(train[,i]))/sd(train[,i]) 
  test[,i] = (test[,i] - mean(test[,i]))/sd(test[,i])
}

# So here after applying standardization we can observe that it has transformed the data to have 0->mean & the unit variance
# That is it will take the data point range from negative to positive as we can see here. further explained in report.


# Lets Clean-up the environment before we proceed further for model development

rmExcept(c("train_actual","test_actual","train","test"))

#************************************SAMPLING OF DATA USING STRATIFIED SAMPLING METHOD*************************************#
#Divide data into train and test using stratified sampling method

set.seed(1234)
train.index = createDataPartition(train$Churn, p = .70, list = FALSE)
train = train[ train.index,]
test  = train[-train.index,]

# Lets Check the target value churn count

table(train$Churn)

# As we can observe that there is target class imbalance problem here 
# 0    1 
# 1995  339 


#**************************************************ROSE***********************************************************#

# ROSE(Random Over Sampling Examples) package which helps us to generate synthetic artificial data

train_res = ROSE(Churn ~ ., data = train, p = 0.5, seed = 1)$data

# Lets CHeck the Churn Count of values in our train dataset 

table(train_res$Churn)

# 0    1 
# 1218 1116

#*******************************************************************************************************#

#********************************************* MODEL DEVELOPMENT ***************************************#

#*******************************************************************************************************#

# Lets Define a Function for Evaluating Error Metrics so that we dont need to eavluate for each model seperately

ErrorMetrics <- function(CM){
  TN=CM$table[1,1]
  FP=CM$table[1,2]
  FN=CM$table[2,1]
  TP=CM$table[2,2]
  # Accuracy
  print(paste0('Accuracy:- ' ,((TP+TN)*100/(TN+FN+TP+FP))))
  # False Negative Rate
  print(paste0('False Negative Rate:-  ' ,((FN*100)/(FN+TP))))
  # False Positive Rate
  print(paste0('False Positive Rate:-  ' ,((FP*100)/(FP+TN))))
  # Sensitivity-TruePositiveRate-Recall
  print(paste0('Sensitivity//TPR//Recall:-  ' ,((TP*100)/(TP+FN))))
  # Specificity-TrueNegativeRate
  print(paste0('Specificity//TNR:-  ' ,((TN*100)/(TN+FP))))
  # Precision
  print(paste0('Precision:-  ' ,((TP*100)/(TP+FP))))
}
############################################# DECISION TREE MODEL #########################################################################
# Lets Develop Decision Tree Model on training data
DT_Model = C5.0(Churn ~., data = train_res, trials = 100, rules = TRUE)

# Lets see the summary of the model
summary(DT_Model)

# Lets write the results back into our hard-disk for further analysis
#write(capture.output(summary(DT_Model)), "DTRules.txt")

# Lets Predit for new test cases
DT_Predictions = predict(DT_Model, test[,-14], type = 'class')
DT_ConfMatrix = table(actualCases = test$Churn, predictedCases = DT_Predictions)

# Lets Evaluate Decision Tree model now by building Confusion Matrix
DT_CM = confusionMatrix(DT_ConfMatrix)

print(DT_CM)

# Lets Evaluate Decision Tree Error Metrics by passing it to the ErrorMetrics Function
ErrorMetrics(DT_CM)

# ERROR METRICS OF DECISION TREE MODEL
# [1] "Accuracy:------------------> 88.3357041251778"
# [1] "False Negative Rate:-------> 15.4545454545455"
# [1] "False Positive Rate:-------> 10.9612141652614"
# [1] "Sensitivity//TPR//Recall:->  84.5454545454545"
# [1] "Specificity//TNR:--------->  89.0387858347386"
# [1] "Precision:---------------->  58.8607594936709"

# Lets Plot AUC-ROC Curve for Decision Tree Model
roc.curve(test$Churn, DT_Predictions) # Area under the curve (AUC): 86.8

##################################### RANDOM FOREST MODEL ##################################################################
RF_Model = randomForest(Churn ~ ., train_res, importance = TRUE, ntree = 500)

# Lets Extract Rules From RandomForest model Now
#transform rf object to an inTrees' format
library(inTrees)
treeList = RF2List(RF_Model)

# Lets Extract the rules now
extract = extractRules(treeList, train_res[,-14])
# 4498 rules (length<=6) were extracted from the first 100 trees.

# Lets Visualize some rules
extract[1:2,]

# Lets make rules more readable form
readableRules = presentRules(extract, colnames(train_res))
readableRules[1:2,]

# Lets Get few rule metrics
ruleMetrics = getRuleMetric(extract, train_res[,-14], train_res$Churn)

# Lets Evaluate few rule metrics
ruleMetrics[1:2,]

# Now lets predict new test cases
RF_Predictions = predict(RF_Model, test[,-14])

RF_ConfMatrix = table(actualCases = test$Churn, predictedCases = RF_Predictions)

# Lets build confusion matrix for random forest model
RF_CM = confusionMatrix(RF_ConfMatrix)

print(RF_CM)

# Lets Evaluate Random Forest Model Error Metrics by passing it to the ErrorMetrics Function
ErrorMetrics(RF_CM) 

# ERROR METRICS OF RANDOM FOREST MODEL
# [1] "Accuracy:------------------> 89.7581792318634"
# [1] "False Negative Rate:-------> 11.8181818181818"
# [1] "False Positive Rate:-------> 9.94940978077572"
# [1] "Sensitivity//TPR//Recall:->  88.1818181818182"
# [1] "Specificity//TNR:--------->  90.0505902192243"
# [1] "Precision:---------------->  62.1794871794872"


# Lets Plot AUC-ROC Curve for Random Forest Model
roc.curve(test$Churn, RF_Predictions) # Area under the curve (AUC): 89.1

##################################### LOGISTIC REGRESSION MODEL ##################################################################
# Lets Develop Logistic Regression Model
LR_Model = glm(Churn ~ ., train_res, family = 'binomial')

# Lets Check the summary 
summary(LR_Model)
# Output:- international.plan1, voice.mail.plan1, total.day.minutes, total.eve.minutes, total.night.minutes
# total.intl.minutes, number.customer.service.calls are very much significant

# Lets Predict new test cases
LR_Predictions = predict(LR_Model, newdata = test, type = 'response')

# Lets Convert them to probabilities
LR_Predictions = ifelse(LR_Predictions > 0.5, 1, 0)

# Lets evaluate the performance of the model 
LR_ConfMatrix = table(actualCases = test$Churn, predictedCases = LR_Predictions)

# Lets build confusion matrix for Logistic Regression model
LR_CM = confusionMatrix(LR_ConfMatrix)
print(LR_CM)

# Lets Evaluate Logistic Regression Model Error Metrics by passing it to the ErrorMetrics Function
ErrorMetrics(LR_CM) 

# ERROR METRICS OF LOGISTIC REGRESSION MODEL
# [1] "Accuracy:------------------> 80.2275960170697"
# [1] "False Negative Rate:-------> 27.2727272727273"
# [1] "False Positive Rate:-------> 18.3811129848229"
# [1] "Sensitivity//TPR//Recall:->  72.7272727272727"
# [1] "Specificity//TNR:--------->  81.6188870151771"
# [1] "Precision:---------------->  42.3280423280423"

# Lets Plot AUC-ROC Curve for Logistic Regression Model
roc.curve(test$Churn, LR_Predictions) # Area under the curve (AUC): 77.2

#################################### K- NEAREST NEIGHBOR MODEL ##################################################################
# Lets Develop KNN  MODEL
library(class)

# Lets Predict the test data
KNN_Predictions = knn(train_res[, 1:13], test[, 1:13], train_res$Churn, k = 5)

# Lets Build Confusion Matrix for KNN MODEL
KNN_ConfMatrix = table(KNN_Predictions, test$Churn)
KNN_CM = confusionMatrix(KNN_ConfMatrix)
print(KNN_CM)

# Lets Evaluate K Nearest Neighbor Error Metrics by passing it to the ErrorMetrics Function
ErrorMetrics(KNN_CM) 

# ERROR METRICS OF K NEAREST NEIGHBOR MODEL
# [1] "Accuracy:------------------> 86.4864864864865"
# [1] "False Negative Rate:-------> 45.2830188679245"
# [1] "False Positive Rate:-------> 4.22794117647059"
# [1] "Sensitivity//TPR//Recall:->  54.7169811320755"
# [1] "Specificity//TNR:--------->  95.7720588235294"
# [1] "Precision:---------------->  79.0909090909091"

# Lets Plot AUC-ROC Curve for KNN Model
roc.curve(test$Churn, KNN_Predictions) # Area under the curve (AUC): 83.5


#################################### NAIVE BAYES MODEL ##################################################################
# Lets Develop Naive Bayes MOdel
NB_Model = naiveBayes(Churn ~ ., data = train_res)

# Lets Predict new test cases
NB_Predictions = predict(NB_Model, test[,1:13], type = 'class')

# Lets Build Confusion Matrix for Naive Bayes Model
NB_ConfMatrix = table(NB_Predictions, test$Churn)
NB_CM = confusionMatrix(NB_ConfMatrix)
print(NB_CM)

# Lets Evaluate Naive Bayes Model Error Metrics by passing it to the ErrorMetrics Function
ErrorMetrics(NB_CM) 

# ERROR METRICS OF NAIVE BAYES MODEL
# [1] "Accuracy:------------------> 86.9132290184922"
# [1] "False Negative Rate:-------> 43.6619718309859"
# [1] "False Positive Rate:-------> 5.3475935828877"
# [1] "Sensitivity//TPR//Recall:->  56.3380281690141"
# [1] "Specificity//TNR:--------->  94.6524064171123"
# [1] "Precision:---------------->  72.7272727272727"

# Lets Plot AUC-ROC Curve for KNN Model
roc.curve(test$Churn, NB_Predictions) # Area under the curve (AUC): 81.1

# Variable importance
library(caret)
varImp(RF_Model)
#Lets Plot the Variable Importance
varImpPlot(RF_Model, type = 2)


# Lets save the results back to hard disk
## Setting up the Working Directory
write.csv(train_res, "train_data.csv", row.names = F)
write.csv(test, "test_data.csv", row.names = F)
