################
# This is the final script used to form the ABT and model where 'coworkers' is the target feature.
################

# Load all necessary libraries
library(party)
library(rpart)
library(entropy)
library(arules)
library(devtools)
library(ggbiplot)
library(corrplot)
library(aplpack)
library(modes)
library(googleVis)
library(ggplot2)
library(plyr)
library(reshape2)
library(OneR)
library(mlbench)
library(e1071)
library(caret)
library(rpart)
library(randomForest)
library(RWeka)

################
# Data is loaded from the CSV file to a data frame.
################

mentalHealthData <- read.csv("C:/Users/miriamcallahan/Desktop/GitHub Projects/Data-Science-Project/survey.csv")
writeLines("\nMental Health in Tech Data Samples:")
print(head(mentalHealthData))

################
# The initial deletion of erroneous features/ normalization at first glance is done.
################

# Deletion of timestamp
mentalHealthData <- subset(mentalHealthData, select = -Timestamp)

# Deletion of respondents with erroneous ages
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Age != 329)
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Age != -1726) 
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Age != 5)
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Age != 8)
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Age != 11)
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Age != -1)

# Replacement of value "-29" to "29"
mentalHealthData[mentalHealthData$Age == -29, 1] <- 29 

################
# The 'comments' section is removed from the data so that it does not interfere with anything. The data
# in this section is too varied to be of any use.
################

# Removal of comments section
mentalHealthData <- subset(mentalHealthData, select = -comments)

################
# Since this was a global survey but many of the countries worldwide were significantly or completely 
# underrepresented, all respondents who were not from the US were deleted. Similarly, since this column 
# was no longer necessary, the 'country' feature was deleted as well.
################

mentalHealthData <- mentalHealthData[mentalHealthData$Country == "United States",]
mentalHealthData <- subset(mentalHealthData, select = -Country)

################
# Since participants could self-identify their gender, there were a wide range of values within this
# category. In order to prevent underrepresentation and make the classifier's job easier, these values
# were generalized into the following categories: Cis Female, Cis Male, Non-binary and Trans Female. Trans Male
# would also have been included but not respondents appeared to identify as such. Respondents with 
# genders that indicated they were not taking the survey seriously were also removed.
################

# Deletion of respondents with non-existent genders
mentalHealthData <- subset(mentalHealthData, mentalHealthData$Gender != "Nah")

# Normalization of cis female values
mentalHealthData[mentalHealthData$Gender == "Female", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "female", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "F", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "f", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "femake", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "woman", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "cis-female/femme", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "female (cis)", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "femail", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "femake", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "Woman", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "Femake", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "Female (cis)", 2] <- "Cis Female"
mentalHealthData[mentalHealthData$Gender == "Female ", 2] <- "Cis Female"

# Normalization of cis male values
mentalHealthData[mentalHealthData$Gender == "M", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Male", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "male", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "m", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "maile", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Mal", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Male (CIS)", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "make", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Make", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Man", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "msle", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Mail", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Malr", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "ostensibly male, unsure what that really means", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Cis Man", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "cis male", 2] <- "Cis Male"
mentalHealthData[mentalHealthData$Gender == "Male ", 2] <- "Cis Male"

# Levels are created for trans female and non-binary values
levels(mentalHealthData$Gender) <- c(levels(mentalHealthData$Gender), "Non-binary")
levels(mentalHealthData$Gender) <- c(levels(mentalHealthData$Gender), "Trans Female")

# Normalization of non-binary values
mentalHealthData[mentalHealthData$Gender == "Male-ish", 2] <- "Non-binary"
mentalHealthData[mentalHealthData$Gender == "queer/she/they", 2] <- "Non-binary"
mentalHealthData[mentalHealthData$Gender == "non-binary", 2] <- "Non-binary"
mentalHealthData[mentalHealthData$Gender == "Genderqueer", 2] <- "Non-binary"

# Normalization of trans female values
mentalHealthData[mentalHealthData$Gender == "Trans-female", 2] <- "Trans Female"
mentalHealthData[mentalHealthData$Gender == "Trans woman", 2] <- "Trans Female"
mentalHealthData[mentalHealthData$Gender == "Female (trans)", 2] <- "Trans Female"

mentalHealthData$Gender <- factor(mentalHealthData$Gender)

################
# Data is formulated to view potential issues of over/underrepresentation. Those which were thought
# to be potentially skewed were analyzed in the DQP.
################

table(mentalHealthData$Age)
table(mentalHealthData$Gender)
table(mentalHealthData$state)
table(mentalHealthData$self_employed)
table(mentalHealthData$family_history)
table(mentalHealthData$treatment)
table(mentalHealthData$work_interfere)
table(mentalHealthData$no_employees)
table(mentalHealthData$remote_work)
table(mentalHealthData$tech_company)
table(mentalHealthData$benefits)
table(mentalHealthData$care_options)
table(mentalHealthData$wellness_program)
table(mentalHealthData$seek_help)
table(mentalHealthData$anonymity)
table(mentalHealthData$leave)
table(mentalHealthData$mental_health_consequence)
table(mentalHealthData$phys_health_consequence)
table(mentalHealthData$mental_health_interview)
table(mentalHealthData$phys_health_interview)
table(mentalHealthData$mental_vs_physical)
table(mentalHealthData$obs_consequence)

################
# Data is rearranged since coworkers needed to be the last feature for testing in certain models and the supervisor
# feature is deleted as it is another target feature.
################

coworkerFeatureSubset <- mentalHealthData[c(1:19, 21:24)]
coworkerFeatureSubset <- coworkerFeatureSubset[, c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                                                   20, 21, 22, 23, 19)]
tableView = gvisTable(coworkerFeatureSubset)
plot(tableView)

################
# Data is split to create training and test subsets for feature selection. Real model accuracy will be tested with 
# 10-fold cross validation.
################

set.seed(0)
ts <- sample(seq_len(nrow(coworkerFeatureSubset)), nrow(coworkerFeatureSubset) * .6)
trainingSet <- coworkerFeatureSubset[ts,]
testSet <- coworkerFeatureSubset[-ts,]
print(trainingSet)
print(testSet)

################
# A 1R algorithm is used to test coworker data to reduce dimensionality and determine 
# which features are most important.
################

# Develop and test model
coworker1RModel <- OneR::OneR(trainingSet, verbose = TRUE)
print(coworker1RModel)
str(coworker1RModel)
summary(coworker1RModel)
Coworker1RPredictions <- predict(coworker1RModel, testSet)
eval_model(Coworker1RPredictions, testSet)

################
# A Naive Bayes algorithm is used to test coworker data to reduce dimensionality and determine 
# which features are most important.
################

# Develop and test model
coworkerNBModel <- naiveBayes(coworkers ~ ., data = trainingSet)
print(coworkerNBModel)
str(coworkerNBModel)
coworkerNBModelPredictions <- predict(coworkerNBModel, testSet)
eval_model(coworkerNBModelPredictions, testSet)
coworkerNBModel

################
# The RIPPER algorithm is used for feature selection.
################

coworkerRIPPERModel <- JRip(coworkers~., data = trainingSet)
summary(coworkerRIPPERModel)
coworkerRIPPERModelPredictions <- predict(coworkerRIPPERModel, testSet)
summary(coworkerRIPPERModelPredictions)
print(coworkerRIPPERModel)

################
# The C4.5 algorithm is used for feature selection (referred to as J48 in R).
################

coworkerC4.5Model <- J48(coworkers~., data = trainingSet)
summary(coworkerC4.5Model)
coworkerC4.5ModelPredictions <- predict(coworkerC4.5Model, testSet)
summary(coworkerC4.5ModelPredictions)
print(coworkerC4.5Model)

################
# The Random Forest algorithm is used to test for erroneous features. Those with a GINI index >10
# were considered important.
################

# Use rfImpute function on both training and test data to use RF algorithm
coworkerTrainingImputed <- rfImpute(coworkers ~ ., trainingSet)
coworkerTestImputed <- rfImpute(coworkers ~ ., testSet)
coworkerRFModel <- randomForest(coworkers ~., data = coworkerTrainingImputed, ntrees = 50,
                                importance = TRUE)
summary(coworkerRFModel)
coworkerRFModelPredictions <- predict(coworkerRFModel, coworkerTestImputed)
summary(coworkerRFModelPredictions)
print(coworkerRFModel)
importance(coworkerRFModel)

################
# Features are deleted from the coworker subset and age is factored to avoid errors within the loops. 
# during testing.
################

coworkerFeatureSubset <- subset(coworkerFeatureSubset, select = c(-Gender, -wellness_program,
                                                                  -seek_help, -leave, -phys_health_consequence, 
                                                                  -mental_vs_physical, -tech_company))
tableView = gvisTable(coworkerFeatureSubset)
plot(tableView)

################
# 10-fold cross validation is used to test the classifier with the OneR algorithm.
################

# Develop models and find percentage of correct predictions
k <- 10
splits <- runif(nrow(coworkerFeatureSubset))
results <- sapply (1:k, function(i)
{
  testIndex <- (splits >= (i - 1) / k) & (splits < i / k)
  trainingIndex <- !testIndex
  testSet <- coworkerFeatureSubset[testIndex,]
  trainingSet <- coworkerFeatureSubset[trainingIndex,]
  coworker1RModel <- OneR::OneR(trainingSet)
  predictions <- predict(coworker1RModel, testSet)
  
  # Align factor levels between testSet$coworkers and predictions
  lp <- levels(predictions)
  lt <- levels(testSet$coworkers)
  for (i in 1:length(lp)) 
  {
    if (!(lp[i] %in% lt)) 
    {
      levels(testSet$coworkers) <- c(levels(testSet$coworkers), lp[i])
    }
  }
  for (i in 1:length(lt)) 
  {
    if (!(lt[i] %in% lp)) 
    {
      levels(predictions) <- c(levels(predictions), lt[i])
    }
  }
  errorRate <- sum(testSet$coworkers != predictions) / nrow(testSet)
  return(1 - errorRate)
})
print(mean(results))

# Develop model
coworker1RModel <- OneR::OneR(coworkerFeatureSubset)
summary(coworker1RModel)

################
# 10-fold cross validation is used to test the classifier with the Naive Bayes algorithm.
################

# Develop models and find average percentage of correct predictions
k <- 10
splits <- runif(nrow(coworkerFeatureSubset))
results <- sapply (1:k, function(i)
{
  testIndex <- (splits >= (i - 1) / k) & (splits < i / k)
  trainingIndex <- !testIndex
  testSet <- coworkerFeatureSubset[testIndex,]
  trainingSet <- coworkerFeatureSubset[trainingIndex,]
  coworkerNBModel <- naiveBayes(coworkers ~ ., data = trainingSet)
  predictions <- predict(coworkerNBModel, testSet)
  
  # Align factor levels between testSet$coworkers and predictions
  lp <- levels(predictions)
  lt <- levels(testSet$coworkers)
  for (i in 1:length(lp)) 
  {
    if (!(lp[i] %in% lt)) 
    {
      levels(testSet$coworkers) <- c(levels(testSet$coworkers), lp[i])
    }
  }
  for (i in 1:length(lt)) 
  {
    if (!(lt[i] %in% lp)) 
    {
      levels(predictions) <- c(levels(predictions), lt[i])
    }
  }
  errorRate <- sum(testSet$coworkers != predictions) / nrow(testSet)
  return(1 - errorRate)
})
print(mean(results))

# Develop model
coworkerNBModel <- naiveBayes(coworkers ~ ., data = coworkerFeatureSubset)
print(coworkerNBModel)

################
# 10-fold cross validation is used to test the classifier with the RIPPER algorithm.
################

# Develop models and find average number of correct predictions
k <- 10
splits <- runif(nrow(coworkerFeatureSubset))
results <- sapply (1:k, function(i)
{
  testIndex <- (splits >= (i - 1) / k) & (splits < i / k)
  trainingIndex <- !testIndex
  testSet <- coworkerFeatureSubset[testIndex,]
  trainingSet <- coworkerFeatureSubset[trainingIndex,]
  coworkerRIPPERModel <- JRip(coworkers ~ ., data = trainingSet)
  predictions <- predict(coworkerRIPPERModel, testSet)
  
  # Align factor levels between testSet$coworkers and predictions
  lp <- levels(predictions)
  lt <- levels(testSet$coworkers)
  for (i in 1:length(lp)) 
  {
    if (!(lp[i] %in% lt)) 
    {
      levels(testSet$coworkers) <- c(levels(testSet$coworkers), lp[i])
    }
  }
  for (i in 1:length(lt)) 
  {
    if (!(lt[i] %in% lp)) 
    {
      levels(predictions) <- c(levels(predictions), lt[i])
    }
  }
  errorRate <- sum(testSet$coworkers != predictions) / nrow(testSet)
  return(1 - errorRate)
})
print(mean(results))

# Develop model
coworkerRIPPERModel <- JRip(coworkers ~ ., data = coworkerFeatureSubset)
summary(coworkerRIPPERModel)
print(coworkerRIPPERModel)

################
# 10-fold cross validation is used to test the classifier with the C4.5 algorithm.
################

# Develop models and find average number of correct predictions
k <- 10
splits <- runif(nrow(coworkerFeatureSubset))
results <- sapply (1:k, function(i)
{
  testIndex <- (splits >= (i - 1) / k) & (splits < i / k)
  trainingIndex <- !testIndex
  testSet <- coworkerFeatureSubset[testIndex,]
  trainingSet <- coworkerFeatureSubset[trainingIndex,]
  coworkerC4.5Model <- J48(coworkers ~ ., data = trainingSet)
  predictions <- predict(coworkerC4.5Model, testSet)
  
  # Align factor levels between testSet$coworkers and predictions
  lp <- levels(predictions)
  lt <- levels(testSet$coworkers)
  for (i in 1:length(lp)) 
  {
    if (!(lp[i] %in% lt)) 
    {
      levels(testSet$coworkers) <- c(levels(testSet$coworkers), lp[i])
    }
  }
  for (i in 1:length(lt)) 
  {
    if (!(lt[i] %in% lp)) 
    {
      levels(predictions) <- c(levels(predictions), lt[i])
    }
  }
  errorRate <- sum(testSet$coworkers != predictions) / nrow(testSet)
  return(1 - errorRate)
})
print(mean(results))

# Develop model
coworkerC4.5Model <- J48(coworkers ~ ., data = coworkerFeatureSubset)
summary(coworkerC4.5Model)
print(coworkerC4.5Model)
plot(coworkerC4.5Model)