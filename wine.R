library(gclus)
library(class)
library(e1071)
library(mclust)
set.seed(1234)
data(wine)
#--------------a-------------------
obs <- 1:nrow(wine)
train <- sample(obs, 130)
train_val <- wine[train, ]
test <- wine[-train, ]
#-----------------b----------------
wine_combined <- rbind(train_val, test)
predictors <- as.matrix(wine_combined[, -1])
predictors_scaled <- scale(predictors)
class <- as.factor(wine_combined[, 1])
train_val_scaled <- predictors_scaled[1:nrow(train_val), ]
test_scaled <- predictors_scaled[(nrow(train_val) + 1):nrow(predictors_scaled), ]
train_val_class <- class[1:nrow(train_val)]
test_class <- class[(nrow(train_val) + 1):nrow(predictors_scaled)]
#-----------------c---------------
k_values <- 1:12
tune <- tune.control(cross = 10)
tune_result <- tune.knn(train_val_scaled, train_val_class, k = k_values, tunecontrol = tune)
optimal_k <- tune_result$best.parameters$k
optimal_k
#------------------e----------------------
errors <- tune_result$performances[, 1]
plot(k_values, errors, type = "b", xlab = "K", ylab = "Error", main = "KNN Classification Error vs.
K")
#--------------------f-------------------
test_scaled <- scale(test[, -1], center = colMeans(train_val[, -1]), scale = apply(train_val[, -1], 2,
                                                                                   sd))
test_pred <- knn(train = train_val_scaled, test = test_scaled, cl = train_val_class, k = optimal_k)
#------------------------g----------------------
conf_matrix <- table(test_pred, test[, 1])
TP <- diag(conf_matrix)
FP <- colSums(conf_matrix) - TP
FN <- rowSums(conf_matrix) - TP
TN <- sum(conf_matrix) - TP - FP - FN
conf_matrix
#---------------------h----------------
misclassification_rate <- 1 - sum(TP) / sum(conf_matrix)
misclassification_rate
#---------------i------------------
ARI <- adjustedRandIndex(test[, 1], test_pred)
ARI
