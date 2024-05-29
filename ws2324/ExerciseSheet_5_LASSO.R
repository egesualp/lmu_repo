# Question 6 LASSO Optımızatıon Exercıse
library(glmnet)

# Replace "your_file.txt" with the actual path to your file
file_path <- "C:\\Users\\esual\\OneDrive\\Documents\\StaDS\\lmu_repo\\data\\model_selection_week_four.txt"

# Read the data with read.table
soep <- read.table(file_path, header = TRUE, sep = " ")

# View the structure of your data
str(soep)

# View the first few rows of your data
head(soep)

n <- nrow(soep)
p <- ncol(soep) - 1

# Train test split
set.seed(123)
ind_train <- sample(x=1:n, size = ceiling(0.8*n))
set_train <- soep[ind_train, ]
ind_test <- setdiff(x=1:n, ind_train)
set_test <- soep[ind_test, ]

model_lasso <- glmnet(
  x = as.matrix(set_train[, -1]),
  y = set_train$ginc,
  alpha = 1
)

model_lasso_log <- glmnet(
  x = as.matrix(set_train[, -1]),
  y = log(set_train$ginc),
  alpha = 1
)

sum_lasso <- summary(model_lasso)
sum_lasso_log <- summary(model_lasso_log)

lambda_lasso <- cv.glmnet(
  x = as.matrix(set_train[, -1]),
  y = set_train$ginc, alpha = 1
)$lambda.min

lambda_lasso_log <- cv.glmnet(
  x = as.matrix(set_train[, -1]),
  y = log(set_train$ginc), alpha = 1
)$lambda.min

sum_lasso
sum_lasso_log

coef(model_lasso_log, s=lambda_lasso_log)

# linear regression
model_unpenalized <- lm(formula = log(ginc) ~ ., data = set_train)
summary(model_unpenalized)
confint(model_unpenalized)

# fit on training data -> to calculate MSE
y_train <- log(set_train$ginc)
predict_train <- matrix(data = 0, nrow = nrow(set_train), ncol = 2)
predict_train[,1] <- predict(
  object = model_unpenalized,
  new_data = set_train[,-1]
)
predict_train[,2] = predict.glmnet(
  object = model_lasso_log,
  newx = as.matrix(set_train[, -1]),
  s = lambda_lasso_log
)

MSE_train <- rep(x=0, length.out = 2)
for (i in 1:2) {
  MSE_train[i] = mean((y_train - predict_train[, i])^2)
}

MSE_train

# fitting on the test data:
y_test <- log(set_test$ginc)
predict_test <- matrix(data = 0, nrow = nrow(set_test), ncol = 2)
predict_test[, 1] <- predict(object = model_unpenalized,
                             newdata = set_test[, -1])
predict_test[,2] <- predict.glmnet(
  object = model_lasso_log,
  newx = as.matrix(set_test[,-1]),
  s=lambda_lasso_log
)
MSE_test <- rep(x=0, length.out = 2)
for (i in 1:2) {
  MSE_test[i] = mean((y_test - predict_test[, i])^2)
}

MSE_test

# Cross Validation (KFold) to choose the best lambda
# Cross validation by hand
n_train <- nrow(set_train)
# lambda sequence
lambda <- seq(from = 0.001, to = .1, by = 0.001)
# initialitze cv criteria
cv_lasso <- rep(x = 0, length.out = length(lambda))
# cv
for (i in seq_along(lambda)) {
  rest <- 1:n_train
  for (k in 1:5) {
    # create test and training data:
    indices_test <- sample(x = rest, size = n_train*0.2)
    indices_train <- setdiff(x = 1:n_train, y = indices_test)
    rest <- setdiff(rest, indices_test)
    set_train_cv <- set_train[indices_train, ]
    set_test_cv <- set_train[indices_test, ]
    # prediction by lambda
    predict_lasso <- predict.glmnet(object = glmnet(as.matrix(set_train_cv[, -1]),
                                                    log(set_train_cv$ginc), alpha = 1),
                                    newx = as.matrix(set_test_cv[, -1]),
                                    s = lambda[i])
    # SSE
    cv_lasso[i] <- cv_lasso[i] + sum((log(set_test_cv$ginc) - predict_lasso)^2)
  }
}
# MSEs:
cv_lasso <- cv_lasso / n_train
cv_lasso

# pick lambda:
lambda_lasso_cv <- lambda[which(cv_lasso == min(cv_lasso))]
lambda_lasso_cv

# glmnet optimal lambda:
lambda_lasso <- cv.glmnet(x = as.matrix(set_train[, -1]),
                          y = log(set_train$ginc), alpha = 1)$lambda.min
lambda_lasso

# coefficient path:
library(plotmo)
plot_glmnet(x = model_lasso_log, label = TRUE, xvar = "lambda")

# Categorical Data
# Creating the data (using cut() to create the factor variable)
data_cat <- soep
data_cat$age <- cut(data_cat$age, breaks = c(0,35,55,100),
                    labels = c("low_age","medium_age","high_age"))
data_cat$height <- cut(data_cat$height, breaks = c(0,155,165,175,185,210))

# creating dummy variables for usage in the LASSO models
# using the model.matrix function
data_cat <- as.data.frame(
  cbind("ginc" = data_cat$ginc,
        model.matrix(ginc ~ ., data = data_cat)[,-1]))

# fit of the LASSO model
model_lasso_cat <- glmnet(x = as.matrix(data_cat[, -1]), y = log(data_cat$ginc),
                          alpha = 1)
lambda_min_cat <- cv.glmnet(x = as.matrix(data_cat[, -1]), y = log(data_cat$ginc),
                            alpha = 1)$lambda.min
coef(model_lasso_cat, s = lambda_min_cat)

# fit of the grouped LASSO model
library(gglasso)
model_glasso_cat <- gglasso(x = as.matrix(data_cat[, -1]), y = log(data_cat$ginc),
                            group = c(rep(1,4), rep(2,2), 3:7))
lambda_mingl_cat <- cv.gglasso(x = as.matrix(data_cat[, -1]), y = log(data_cat$ginc),
                               group = c(rep(1,4), rep(2,2), 3:7))$lambda.min
coef(model_glasso_cat, s = lambda_mingl_cat)
