# Exercise 01-01 
# Train Predict Evaluate

# install.packages("rchallenge")
# install.packages("skimr")

library("rchallenge")
library(skimr)

data("german")
skimr::skim(german) # a broad overview of a data frame

# Splittin Data in Training and Test
set.seed(100L)
train_ids = sample(row.names(german), 0.7*nrow(german))
test_ids = setdiff(row.names(german), train_ids)

train_set = german[train_ids, ]
test_set = german[test_ids, ]

# install.packages("mlr3verse")
library("mlr3verse")

task <- TaskClassif$new(
  id = "german_credit", 
  backend = train_set, 
  target = "credit_risk",
  positive = "good"
  )

learner <- lrn("classif.log_reg", predict_type = 'prob')
model <- learner$train(task)
summary(model$model)

pred = model$predict_newdata(test_set)
pred$score() # classif.ce
pred$score(msr("classif.auc"))

as.data.table(msr())

# Exercise 01-02
# Resampling
taask = tsk("german_credit")
task$positive
install.packages("kknn")

log_reg = lrn("classif.log_reg", predict_type = 'prob')
knn = lrn("classif.kknn", k = 5)

cv5 = rsmp("cv", folds = 5)
res_log_reg <- resample(task = task, learner = log_reg, resampling = cv5)
res_knn <- resample(task = task, learner = knn_reg, resampling = cv5)

res_log_reg$aggregate(msr("classif.acc"))
res_knn$aggregate(msr("classif.acc"))

# Exercise 01-03
# Benchmarking
set.seed(20220801)

