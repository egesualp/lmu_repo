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
task = tsk("german_credit")
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
# Compare identical learners with different hyperparameters
set.seed(20220801)

# Comparing a range of different KNN models ranging from a k = 3 to k = 30
# Assesing the performance of logistic regression

# Step 1: Creating the learners
# We create a list for KNN learners
lrn_list <- lapply(3:33, function(x) lrn("classif.kknn", k = x, id = paste0("classif.knn", x)))
# Appending logistic learner
lrn_list <- append(lrn_list, lrn("classif.log_reg", predict_type = 'prob'))

# Create a 4-fold CV resampling
cv4 = rsmp("cv", folds = 4)

# Usage of benchmark_grid
# For this instance, we'll use only one task and one resampling tech with multiple learners
task = tsk("german_credit")
task$positive

# set up the design and execute benchmark
design = benchmark_grid(task, lrn_list, cv4)
bmr = benchmark(design)

# Evaluating benchmark: Choose two appropriate metrics to evaluate the different learners performance
bmr_agg = bmr$aggregate(c(msr('classif.acc'), msr('classif.bbrier'), msr('classif.logloss'))) 
# can always set loss measure, check options: msr()
bmr_agg[,c('learner_id', 'classif.acc', 'classif.bbrier', 'classif.logloss')]

# Compyute and visualize at least one metric with autoplot
library(mlr3viz)
autoplot(bmr, measure = msr('classif.acc'))

# Parallelization of benchmark function
future::plan('multicore')
  # runs each resampling iteration as a job
  # also allows nested resampling

learner$help()

# Exercise 01-04
# Decision trees and random forest

# Vanilla training
task = tsk('german_credit')
learner = lrn('classif.rpart')
model = learner$train(task)

# Checking the raw model object
model$model

# Visualizing the tree structure
?rpart::plot.rpart # Checking the example

library("rpart")
plot(learner$model)
text(learner$model)

# Fitting a random forest
# One of the drawbarcks of using trees is the instability of the predictor.
# Small changes in the data may lead to very different model
# and therefore a high variance of the predictions. The random forest
# takes advantages of that and reduces the variance by applying bagging to decision trees.

learner = lrn('classif.ranger') # fitting random forest
model = learner$train(task)

# Understanding hyperparams
  # List of random forest learners with different fraction of variables
ranger_list = lapply(
  seq(0.1,1,0.1),
  function(i) lrn(
    'classif.ranger', 
    mtry.ratio = i,
    id = paste0('classif.ranger', i))
  )

ranger_list = append(ranger_list, lrn('classif.ranger'))

  # List of decision trees with different depths
rpart_list = lapply(
  seq(5,30,5),
  function(i) lrn(
    'classif.rpart',
    maxdepth = i,
    id = paste0('classif.rpart', i)
  )
)

rpart_list = append(rpart_list, lrn('classif.rpart'))
rpart_list = append(rpart_list, lrn('classif.rpart', maxdepth = 1, id = paste0('classif.rpart', 1) ))

# Now benchmarking process with 5-fold CV
cv5 = rsmp('cv', folds = 5)
all_learners = c(ranger_list, rpart_list)
design = benchmark_grid(
  task = task, 
  learners = all_learners, 
  resamplings = cv5
  )

bmr = benchmark(design)
bmr_agg = bmr$aggregate(msr('classif.ce'))
bmr_agg[,c('learner_id', 'classif.ce')]

mlr3viz::autoplot(bmr, measure = msr('classif.ce'))
