library(mlr3verse)
library(mlbench)
library(data.table)
set.seed(7832)

# 1. Stratified Resampling

# In classification tasks, the ratio of the target class dist. should be similar
# in each train / test split, which is achieved by stratification.

# particularly useful in the case of the imbalanced classes and small data sets

task_gc = tsk("german_credit")
task_gc$target_names
task_gc$col_roles

# Do we need statum?
as.data.frame(table(task_gc$data()$credit_risk)) # 70% good 30% bad
task_gc$col_roles$stratum <- "credit_risk"
task_gc$strata

# Specify a 3-fold CV
cv <- rsmp('cv', folds=3)
# Instantiate the resampling on the task
cv$instantiate(task_gc)
cv$instance

# Chcking each fold
as.data.frame(table(cv$instance$fold))

# As a sanity chck, the target class distribution should be
# similar within each CV fold.

dt <- merge(
  cv$instance,
  transform(task_gc$data(),
            row_id = seq_len(nrow(task_gc$data()))),
            by = "row_id")

aggregate(
  credit_risk ~ fold,
  data = dt,
  FUN = function(x) sum(x == "bad") / sum(x == "good")
)

# 2. Block resampling

data("BreastCancer", package = "mlbench")
task_bc = as_task_classif(
  BreastCancer, 
  target = "Class",
  positive = "malignant")

## 2.1 Count groups
# Several observations have the same ID
# samples taken from the same patient at different times
grp_by_id <- aggregate(
  . ~ Id,
  data = BreastCancer,
  FUN = length
)[, c("Id", "Class")]

grp_by_id[grp_by_id$Class > 1,]

# Solution for 2.1
sum(table(BreastCancer$Id) > 1) # table uses the cross-classifying factors to build a contingency table of the counts at each combination of factor levels.

# We want to make sure that each ID is used either for test or training
task_bc$col_roles$group <- "Id"

cv <- rsmp("cv", folds = 5)
cv$instantiate(task_bc)
cv$instance

# Sanity Check
sum(table(cv$instance$row_id) > 1)

# Solution for sanity check (2.4)
dt <- aggregate(
  fold ~ Id,
  data = merge(
    task_bc$data(),
    cv$instance,
    by.x = "Id",
    by.y = "row_id"
  ),
  FUN = function(x) length(unique(x))
)

dt[dt$fold > 1]

# Also an alternative check might be
dt = merge(task_bc$data(), cv$instance, by.x = "Id", by.y = "row_id")
dt = dt[, .(unique_folds = length(unique(fold))), by = Id]
dt[unique_folds > 1, ]

# 3. Custom perfromance measures
# All available measures with mlr3
as.data.table(mlr_measures)

# Adding custom measure
# A regression measure that scores a prediction as 1 if the diff between the true and predicted values
# is less than one standard deviation of the truth, 0 otherwise
# f(y, y_hat) = 1/n sum from i = 1 to n of I(abs(y_i - y_hat_i) < sigma_y)

MeasureRegrThresholdAcc = R6::R6Class("MeasureRegrThresholdAcc",
                                      inherit = mlr3::MeasureRegr, # regression measure
                                      public = list(
                                        initialize = function() { # initialize class
                                          super$initialize(
                                            id = "thresh_acc", # unique ID
                                            packages = character(), # no package dependencies
                                            properties = character(), # no special properties
                                            predict_type = "response", # measures response prediction
                                            range = c(0, 1), # results in values between (0, 1)
                                            minimize = FALSE # larger values are better
                                          )
                                        }
                                      ),
                                      
                                      private = list(
                                        # define score as private method
                                        .score = function(prediction, ...) {
                                          # define loss
                                          threshold_acc = function(truth, response) {
                                            mean(ifelse(abs(truth - response) < sd(truth), 1, 0))
                                          }
                                          # call loss function
                                          threshold_acc(prediction$truth, prediction$response)
                                        }
                                      )
)


