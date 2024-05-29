# Simulation Study
library(MASS)
library(dplyr)

# We generate 10 different covariates x ~ N(0,1) 
# and simulate Y ~ N(10 + 0.25xi1 + 0.25xi2 + 0.1xi3 + 0.1xi4, 1)
# Therefore, the response variable depends only on the fist 4 covariates
# Perform the following with differentsample sizes n = 50, 100, 150, 200, 500, 1000, 5000 
# and with S = 100 repetitions

# a) Generate the above mentioned data structure
simulate_data <- function(n, beta_true, sigma = 1) {
  # Generate 10 independent covariates from N(0, 1)
  X <- matrix(rnorm(n * 10), nrow = n, ncol = 10)
  design <- cbind(Intercept = 1, X)
  # Calculate Y based on the first four covariates and intercept
  Y <- (design %*% beta_true) + rnorm(n, sigma)
  return(data.frame(Y = Y, X))
}
set.seed(123) # For reproducibility
sample_sizes <- c(50, 100, 150, 200, 500, 1000, 5000)
beta_true <- c(10, 0.25, 0.25, 0.1, 0.1, 0, 0, 0, 0, 0, 0)
data_list = lapply(sample_sizes, function(n) {
  lapply(1:100, function(rep) {
    simulate_data(n, beta_true)
  })
})

)

# Appying step AIC to select a final model
# Evaluating the coverage probabilities (i.e. the percentage of cases where the confidence interval contains the true coef)

# A function for evaluation of one data set
eval_simulation <- function(sim_dta, beta_true, coeffs, in_confint){
  # estimate linear model & shrink by step aic
  lm_full <- lm(Y~., data = sim_dta)
  stepAIC_run <- stepAIC(lm_full, trace = FALSE)
  # extract coef and conf_int
  coef_run <- coef(stepAIC_run)
  conf_int <- confint(stepAIC_run)
  conf_int <- cbind(conf_int, beta_true = beta_true[names(coef_run)])
  # Get full coefficient list and check if coefs are in correct conf int
  coeffs[names(coef_run)] <- coef_run
  in_confint[row.names(conf_int)] <- apply(conf_int, 1, function(int) {
    between(int[3], int[1], int[2])
  })
  # return a one-row df with all coeffs and wheater coeff is in the intervall
  list(coeffs = cbind(coeffs, sample_size = nrow(sim_dta)),
       conf_int = cbind(in_confint, sample_size = nrow(sim_dta)))
}


# The arguments (this acts like a template for the return values)
# The arguments (this acts like a template for the return values)
coeffs <- data.frame(t(rep(0, times = 11)))
in_confint <- data.frame(t(ifelse(beta_true == 0, TRUE, FALSE)))
coef_names <- c("(Intercept)", names(data_list[[1]][[1]][,-1]))
colnames(coeffs) = coef_names
colnames(in_confint) = coef_names
names(beta_true) = coef_names

# Evaluate for all data sets
start = Sys.time()
results = lapply(data_list, function(sample_n) {
  lapply(sample_n, eval_simulation, beta_true, coeffs, in_confint)
})
Sys.time() - start
## Time difference of 17.59957 secs

# Check confidence intervals and calculate coverage probabilities per var
in_confint = rbindlist(lapply(results, function(size) {
  rbindlist(lapply(size, function(rep) rep$conf_int))
}))
percent_inconfint <- in_confint[, lapply(.SD, sum), by = sample_size] # not a good function, use the below one
percent_inconfint <- in_confint %>% group_by(sample_size) %>% summarise(across(where(is.logical), sum))
percent_inconfint

coeffs = rbindlist(lapply(results, function(size) {
  rbindlist(lapply(size, function(rep) rep$coeffs))
}))
coeffs[, sample_size := as.factor(sample_size)]
# long format and filter non-zero coefficients
coeffs = melt(coeffs, id.vars = "sample_size", variable.name = "coefficient",
              value.name = "coeff_size")
coeffs = coeffs[coefficient %in% coef_names[1:10], ]

ggplot(coeffs, aes(x = sample_size, y = coeff_size, colour = sample_size)) +
  geom_boxplot() +
  geom_jitter(alpha = 0.3) +
  facet_wrap(~coefficient, scales = "free_y")

# Evaluate how the results change, when you increase or decrease the absolute size of the coefficients or the number of coefficients which are different from zero. As an example, consider
# β = (10, 100, 100, 0.01, 0.01, −100, −100, 0, 0, 0, 0)
library(knitr)
beta_true <- c(10, 100, 100, 0.01, 0.01, -100, -100, 0, 0, 0, 0)
data_list = lapply(sample_sizes, function(n) {
  lapply(1:100, function(rep) {
    simulate_data(n, beta_true)
  })
})
coeffs <- data.frame(t(rep(0, times = 11)))
in_confint <- data.frame(t(ifelse(beta_true == 0, TRUE, FALSE)))
coef_names <- c("(Intercept)", names(data_list[[1]][[1]][,-1]))
colnames(coeffs) = coef_names
colnames(in_confint) = coef_names
names(beta_true) = coef_names

# Evaluate for all data sets
start = Sys.time()
results = lapply(data_list, function(sample_n) {
  lapply(sample_n, eval_simulation, beta_true, coeffs, in_confint)
})
Sys.time() - start
## Time difference of 15.30623 secs
# Check confidence intervals and calculate coverage probabilities per var
in_confint = rbindlist(lapply(results, function(size) {
  rbindlist(lapply(size, function(rep) rep$conf_int))
}))
percent_inconfint <- in_confint[, lapply(.SD, sum), by = sample_size]
kable(percent_inconfint)

coeffs = rbindlist(lapply(results, function(size) {
  rbindlist(lapply(size, function(rep) rep$coeffs))
}))
coeffs[, sample_size := as.factor(sample_size)]
# long format and filter non-zero coefficients
coeffs = melt(coeffs, id.vars = "sample_size", variable.name = "coefficient",
              value.name = "coeff_size")
coeffs = coeffs[coefficient %in% coef_names[1:11], ]
ggplot(coeffs, aes(x = sample_size, y = coeff_size, colour = sample_size)) +
  geom_boxplot() +
  geom_jitter(alpha = 0.3) +
  facet_wrap(~coefficient, scales = "free_y")
