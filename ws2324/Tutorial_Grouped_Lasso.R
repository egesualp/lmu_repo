library(AER)
data("DoctorVisits")
DoctorVisits <- DoctorVisits[, c("visits", "gender", "age", "illness")]
DoctorVisits$age <- DoctorVisits$age * 100
str(DoctorVisits)

model1 <- glm(formula = visits ~ age * gender + illness, family = "poisson",
              data = DoctorVisits)

model2 <- glm(formula = visits ~ age * gender + illness, family = "quasipoisson",
              data = DoctorVisits)

model3 <- glm.nb(formula = visits ~ age * gender + illness, data = DoctorVisits,
                 link = log, init.theta = 0.5399626194)

# Construction of binary variable
DoctorVisits$visits_bin <- ifelse(DoctorVisits$visits == 0, 0, 1)

model4 <- glm(formula = visits_bin ~ age * gender + illness, family = "binomial",
              data = DoctorVisits)

# StatMod Tutorial
# Solution exercise Variable Selection

# a
load("./StaDS/lmu_repo/data/soep_2017.RData")

str(data_2017)
summary(data_2017)

# to avoid issues
table(data_2017$Employment)
data_2017 <- data_2017[!data_2017$Employment %in% 5:8, ]
data_2017$Employment <- droplevels(data_2017$Employment)

set.seed(11111)
n <- nrow(data_2017)
s <- sample(x = 1:n, size = n * 0.75)
train <- data_2017[s, ]
test <- data_2017[-s, ]

# b
# Since the variable Satisfaction consists of natural numbers (including 0), 
# Poisson regression should be applied.

glm_pois <- glm(formula = Satisfaction ~ ., family = "poisson", data = train)
summary(glm_pois)

# c
library(glmnet) # for LASSO
library(smurf)  # for grouped LASSO

# AIC and BIC
glm_AIC <- step(object = glm_pois, trace = 0)
glm_BIC <- step(object = glm_pois, trace = 0, k = log(nrow(train)))

# LASSO
glm_lasso <- glmnet(x = model.matrix(glm_pois),
                    y = train$Satisfaction,
                    family = "poisson",
                    alpha = 1)
# NOTE: Standard LASSO cannot deal with categorical variables. Therefore it is
#       unreasonable and wrong to use it in such a setting. Here it is used to
#       illustrate the differences to grouped LASSO.

# grouped LASSO
form_lasso_gr <- Satisfaction ~ p(Gender, pen = "lasso") +
  p(Age, pen = "lasso") +
  p(Employment, pen = "grouplasso") +
  p(Disability, pen = "grouplasso") +
  p(HospitalNights, pen = "lasso") +
  p(Relationship, pen = "grouplasso")


glm_lasso_gr <- glmsmurf(formula = form_lasso_gr,
                         family = "poisson",
                         data = train,
                         pen.weights = "glm.stand")

# d
plot(glm_lasso, "lambda", label = TRUE)

library(plotmo)
plot_glmnet(x = glm_lasso, label = TRUE, xvar = "lambda")

# e
# LASSO
min_lambda <- cv.glmnet(x = model.matrix(glm_pois),
                        y = train$Satisfaction,
                        family = "poisson",
                        alpha = 1)$lambda.min

glm_lasso_min <- glmnet(x = model.matrix(glm_pois),
                        y = train$Satisfaction,
                        family = "poisson",
                        alpha = 1, 
                        lambda = min_lambda)
glm_lasso_min$beta

# grouped LASSO
min_lambda_gr <- glmsmurf(formula = form_lasso_gr,
                          family = "poisson",
                          data = train,
                          pen.weights = "glm.stand",
                          lambda = "cv.mse")$lambda

glm_lasso_gr_min <- glmsmurf(formula = form_lasso_gr,
                             family = "poisson",
                             data = train,
                             pen.weights = "glm.stand",
                             lambda = min_lambda_gr)
summary(glm_lasso_gr_min)
# The summary also contains the coefficients of a
# re-estimated glm model (see help of glmsmurf-class).

# f
glm_AIC$coefficients
glm_BIC$coefficients
glm_lasso_min$beta
glm_lasso_gr_min$coefficients

# The models resulting from the AIC and BIC have the same complexity as do both 
# LASSO models. The models resulting from the information criteria have less 
# coefficients only including Disability, Hospital Nights and Relationship.
# The LASSO model did not penalize any variable coefficient to 0.
# The Grouped LASSO dropped the variables gender, age and employment, i.e.
# shrunk the coefficients to 0.

# g
# AIC and BIC
y_IC <- predict.glm(object = glm_AIC, newdata = test, type = "response") 
mse1 <- mean((y_IC - test$Satisfaction) ^ 2)

# model matrix for test data
mat_test <- model.matrix(lm(formula = Satisfaction ~ ., data = test))

# LASSO
y_lasso <- exp(predict.glmnet(object = glm_lasso,
                              newx = mat_test,
                              s = min_lambda))
mse2 <- mean((y_lasso - test$Satisfaction) ^ 2)

# grouped LASSO
# (exclude the columns that were not selected)
mat_test <- model.matrix(lm(Satisfaction ~ Disability + HospitalNights + Relationship, data = test))
coff_gr <- glm_lasso_gr_min$coefficients.reest[glm_lasso_gr_min$coefficients.reest != 0]

mse3 <- mean((exp(mat_test %*% coff_gr) - test$Satisfaction) ^ 2)

mse1
mse2
mse3

# All models are very close together.