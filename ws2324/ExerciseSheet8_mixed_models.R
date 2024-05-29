# libraries
library(ggplot2)

# Question 18: Linear Mixed Model with Blood Pressure Data

# Load the data from the .rds file
blood_pressure <- readRDS("C:\\Users\\esual\\OneDrive\\Documents\\StaDS\\lmu_repo\\data\\sheet8_blood_pressure.rds")

# EDA
head(blood_pressure)
summary(blood_pressure)

ggplot(
  data = blood_pressure,
  mapping = aes(x = dose, y = SBP, col = as.factor(person))
) + geom_line() + geom_point()

unique(blood_pressure$person)
blood_pressure[blood_pressure$SBP <= min(blood_pressure$SBP),]

# Naive model
model_linear <- lm(
  formula = SBP ~ sex + dose + sex * dose,
  data = blood_pressure
)

summary(model_linear)

# Dose is negatively associated with SBP (as was to be expected).
# We are ignoring intra-subject correlation due to repeated measurements.

library(lme4)
model_ri <- lme4::lmer(
  formula = SBP ~ sex + dose + sex * dose + (1|person),
  data = blood_pressure,
  REML = FALSE
)

predictions <- predict(model_ri, blood_pressure)

var(blood_pressure$SBP - predictions)
var(predictions)
var(blood_pressure$SBP)
summary(model_ri)$varcor


# Assuming you have a linear mixed-effects model named 'model_ri'
library(lme4)

# Fit the linear mixed-effects model
model_ri <- lmer(
  formula = SBP ~ sex + dose + sex * dose + (1|person),
  data = blood_pressure,
  REML = FALSE
)

residuals_linear <- residuals(model_linear)
residuals_ri <- residuals(model_ri)
# Extract the variance components
var_components <- VarCorr(model_ri)

# Extract the variance of random intercepts
var_random_intercepts <- unname(var_components$person[1])

# Conduct a hypothesis test at α = 0.05 to test whether there is a (statistically) significant interaction effect 
# between dose and sex. What needs to be considered when conducting likelihood ratio
# tests with nested fixed effects?

comp <- lme4::lmer(
  formula = SBP ~ sex + dose + (1|person),
  data = blood_pressure, REML = FALSE
)

anova(model_ri, comp)

# Does random intercept improve the model?

# Approximate LLR test (based on ML estimation)
anova(model_ri, model_linear)

library(RLRsim)
RLRsim::exactRLRT(m = model_ri)

# Adding random slope additionally

# no convergence
model_rirs <- lme4::lmer(
  formula = SBP ~ sex + dose+ sex * dose + (1 + dose | person),
  data = blood_pressure, REML = FALSE
)
# change optimization algorithm, yet getting singular fit warning
model_rirs_wo_warn <- lmer(formula = SBP ~ sex + dose + sex * dose + (1 + dose|person),
                    data = blood_pressure, REML = FALSE,
                    control = lmerControl(optimizer ="Nelder_Mead"))

summary(model_rirs)

summary(model_rirs)$coefficients
coefficients(model_rirs)

# Approximate likelihood ratio test (based on ML estimation)
anova(model_ri, model_rirs)

# Exact thest (based on REML estimation)
# model with random slope only (no random intercept)
# Exact test (based on REML estimation):
# model with random slope only (no random intercept):
model_rs <- lmer(formula = SBP ~ sex + dose + sex * dose + (-1 + dose|person),
                 data = blood_pressure, REML = FALSE)
exactRLRT(m = model_rirs, mA = model_rs, m0 = model_ri)

# required components:
# m: model with random slope and random intercept (alternative hypothesis)
# mA: model with random slope only
# m0: model with random intercept only (null hypothesis)

exactRLRT(m = model_rirs, m0 = model_ri)

# Using the corrected AIC, compare the fit of the models
library(cAIC4)
cAIC4::cAIC(model_linear)
cAIC4::cAIC(model_ri)
cAIC4::cAIC(model_rirs)

# Question 19: Linear Mixed Model with Reading Data
eye <- read.csv("C:\\Users\\esual\\OneDrive\\Documents\\StaDS\\lmu_repo\\data\\sheet8_reading_time.csv", sep = ";", dec = ",")
require(tidyverse)  

eye <- eye %>%
  filter(AOI %in% c("A", "F", "P"),
         Kondition %in% c("1a2a", "1b2b", "1b3a", "1c3a")) %>%
  mutate(AOI.condition = droplevels(as.factor(AOI.condition)))

table(eye$AOI)
table(eye$Kondition)
table(eye$AOI.condition)

eye$AOI.condition <- relevel(eye$AOI.condition, ref = "F_1b2b")
eye_lm <- lm(
  formula = FRT.WD ~ AOI.condition,
  data = eye
)
summary(eye_lm)
unique(eye$AOI.condition)

eye_lmm <- lmer(
  FRT.WD ~ AOI.condition + (1 | Participant) + (1 | Topic),
  data = eye
)

summary(eye_lmm)

# Test whether the random intercept is necessary (whether its variance is larger than 0)
#  First, test each random intercept individually, using an exact test.
eye_topic <- update(eye_lmm, . ~ . - (1 | Participant))
exactRLRT(m = eye_topic)
summary(eye_topic)

eye_participant <- update(eye_lmm, . ~ . - (1 | Topic))
exactRLRT(m = eye_participant)

# Using an approximate LR test, compare your model from c (Participant and Topic as random intercepts)
# with the linear form (naive linear model)

eye_lmm_ml <- lmer(FRT.WD ~ AOI.condition + (1 | Participant) + (1 | Topic),
                   data = eye, REML = FALSE)
anova(eye_lmm_ml, eye_lm)
