# Exercise Sheet 7
# Propensity Score

library(MatchIt)
data(lalonde)

# Simple regression
reg1 <- lm(
  formula = re78 ~ treat,
  data = lalonde
)
summary(reg1)

# Running welch's t-test to check whether the effect is significant
tt1 = t.test(
  lalonde$re78[lalonde$treat == 1],
  lalonde$re78[lalonde$treat == 0],
  paired = FALSE
)
tt1

# This part is arbitrary

# Extract the relevant data for the treatment and control groups
treatment_data <- lalonde$re78[lalonde$treat == 1]
control_data <- lalonde$re78[lalonde$treat == 0]

# Plot the density curves
plot(density(log(treatment_data)), main="Distribution Comparison", col="blue", lwd=2, ylim=c(0, 0.5), xlab="re78", ylab="Density")
lines(density(log(control_data)), col="red", lwd=2)

# Add legend
legend("topright", legend=c("Treatment", "Control"), col=c("blue", "red"), lwd=2)

# Plot the cumulative distribution functions (CDFs)
plot(ecdf(treatment_data), main="CDF Comparison", col="blue", lwd=2, xlim=c(min(lalonde$re78), max(lalonde$re78)), xlab="re78", ylab="Cumulative Probability")
lines(ecdf(control_data), col="red", lwd=2)

# Add legend
legend("bottomright", legend=c("Treatment", "Control"), col=c("blue", "red"), lwd=2)

## Arbitrary part ends

# comparison of some covariate descriptive statistics between two groups
library(dplyr)
lalonde %>% 
  group_by(treat) %>%   
  summarize(across(c(1,2,4,5,6,7,8), mean, na.rm = TRUE))

# We observed negative treatment effect given the fact that
# there is selection bias 

m <- matchit(
  treat ~ age + educ + race + married + nodegree + re74 + re75,
  data = lalonde,
  method = "nearest",
  distance = "logit"
)

summary(m)

# We now create a dataframe from previously created matchit object.
lalonde_matched <- match.data(m,, distance = "pscore")
hist(lalonde_matched$pscore)
summary(lalonde_matched$pscore)

# New t-test
tt2 <- t.test(
  lalonde_matched$re78[lalonde_matched$treat == 1],
  lalonde_matched$re78[lalonde_matched$treat == 0],
  paired = FALSE
)
tt2
