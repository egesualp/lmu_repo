library(AER)
library(ggplot2)
data("SmokeBan")
smoke <- SmokeBan

# Question A Proportion of smokers
prop.table(table(smoke$smoker))

summary(smoke)

model1 <- glm(formula = smoker ~ ban + age + education + gender,
              data = smoke, family = binomial())

summary(model1)

model2 <- glm(formula = smoker ~ ban + age + education + gender,
              data = smoke, family = binomial(link = "probit"))

summary(model2)

model3 <- glm(formula = smoker ~ ban + age + education + gender + afam + hispanic,
              data = smoke, family = binomial())

summary(model3)

# Likelihood Ratio Test for model3 and model2
probs_m1 <- predict(model1, type = "response")
probs_m3 <- predict(model3, type = "response")
response <- as.numeric(smoke$smoker == "yes")

Log_like_m1 <- sum(response * log(probs_m1) + (1 - response) * log(1 - probs_m1))
Log_like_m3 <- sum(response * log(probs_m3) + (1 - response) * log(1 - probs_m3))

# Display results
cat("Log-Likelihood for model 2:", Log_like_m1, "\n")
cat("Log-Likelihood for model 3:", Log_like_m3, "\n")

LTR <- 2*(Log_like_m3 - Log_like_m1)

# Calculate p-value
p_value <- 1 - pchisq(LTR, 2)

# Display results
cat("Likelihood Ratio Test (LRT) Statistic:", LTR, "\n")
cat("Degrees of Freedom:", 2, "\n")
cat("p-value:", p_value, "\n")

# Alternative: Using predefined functions
lrt_result <- lrtest(model1, model3)
print(lrt_result)

anova(model1, model3, test = "LRT")
