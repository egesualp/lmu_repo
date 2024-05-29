library(quantreg)
library(effects)
library(gamlss)
library("e1071")
data("engel")

summary(engel)

ggplot(engel) +
  geom_point(aes(y=foodexp, x=income))+
  ggtitle("Overview of the data")+
  ylab("Food expenditure per year (BEF)")+
  xlab("Household income per year (BEF")

# Fitting a GLM with poly
model_glm <- glm(
  formula = foodexp ~ poly(income, degree = 2),
  data = engel
)

summary(model_glm)

par(mfrow = c(2,1))
plot(predictorEffects(model_glm), partial.residuals=TRUE, lines=list(multiline=TRUE))
plot(formula = model_glm$residuals ~ engel$income, xlab = "Log Income [in BEF]",
     ylab = "Residuen", cex.axis = 0.8)
par(mar = c(5, 4, 0.5, 2))
plot(formula = model_glm$residuals ~ model_glm$fitted.values,
     xlab = "Fitteted Values", ylab = "Residuals", cex.axis = 0.7)


# Fitting gamlss model to capture the heteroscedasticity
model_gamlss <- gamlss(
  formula = foodexp ~ poly(income, degree = 2),
  sigma.formula = ~ poly(income, degree = 2),
  family = NO(),
  data = engel
)

# Log variable to test
engel[,'foodexp_log'] <- log(engel$foodexp)
engel[,'income_log'] <- log(engel$income)


model_gamlss_log <- gamlss(
  formula = foodexp_log ~ poly(income_log, degree = 2),
  sigma.formula = ~ poly(income_log, degree = 2),
  family = NO(),
  data = engel
)


summary(model_gamlss)

# Visualization for the model results
par(mfrow = c(2,2))
term.plot(object = model_gamlss, what = "mu", rug = TRUE,
          main = "Smooth effect with respect to the expected value",
          xlab = "Log Income [BEF]", cex.main = 1)
term.plot(object = model_gamlss, what = "sigma", rug = TRUE,
          main = "Smooth effect with respect to the standard deviation",
          xlab = "Log income [BEF]", cex.main = 0.9)

# Visualization for the model results (log vars)
par(mfrow = c(2,2), mar = c(4, 4, 2, 1))
term.plot(object = model_gamlss_log, what = "mu", rug = TRUE,
          main = "Smooth effect with respect to the expected value",
          xlab = "Log Income [BEF]", cex.main = 1)
term.plot(object = model_gamlss_log, what = "sigma", rug = TRUE,
          main = "Smooth effect with respect to the standard deviation",
          xlab = "Log income [BEF]", cex.main = 0.9)

# Question 5 Model Selection
# Replace "your_file.txt" with the actual path to your file
file_path <- "C:\\Users\\esual\\OneDrive\\Documents\\StaDS\\lmu_repo\\data\\model_selection_week_four.txt"

# Read the data with read.table
soep <- read.table(file_path, header = TRUE, sep = " ")

# View the structure of your data
str(soep)

# View the first few rows of your data
head(soep)

# EDA
par(mfrow=c(1,3))
hist(soep$ginc)
plot(density(soep$ginc))
boxplot(soep$ginc)

# Natural Link (inverse)
gammaNat <- glm(
  formula = ginc ~ height + age + length + mar + sex + german + alevel,
  family = Gamma(),
  data = soep
)

summary(gammaNat)

gammaLog <- glm(
  formula = ginc ~ height + age + length + mar + sex + german + alevel,
  family = Gamma(link = 'log'),
  data = soep
)

summary(gammaLog)

# How many model is possible?
covar.combinations <- bincombinations(7)
colnames(covar.combinations) <- colnames(soep[, c(-1, -9)])
tail(covar.combinations)# last row -> complete model

# alternatively
choose(7,1)*2 + choose(7,2)*2 + choose(7,3)*2 + choose(7,0)*2 # 128


# Stepwise variable selection based on the AIC criterion
# starting with the complete model:
fullstepml.fit <- step(gammaLog, direction = "both")

# Null-model:
partialfit <- glm(ginc ~ 1, family=Gamma(link=log), data = soep)
summary(partialfit)


# Stepwise variable selection starting with the null-model:
# scope shows up do which model variables are added maximally.
partialstepml.fit <- step(partialfit,
                          scope = ginc ~ height + age + length + mar + sex + german + alevel,
                          direction = "both")

AIC(fullstepml.fit); AIC(partialstepml.fit)

all(names(fullstepml.fit$coefficients) %in% names(partialstepml.fit$coefficients))

