library(ISLR)
data("Hitters")

names(Hitters)

Hitters = na.omit(Hitters)
str(Hitters)

summary(Hitters)

# Subsetting data into training and test
index = sample(
  c(TRUE, FALSE), 
  nrow(Hitters), 
  replace=TRUE, 
  prob = c(0.8, 0.2)
  )

train=Hitters[index,]
test=Hitters[!index,]

# Use a backward stepwise variable selection, using the AIC
library(MASS)
LinMod <- lm(Salary ~ ., data=train)
LinModSel <- stepAIC(LinMod, score = lm(Salary~1, data=train, direction = "both", trace=0))

anova(LinModSel, lm(Salary ~ 1, data=train))

# Model diagnostic
par(mfrow=c(2,2))
plot(LinModSel)

# We can observe, even after the variable selection, some rather strong correlation between some
# covariates. This could introduce bias.
library(pls)
PcrMod<-pcr(Salary~., data=train, scale=TRUE, validation="CV")
summary(PcrMod)

# find the number of components that minimize the MSE
par(mfrow=c(1,1))
validationplot(PcrMod, val.type = "MSEP")

# Compare the test MSE of the best sample linear model and the pcr model you have
# fitted
library(mlr3measures)

pcr.pred=predict(PcrMod, test, ncomp=7)
linmod.pred=predict(LinModSel, test)
mse(pcr.pred, test$Salary)
mse(linmod.pred, test$Salary)
