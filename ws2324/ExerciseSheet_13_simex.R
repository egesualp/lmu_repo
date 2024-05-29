vote <- read.csv("./StaDS/lmu_repo/data/Elect.csv")[,-1]
vote$Union <- as.factor(vote$Union)

head(vote)
str(vote)


# Fit the model specified above in R and interpret the effect of the covariates Age c and Union on the
# odds of voting for the Social Democrats (SPD) as compared to the Christian Democratic Union
# (CDU).
naive <- glm(formula = Pref ~ .,
             family = binomial(),
             data = vote,
             x=T, y=T)
summary(naive)

# From the literature, assume that it is known that 
# the measurement error variance is σ2u = 16.
#How does that compare to the overall variance of the mismeasured/error-prone age?

var(vote$Age_c)
## [1] 282.8497
16/(var(vote$Age_c))
## [1] 0.05656715

# Use the simex() function from the simex package to estimate the SIMEX for the model in (b),
# but now taking into account the measurement error

library(simex)
set.seed(123)
sim1 <- simex(model = naive,
              SIMEXvariable = "Age_c",
              measurement.error = 4) # standard deviation
summary(sim1)

plot(sim1, mfrow=c(3,3))

# Now repeat the SIMEX, but instead use σ2u = 16^2
# What do you observe
set.seed(123)

sim2 <- simex(model = naive,
              SIMEXvariable = "Age_c",
              measurement.error = 16) # standard deviation
summary(sim2)
plot(sim2, mfrow=c(3,3))

# Non parametrix extrapolation instead of quadratic
set.seed(123)
sim3 <- refit(sim2, "nonl") # ERROR
summary(sim3) 

# Now assume that age does not contain measurement error. Instead, suppose that it is known that
# around 9% of union members wrongly state that they do not belong to any union, while 5% of
# non-members wrongly state that they do. What kind of measurement error do we now have at
# hand for the covariate Union? What assumption must be made for us to be able to subsequently
# use MCSIMEX?

set.seed(123)
mc.union <- matrix(c(0.95,0.05,0.09,0.91),nrow=2)
dimnames(mc.union) <- list(levels(vote$Union), levels(vote$Union))
mc.sim1 <- mcsimex(model = naive,
                   SIMEXvariable = "Union",
                   mc.matrix = mc.union)
summary(mc.sim1)
plot(mc.sim1, mfrow=c(3,3))
