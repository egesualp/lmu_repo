# Generate and plot training data:
set.seed(2)
n <- 250
ntest <- 500
f <- function(x) {sin(2*(4*x - 2)) + 2*exp(-(16)^2*(x - 0.5)^2)}
x <- seq(from=0, to=1, length.out = n)
eps <- rnorm(n, 0, 0.3)
y <- f(x) + eps
plot(x,y, ylim = c(-2,3))
curve(f(x), 0, 1, add=T, lwd = 2)

# Generate and plot test data:
set.seed(2)
xtest <- seq(from=0, to=1, length.out = ntest)
epstest <- rnorm(ntest, 0, 0.3)
ytest <- f(xtest) + epstest
plot(xtest, ytest, ylim = c(-2,3))
     curve(f(x), 0, 1, add=T, lwd = 2)

# Fit a linear, quadratic and cubic model as well as higher dimensional polynomials: d ∈
# {1, 2, 3, 10, 50, 250}
     
polynomfit <- function(d){
       # create training dataframe
       traindata <- data.frame(y = y)
       for(i in 1:d){
         traindata <- cbind(traindata, x^i)
         names(traindata)[length(traindata)] <- paste("x^",i)
       }
       # create test dataframe
       testdata <- data.frame(y = ytest)
       for(i in 1:d){
         testdata <- cbind(testdata, xtest^i)
         names(testdata)[length(testdata)] <- paste("x^",i)
       }
       # fit model, create predictions and RSS
       fit <- lm(y ~ . , data = traindata)
       pred <- predict(fit, newdata = testdata)
       out <- list(fit = fit,
                   prediction = pred,
                   rss = sum(resid(fit)^2),
                   rsstest = sum((ytest - pred)^2)
       )
       return(out)
}

# linear:
fit1 <- polynomfit(d = 1)
# quadratic:
fit2 <- polynomfit(d = 2)
# cubic:
fit3 <- polynomfit(d = 3)
# and so on:
fit10 <- polynomfit(d = 10)
fit50 <- polynomfit(d = 50)
fit250 <- polynomfit(d = 250)

# Checking even higher degrees
fit300 <- polynomfit(d = 300)
fit400 <- polynomfit(d = 400)
fit500 <- polynomfit(d = 500)


# plot of the estimated curves:
par(mfrow=c(3,3))
# linear
plot(x, y, main = "d = 1")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit1$fit), lwd = 2, col = "red")
# quadratic
plot(x, y, main = "d = 2")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit2$fit), lwd = 2, col = "red")
# cubic
plot(x, y, main = "d = 3")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit3$fit), lwd = 2, col = "red")
# higher order
plot(x, y, main = "d = 10")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit10$fit), lwd = 2, col = "red")
plot(x, y, main = "d = 50")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit50$fit), lwd = 2, col = "red")
plot(x, y, main = "d = 250")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit250$fit), lwd = 2, col = "red")

plot(x, y, main = "d = 300")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit300$fit), lwd = 2, col = "red")
plot(x, y, main = "d = 400")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit400$fit), lwd = 2, col = "red")
plot(x, y, main = "d = 500")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit500$fit), lwd = 2, col = "red")


# plot of the estimated curves for test data:
par(mfrow=c(2,3))
x_test <- xtest
y_test <- ytest
# linear
plot(x_test, y_test, main = "d = 1")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x_test, fit1$pred, lwd = 2, col = "red")
# quadratic
plot(x_test, y_test, main = "d = 2")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x_test, fit2$pred, lwd = 2, col = "red")
# cubic
plot(x_test, y_test, main = "d = 3")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x_test, fit3$pred, lwd = 2, col = "red")
# higher order
plot(x_test, y_test, main = "d = 10")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x_test, fit10$pred, lwd = 2, col = "red")
plot(x_test, y_test, main = "d = 50")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x_test, fit50$pred, lwd = 2, col = "red")
plot(x_test, y_test, main = "d = 250")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x_test, fit250$pred, lwd = 2, col = "red")

# Conclusion: From a certain polynomial degree onwards we achieve a good approximation by
# the polynomial but the overfitting increases and the estimation becomes unstable - especially
# at the margins

# Impact on test data:
# Compute and order the residual sum of squares:
RSSs <- matrix(nrow=9, ncol=2)
RSSs[1,] <- c(fit1$rss, fit1$rsstest)
RSSs[2,] <- c(fit2$rss, fit2$rsstest)
RSSs[3,] <- c(fit3$rss, fit3$rsstest)
RSSs[4,] <- c(fit10$rss, fit10$rsstest)
RSSs[5,] <- c(fit50$rss, fit50$rsstest)
RSSs[6,] <- c(fit250$rss, fit250$rsstest)
RSSs[7,] <- c(fit300$rss, fit300$rsstest)
RSSs[8,] <- c(fit400$rss, fit400$rsstest)
RSSs[9,] <- c(fit500$rss, fit500$rsstest)
colnames(RSSs) <- c("RSS on training data", " Predictive RSS on test data")
rownames(RSSs) <- c("d = 1", "d = 2", "d = 3", "d = 10","d = 50", "d = 250", "d = 300", "d = 400", "d = 500")
# the following options prevents big numbers to be displayed in the form of "1.12345e10"
options(scipen = 15)

# Showing the results
RSSs


# We see: the more complex the used model, the better the fit on the training data.
# Yet an overly complex model leads to overfitting on the training data which deteriorates the quality of prediction on test data.
# Ordinary polynomials are defined globally such that an x-value to the left of the domain influences the estimation of the right part of the domain. 
# (The estimation of a smooth curve is more inaccurate at the margins compared to the center of 
# the data for all smoothing methods, but for this reason ordinary polynomials are especially unstable there)

# b: Divide the domain [0, 1] of x into ten equidistant intervals and fit a cubic 
#polynomial based on the observations corresponding to each interval. Compare these piecewise estimated polynomials
# with the results in a)

ind <- matrix(nrow = 10, ncol = 25)
knots = seq(0, 1, by = 0.1)

for(i in 1:(length(knots)-1) ){
  ind[i,] <- which(x >= knots[i] & x <= knots[i+1])
}
par(mfrow=c(1,1))
plot(x, y)
curve(f(x), 0, 1, add=T, lwd = 2)
for(i in 1:10){
  lines(x[ind[i,]], lm(y[ind[i,]] ~ poly(x[ind[i,]], 3, raw=T))$fitted.values
        , lwd = 2, col = "red")
}

# The estimation appears to have less artefacts / to be less rough. But the polynomial pieces
# have different values at the intervals borders such that the estimated function becomes discontinuous.

# Visualize the influence of the spline degree and the number of knots.
# Different spline degrees for 13 inner knots.
# B-spline base constructed using the bs()-function from the package splines.
library(splines)
# help(bs)

par(mfrow=c(2,2))

# degree 1
fit <- lm(y ~ bs(x, degree = 1, df = 13 + 1))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "Spline degree 1")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# degree 2
fit <- lm(y ~ bs(x, degree = 2, df = 13 + 2))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "Spline degree 2")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# degree 3
fit <- lm(y ~ bs(x, degree = 3, df = 13 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "Spline degree 3")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# degree 10
fit <- lm(y ~ bs(x, degree = 10, df = 13 + 10))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "Spline degree 10")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)


# Conclusion: for a spline degree of 1, the estimation is not smooth, also for degree = 2 the
# smoother is lacking flexibility. From a degree of 3 onwards, the estimation is sufficiently
# smooth but with an increasing spline degree (for constant number of knots) we obtain
# overfitting. The AIC values support the graphical observations. In practical applications,
# cubic splines are mostly used

# Influence of the number of knots for fixed spline degree of 3:
  
par(mfrow=c(3,3))
# 1 knot
  fit <- lm(y ~ bs(x, degree = 3, df = 1 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "1 knot")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 2 knots
fit <- lm(y ~ bs(x, degree = 3, df = 2 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "2 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 3 knots
fit <- lm(y ~ bs(x, degree = 3, df = 3 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "3 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 5 knots
fit <- lm(y ~ bs(x, degree = 3, df = 5 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "5 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 7 knots
fit <- lm(y ~ bs(x, degree = 3, df = 7 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "7 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 10 knots
fit <- lm(y ~ bs(x, degree = 3, df = 10 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "10 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 13 knots
fit <- lm(y ~ bs(x, degree = 3, df = 13 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "13 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 20 knots
fit <- lm(y ~ bs(x, degree = 3, df = 20 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "20 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)
# 50 knots
fit <- lm(y ~ bs(x, degree = 3, df = 50 + 3))
top_left_text <- paste("AIC: ", toString(round(AIC(fit),2)))
plot(x, y, main = "50 knots")
curve(f(x), 0, 1, add=T, lwd = 2)
lines(x, predict(fit), lwd = 2, col = "red")
text(0, 0.9*max(y), top_left_text, pos = 4, cex = 0.8, font = 1)

# Estimate f via P-splines
# load package mgcv and several help pages
library(mgcv)
# help(gam)
# help(s)
# help(smooth.terms)
# help(plot.gam)
# fit the model with 13 knots, implying k=15 (knots + degree - 1 = k)
fit <- gam(y ~ s(x, bs = "cr", fx = TRUE, k = 15))

#fx = T, because we do not want to penalize yet.
summary(fit)

# of course, the estimated curves are not contained in summary!
# plot the estimated curve:
par(mfrow=c(1,1))
?plot.gam
# now, a plot equivalent to the one in sub-exercise a)
plot(fit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3))
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)

# We clearly see a distortion, the estimated curve is systematically lower. What went wrong?
# Smooth terms in gam() are always centered around 0 
# but here a model
# without intercept and a function that has its mean at around 0.25 is fitted. If we fit a
# model with intercept, the intercept has to be added to the smoother (via the option shift)
# to obtain the fitted values

plot(fit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(fit)[1])
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)
# Example interpretation for a point x_i = 0.5:
abline(v = 0.5)
abline(h = 2)

#  Now P-splines
# fit with p-splines and the gam()-default value of basis dimension k=10:
pfit <- gam(y ~ s(x, bs = "ps"))
summary(pfit)

summary(pfit)$edf
length(coef(pfit))

# Plot
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1])
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)

# The default smoother is not able to capture the true curve shape. Now the same with
# k = 30
pfit <- gam(y ~ s(x, bs = "ps", k = 30))
summary(pfit)

summary(pfit)$edf
length(coef(pfit))

coef(pfit)
#plot
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1])
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)

# Much better! What happens if we choose a large k (say, 200)? Does the p-spline smoother
# stay stable?
pfit <- gam(y ~ s(x, bs = "ps", k = 200))
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1])
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)

# Despite too many knots (implied by the large k) the p-spline smoother stays stable since
# the penalization removes redundant complexity. Compare the effective degrees of freedom
# of the smoother with the number of coefficients:

summary(pfit)$edf
length(coef(pfit))

# An edf equal to 1 is equivalent to a linear relationship, edf between 1 and 2 is considered a
# weakly non-linear relationship, and edf larger than 2 implies a highly non-linear relationship.

# Let us now visualize the effect of the smoothing parameter lambda for k=30:
#Visualize the effect of the smoothing parameter lambda for k=30:
par(mfrow=c(3,2))
# lambda = 20
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 20)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 20")
# points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)
# lambda = 6
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 6)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 6")
# points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)
# lambda = 3
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 3)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 3")
# points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)
# lambda = 1
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 1)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 1")
# points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)
# lambda = 0.1
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 0.1)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 0.1")
# points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)
# lambda = 0
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 0)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 0")
# points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)

#What happens for lambda going to infinity?
pfit <- gam(y ~ s(x, bs = "ps", k = 30), sp = 10000000)
par(mfrow=(c(1,1)))
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 10000000")
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)


#the smaller lambda, the wigglier the curve. For λ → ∞, we obtain a straight line! Reason: per default, second differences are chosen. Due to the almost infinite penalization of
#the second differences, the estimator corresponds to functions of the form where second
#differences are 0. We can show that this is the class of linear functions.

#Equivalently, we obtain quadratic function if third instead of second differences are considered (for λ → ∞)

pfit <- gam(y ~ s(x, bs = "ps", k = 30, m = 3), sp = 10000000)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 10000000")
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)

#And a constant function for first differences:
pfit <- gam(y ~ s(x, bs = "ps", k = 30, m = 1), sp = 10000000)
plot(pfit, se = FALSE, rug = F, lwd = 2, col = "red", ylim = c(-2,3),
     shift = coef(pfit)[1], main = "lambda = 10000000")
points(x,y)
curve(f(x), 0, 1, lwd = 2, add = TRUE)




