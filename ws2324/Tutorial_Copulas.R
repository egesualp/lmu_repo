# Probability Integral Transform (PIT)

library(ggplot2)
library(patchwork)
theme_set(theme_bw())

?distributions

n <- 5000
bins <- 50

## transform data from a continuous distribution to uniform distributed data
# Exponential distribution
# draw random numbers
x <- rexp(n)
# PIT with cumulative distribution function
x_pit <- pexp(x)

# plot empirical distributions
p_x <- ggplot() +
  geom_histogram(aes(x = x), bins = bins) +
  labs(title = "Histogram of x")
p_xpit <- ggplot() +
  geom_histogram(aes(x = x_pit), bins = bins) +
  labs(title = "Histogram of F(x)", x = "F(x)")
p_x + p_xpit


## any continuous distribution
# function to "n" simulate random numbers from a distribution "dist"
# distribution parameters can be passed via "..."
# "dist" needs to be a string of the distribution name of a distribution implemented in package stats
pit_x_to_u <- function(dist, n = 50000, ...){
  # simulate random numbers
  x <- do.call(paste0("r", dist), list(n = n, ...))
  # PIT
  u <- do.call(paste0("p", dist), list(q = x, ...))
  
  data.frame(x = x, u = u)
}
plot_x_to_u <- function(dat, bins = 30){
  p_x <- ggplot(dat) +
    geom_histogram(aes(x = x), bins = bins) +
    labs(title = "Histogram of x", x = "x")
  p_xpit <- ggplot(dat) +
    geom_histogram(aes(x = u), bins = bins) +
    labs(title = "Histogram of F(x)", x = "F(x)")
  p_x + p_xpit
}

dat <- pit_x_to_u("normal", shape = 10, scale = 3,
                  n = n)
plot_x_to_u(dat, bins = bins)

########################################################################################
# transform uniform data to another distribution
# Gaussian distribution
# draw random numbers from a uniform distribution
u <- runif(n)
# PIT with inverse CDF = quantile function
u_pit <- qexp(x_pit)

# plot empirical distributions
p_u <- ggplot() +
  geom_histogram(aes(x = x), bins = bins) +
  labs(title = "Histogram of u")
p_upit <-
  ggplot() +
  geom_histogram(aes(x = u_pit), bins = bins) +
  labs(title = expression("Histogram of" ~ F^{-1} ~ (u)), x = expression(F^{-1} ~ (u)))
p_u + p_upit


## any continuous distribution
# The function simulates "n" random numbers from a uniform distribution.
# These are then transformed to another distribution using a quantile function
# "dist" needs to be a string of the distribution name of a distribution implemented in package stats
pit_u_to_x <- function(dist, n = 50000, ...){
  # simulate random numbers
  u <- runif(n)
  # PIT with quantile function
  x <- do.call(paste0("q", dist), list(p = u, ...))
  
  data.frame(x = x, u = u)
}
plot_u_to_x <- function(dat, bins = 30){
  p_u <- ggplot(dat) +
    geom_histogram(aes(x = u), bins = bins) +
    labs(title = "Histogram of u")
  p_upit <-
    ggplot(dat) +
    geom_histogram(aes(x = x), bins = bins) +
    labs(title = expression("Histogram of" ~ F^{-1} ~ (u)), x = expression(F^{-1} ~ (u)))
  p_u + p_upit
}

dat <- pit_u_to_x("weibull", shape = 1.5, scale = 0.5,
                  n = n)
plot_u_to_x(dat, bins = bins)


## chatgpt example
# Set the rate parameter for the exponential distribution
lambda <- 0.1  # You can adjust this parameter

# Generate random samples from the exponential distribution
random_samples <- rexp(1000, rate = lambda)

# Compute the cumulative distribution function (CDF)
cdf_values <- pexp(random_samples, rate = lambda)

# Plot the original data and the PIT-transformed data
par(mfrow = c(1, 2))  # Set up a 1x2 grid for plots

# Plot the original data
hist(random_samples, main = "Original Data", xlab = "Time", col = "lightblue", freq = FALSE)

# Plot the PIT-transformed data
hist(cdf_values, main = "Probability Integral Transform", xlab = "Probability", col = "lightgreen", freq = FALSE)

# Add a uniform distribution for comparison
curve(dunif(x), col = "red", add = TRUE, lty = 2, lwd = 2)

# Reset the plotting layout
par(mfrow = c(1, 1))


## Tutorial Sheet 11

library(VineCopula)
library(dplyr)

load("C:\\Users\\esual\\Downloads\\muc.Rdata")
head(muc3)
str(muc3)
summary(muc3)

# restrict to stunde 20 and remove NAs
muc20 <- muc3 %>%
  filter(stunde == 20) %>%
  select(PM10_luis, PM10_Loth, PM10_Prin, PM10_Stach, NO_Loth, NO2_Loth) %>%
  na.omit()

# scatterplot matrix of original data
pairs(muc20, pch = ".", gap = 0)
# There are a lot of outliers/extreme values.
# The variables are positively correlated.

## b) Probability integral transform using pseudo_obs()\
u <- pobs(muc20)

# histogram for transformed data
par(mfrow = c(2,3))
for (i in 1:6){
  hist(u[,i], main = names(muc20)[[i]], bg='grey')
}
par(mfrow=c(1,1))

# c) scatter plot of transformed data
pairs(u, pch = ".", gap = 0)

# d) Fitting bivariate copula between variables

## Between PM10_luis and PM10_Loth
cop12 <- BiCopSelect(u[,1], u[,2], familyset = c(0,1,2,3,4,5))
summary(cop12)

# Kendall's Tau: The Kendall's tau correlation coefficient is given as 0.72. 
# This coefficient measures the strength and direction of dependence between the two variables. 
# A value of 0.72 indicates a moderate to strong positive dependence.
# Upper and Lower Tail Dependence (TD): Upper and lower tail dependence measures are provided. 
# In this case, both are 0.56. 
# These values indicate the strength of dependence in the upper and lower tails of the joint distribution.

# The empirical correlation can also be calculated manually.
cor(u[,1], u[,2], method = "kendall")
cor(u[,1], u[,2], method = "spearman")

# contour plot of the estimated copula
plot(cop12, type = "contour", margins = "unif")
points(u[,1], u[,2], pch=20, cex=0.5)

plot(cop12, type="surface")

# bivariate copula of PM10_luis and PM10_Prin
cop13 <- BiCopSelect(u[,1], u[,3], familyset = c(0,1,2,3,4,5))
summary(cop13)

# contour plot
plot(cop13, type = "contour", margins = "unif")
points(u[,1], u[,3], pch = 20, cex = 0.5)

plot(cop13, type="surface")
