library(mvtnorm)
library(ggplot2)

set.seed(329847)

# Gibbs sampler, bivariate normal, Cov=Cor=known, non-informative
# prior \propto const
varx1 <- 1
varx2 <- 1

# Check out:
# Sampler needs more runs if the correlation between the two components is high
corrx1x2 <- 0.1
covmat <- matrix( nrow=2, ncol=2, data=c(varx1, corrx1x2, corrx1x2, varx2))
print(covmat)

# Generate some data
n <- 50
x <- rmvnorm(n=n, mean=c(3,10), sigma=covmat )

# Summary measures
x1bar <- mean( x[,1] )
x2bar <- mean( x[,2] )

# Simulate draws from the posterior using the Gibbs sampler 

numBurnIn <- 500  # these samples are dropped later 
numSamples <- 1000
mu <- matrix( nrow=(numBurnIn+numSamples), ncol=2, data=0 )

# Starting value
mu[1,2] <- -7

# Full-conditional mu_1 | mu_2, data, START
mu[1,1] <- rnorm( n=1, mean=x1bar + corrx1x2 * (mu[1,2]-x2bar), sd=sqrt( (1-corrx1x2^2)/n) )

# now lets iterate ...

for ( i in 2:(numBurnIn+numSamples) ) {
  # Full-conditional mu_2 | mu_1, data
  mu[i,2] <- rnorm( n=1, mean=x2bar + corrx1x2 * (mu[i-1,1]-x1bar), sd=sqrt( (1-corrx1x2^2)/n))
  # Full-conditional mu_1 | mu_2, data
  mu[i,1] <- rnorm( n=1, mean=x1bar + corrx1x2 * (mu[i,2]-x2bar), sd=sqrt( (1-corrx1x2^2)/n))
  #matplot( mu[1:i, ], type="l")
}

# samples without burnin:
mu.used <- as.data.frame(mu[(numBurnIn+1):(numBurnIn+numSamples),])
colnames(mu.used) <- c("mu.1","mu.2")

print("Estimated posterior expectations of mu_1 and mu_2 (without burnin):")
print( colMeans(mu.used) )
print("For comparison, the ML estimate is:")
print(colMeans(x))

p <- ggplot(mu.used, aes(x=mu.1)) + geom_density()
print(p)

# trace plot of the first 100 samples
plot(mu[1:600,1], type="l")

# correlation of the samples
print("Correlation of the samples:")
print(cor(mu.used))



