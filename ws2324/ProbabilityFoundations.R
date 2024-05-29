### Foundations of Probability (datacamp)

## Flipping multiple coins

# Flip one coin in 10 trials
rbinom(10, 1, .5) # results a vector of results

# Flip ten coins in 1 trial
rbinom(1, 10, .5) # results total success

# size -> number of flips
# n -> number of trials
# p -> probability

# Generate 100 occurrences of flipping 10 coins, each with 30% probability
rbinom(100, 10, 0.3)

# What is the expected value of a binomial distribution where 25 coins are flipped, each having a 30% chance of heads?
# Calculate the expected value using the exact formula
25*0.3

# Confirm with a simulation using rbinom
mean(rbinom(10000, 25, 0.3))

# Calculate the variance using the exact formula
25 * 0.3 * (1-0.3)

# Confirm with a simulation using rbinom
var(rbinom(10000, 25, 0.3))

# Independent events & nested probability
# You've already simulated 100,000 flips of coins A and B
A <- rbinom(100000, 1, .4)
B <- rbinom(100000, 1, .2)

# Simulate 100,000 flips of coin C (70% chance of heads)
C <- rbinom(100000,1,0.7)

# Estimate the probability A, B, and C are all heads
mean(A&B&C)

# Probability of A OR B
# Use rbinom to simulate 100,000 draws from each of X and Y
X <- rbinom(100000, 10, 0.6)
Y <- rbinom(100000, 10, 0.7)

# Estimate the probability either X or Y is <= to 4
mean(X <= 4 | Y <= 4) - mean(X <= 4 & Y <= 4)

# Use pbinom to calculate the probabilities separately
prob_X_less <- pbinom(4, 10, 0.6)
prob_Y_less <- pbinom(4, 10, 0.7)

# Combine these to calculate the exact probability either <= 4
prob_X_less + prob_Y_less - (prob_X_less *s prob_Y_less)

# Multiplying random variables (with scalar)
# Simulate 100,000 draws of a binomial with size 20 and p = .1
# Estimate the expected value of X
mean(X)

# Estimate the expected value of 5 * X
mean(5*X)

# Estimate the variance of X
var(X)

# Estimate the variance of 5 * X
var(5*X)

# Adding random variables
# Estimate the expected value of X + Y
mean(X+Y)

# Find the variance of X + Y
var(X + Y)

### Bayesian Statistics
# Simulate 50000 cases of flipping 20 coins from fair and from biased
fair <- rbinom(50000, 20, 0.5)
biased <- rbinom(50000, 20, 0.75)

# How many fair cases, and how many biased, led to exactly 11 heads?
fair_11 <- sum(fair == 11)
biased_11 <- sum(biased == 11)

# Find the fraction of fair coins that are 11 out of all coins that were 11
fair_11 / (biased_11 + fair_11)

# Simulate 8000 cases of flipping a fair coin, and 2000 of a biased coin
fair_flips <- rbinom(8000, 20, 0.5)
biased_flips <- rbinom(2000, 20, 0.75)

# Find the number of cases from each coin that resulted in 14/20
fair_14 <- sum(fair_flips == 14)
biased_14 <- sum(biased_flips == 14)

# Use these to estimate the posterior probability of having a fair coin
fair_14 / (biased_14 + fair_14)

# There is a 50% chance the coin is fair and a 50% chance the coin is biased.
# Use dbinom to calculate the probability of 11/20 heads with fair or biased coin
probability_fair <- dbinom(11, 20, 0.5)
probability_biased <- dbinom(11, 20, 0.75)

# Calculate the posterior probability that the coin is fair
probability_fair / (probability_fair + probability_biased)

# suppose we had set a prior probability of a 99% chance that the coin is fair (50% chance of heads), and only a 1% chance that the coin is biased (75% chance of heads).
# Use dbinom to find the probability of 16/20 from a fair or biased coin
probability_16_fair <- dbinom(16, 20, .5)
probability_16_biased <- dbinom(16, 20, .75)

# Use Bayes' theorem to find the posterior probability that the coin is fair
(probability_16_fair * .99) / (probability_16_fair * .99 + probability_16_biased * .01)

# Comparing cumulative density
# If you flip 1000 coins that each have a 20% chance of being heads, what is the probability you would get 190 heads or fewer?

# Simulations from the normal and binomial distributions
binom_sample <- rbinom(100000, 1000, .2)
normal_sample <- rnorm(100000, 200, sqrt(160))

# Use binom_sample to estimate the probability of <= 190 heads
mean(binom_sample <= 190)

# Use normal_sample to estimate the probability of <= 190 heads
mean(normal_sample <= 190)

# Calculate the probability of <= 190 heads with pbinom
pbinom(190, 1000, .2)

# Calculate the probability of <= 190 heads with pnorm
pnorm(190, 200, sqrt(160))

# Poission distribution
# When n is large and p is small

# Draw a random sample of 100,000 from the Binomial(1000, .002) distribution
binom_sample <-  rbinom(100000, 1000, .002)

# Draw a random sample of 100,000 from the Poisson approximation
poisson_sample <- rpois(100000, 2)

# Compare the two distributions with the compare_histograms function
compare_histograms(binom_sample, poisson_sample) # this is a function to creates a figure with 2 subplots

# Simulate 100,000 draws from Poisson(2)
poisson_sample <- rpois(100000, 2)

# Find the percentage of simulated values that are 0
mean(poisson_sample == 0)

# Use dpois to find the exact probability that a draw is 0
dpois(0, 2)

## A new machine arrives in a factory. 
## This type of machine is very unreliable: every day, it has a 10% chance of breaking permanently. 
## How long would you expect it to last?

# Notice that this is described by the cumulative distribution 
# of the geometric distribution, and therefore the pgeom() function. 
# pgeom(X, .1) would describe the probability that there are X working days before the day it breaks (that is, that it breaks on day X + 1).

# Find the probability the machine breaks on 5th day or earlier
pgeom(4, .1)

# Find the probability the machine is still working on 20th day
1 - pgeom(19, .1)

# Calculate the probability of machine working on day 1-30
still_working <- 1 - pgeom(0:29, .1)

# Plot the probability for days 1 to 30
qplot(1:30, still_working)

