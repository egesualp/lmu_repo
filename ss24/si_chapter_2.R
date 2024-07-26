# Classical Estimation Theory
# Step 1: Generate Sample Data
set.seed(123)  # For reproducibility
n <- 100  # Sample size
mu <- 175  # True mean
sigma <- 30  # Known standard deviation

# Generate sample data from N(mu, sigma^2)
sample_data <- rnorm(n, mean = mu, sd = sigma)

# Step 2: Calculate the Sample Mean
sample_mean <- mean(sample_data)
cat("Sample Mean:", sample_mean, "\n")

# Step 3: Demonstrate Sufficiency
# Calculate the sample mean and variance
sample_variance <- var(sample_data)
cat("Sample Variance:", sample_variance, "\n")

# Likelihood function (simplified for demonstration)
likelihood <- function(mu, data, sigma) {
  n <- length(data)
  sum_sq_diff <- sum((data - mu)^2)
  exp(-sum_sq_diff / (2 * sigma^2)) / (sqrt(2 * pi * sigma^2) ^ n)
}

# Compute likelihood for a range of mu values
mu_values <- seq(160, 170, by = 0.1)
likelihood_values <- sapply(mu_values, likelihood, data = sample_data, sigma = sigma)

# Plot the likelihood function
plot(mu_values, likelihood_values, type = "l", col = "blue", lwd = 2,
     main = "Likelihood Function", xlab = expression(mu), ylab = "Likelihood")

# Step 4: Demonstrate Completeness
# Let's consider a non-trivial function of the sample mean
h <- function(x) x - mu

# Generate different samples and compute h(sample_mean)
set.seed(456)  # For reproducibility
num_trials <- 1000
h_values <- numeric(num_trials)

for (i in 1:num_trials) {
  sample_data <- rnorm(n, mean = mu, sd = sigma)
  sample_mean <- mean(sample_data)
  h_values[i] <- h(sample_mean)
}

# Check if the mean of h(sample_mean) is zero
cat("Mean of h(sample_mean):", mean(h_values), "\n")

# The mean of h(sample_mean) should be approximately zero for a complete statistic

# Another example
# Step 1: Generate Sample Data
set.seed(123)  # For reproducibility
n <- 100  # Sample size
mu <- 165  # True mean
sigma <- 10  # Known standard deviation

# Generate sample data from N(mu, sigma^2)
sample_data <- rnorm(n, mean = mu, sd = sigma)

# Step 2: Calculate S_1 and S_2
S1 <- sum(sample_data)
S2 <- sum(sample_data^2)
cat("S1 (Sum of X_i):", S1, "\n")
cat("S2 (Sum of X_i^2):", S2, "\n")

# Step 3: Define a Non-Trivial Function
h <- function(S1, S2, mu) {
  S1 - n * mu
}

# Step 4: Check Completeness
# Generate multiple samples and compute h(S1, S2)
set.seed(456)  # For reproducibility
num_trials <- 1000000
h_values <- numeric(num_trials)

for (i in 1:num_trials) {
  sample_data <- rnorm(n, mean = mu, sd = sigma)
  S1 <- sum(sample_data)
  S2 <- sum(sample_data^2)
  h_values[i] <- h(S1, S2, mu)
}

# Check if the mean of h(S1, S2) is zero
cat("Mean of h(S1, S2):", mean(h_values), "\n")

# Plot the distribution of h(S1, S2) to visualize the result
hist(h_values, breaks = 30, main = "Distribution of h(S1, S2)",
     xlab = "h(S1, S2)", col = "lightblue", border = "black")
abline(v = 0, col = "red", lwd = 2)

# change in different trial sizes
num_trials_vec <- c()
mean_h_values_vec <- c()

for (num_tr in seq(100000, 1000000, 100000)) {
  num_trials <- num_tr
  num_trials_vec <- c(num_trials_vec, num_trials)
  h_values <- numeric(num_trials)
  for (i in 1:num_trials) {
    sample_data <- rnorm(n, mean = mu, sd = sigma)
    S1 <- sum(sample_data)
    S2 <- sum(sample_data^2)
    h_values[i] <- h(S1, S2, mu)
  }
  mean_h_values_vec  <- c(mean_h_values_vec , mean(h_values))
  
  cat("Number of Trials", num_trials, "Mean of h(S1,S2):", mean(h_values), "\n")
}

plot(
  num_trials_vec,
  mean_h_values_vec,
  type = "o", col = "blue", lwd=2,
  xlab = "# of Trials", ylab = "Mean of h(S1, S2)")

abline(h = 0, col = "red", lwd = 2, lty = 2)  # Add horizontal line at y=0

# Cramer Rao bound
# Define parameters of the true distribution
true_mean <- 5
true_variance <- 4
sigma <- sqrt(true_variance)  # Standard deviation (known)

# Generate a sample from the true distribution
set.seed(123)  # For reproducibility
sample_size <- 100
sample <- rnorm(sample_size, mean = true_mean, sd = sigma)

# Maximum Likelihood Estimation (MLE) for mean of Normal distribution
mle_mean <- mean(sample)

# Cramer-Rao bound calculation for the mean parameter of Normal distribution
cr_bound <- sigma^2 / sample_size

# Print results
cat("True mean:", true_mean, "\n")
cat("MLE estimate of mean:", mle_mean, "\n")
cat("Cramer-Rao bound for mean:", cr_bound, "\n")

# Visualization: Compare true distribution with MLE estimate
library(ggplot2)

# Generate data for plotting true and estimated distributions
x <- seq(-2*sigma + true_mean, 2*sigma + true_mean, length.out = 100)
density_true <- dnorm(x, mean = true_mean, sd = sigma)
density_mle <- dnorm(x, mean = mle_mean, sd = sigma)

# Create data frames for plotting
df_true <- data.frame(x = x, density = density_true, distribution = "True Distribution")
df_mle <- data.frame(x = x, density = density_mle, distribution = "MLE Estimate")

# Combine data frames
df_plot <- rbind(df_true, df_mle)

# Plotting
ggplot(df_plot, aes(x, density, color = distribution, linetype = distribution)) +
  geom_line(size = 1) +
  labs(title = "True Distribution vs MLE Estimate",
       x = "x", y = "Density") +
  theme_minimal() +
  theme(legend.position = "top")


