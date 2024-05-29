# Supervised Learning
## Information Theory 1

# Sample points according to the true distribution and visualize the 
# KLD for different parameter settings of the 
# Gaussian distribution (including the optimal one if available)

nr_points = 1000
p = 0.5
n = 100
# Creating data
X = rbinom(nr_points, prob = p, size = n)

# define different Normal density functions
normal_optimal = function(x) dnorm(
  x, mean = p*n, sd = sqrt(p*(1-p)*n)
)
normal_shift = function(x) dnorm(
  x, mean = p*n + 10, sd = sqrt(p*(1-p)*n)
)
normal_scale_increase = function(x) dnorm(
  x, mean = p*n, sd = sqrt(p*(1-p)*n)*2
)
normal_right_scale_decrease <- function(x) dnorm(
  x, mean = n*p + 20, sd = p*(1-p))


hist(X, breaks = 25, xlim = c(10, 100), freq = FALSE)
curve(normal_optimal, from = 10, to = 100, add = TRUE, col = "green")
curve(normal_shift, from = 10, to = 100, add = TRUE, col = "blue")
curve(normal_scale_increase, from = 10, to = 100, add = TRUE, col = "orange")
curve(normal_right_scale_decrease, from = 10, to = 100, add = TRUE, col = "red")

kld_value <- function(mu, sigma2) {
  0.5*log(sigma2) + 0.5*sigma2^(-1) * (n*p*(1-p) + (n*p -mu)^2)
}

(optimal_green <- kld_value(n*p,n*p*(1-p)))
(shift_blue <- kld_value(n*p-10,n*p*(1-p)))
(scale_increase_orange <- kld_value(n*p,n*p*(1-p)*4))
(right_scale_decrease_red <- kld_value(n*p+20, (p*(1-p))^2))

# (c) Create a surface plot with axes n and p and colour value equal to the KLD for the optimal normal distribution.

p_seq <- seq(0.01, 0.99, l = 100)
n_seq <- seq(10, 500, by = 100)
B <- 1000
kld_value_approx <- function(n,p){
  # sample a large number of data points from true distribution
  x <- rbinom(B, prob = p, size = n)
  # approximate the mean; threshold values to 0 if < 0 due
  # to the approximation
  pmax(
    mean(
      dbinom(x, prob = p, size = n, log = TRUE) -
        dnorm(x, mean = n*p, sd = sqrt(n*p*(1-p)), log = TRUE),
      na.rm = TRUE
    ),
    0)
}
kld_val <- sapply(n_seq, function(this_n)
  sapply(p_seq, function(this_p) kld_value_approx(this_n, this_p)))
cols <- rev(colorRampPalette(c("darkred","red","blue","lightblue"))(50))

filled.contour(x = p_seq, y = n_seq, z = kld_val,
               xlab = "p", ylab = "n",
               col = cols
)

# Information Theory 2
# Implementation of Smoothed Cross-Entropy Loss

#’ @param label ground truth vector of the form (n_samples,).
#’ Labels should be "1","2","3" and so on.
#’ @param pred Predicted probabilities of the form (n_samples,n_labels)
#’ @param smoothing Hyperparameter for label-smoothing
smoothed_ce_loss <- function(
    label,
    pred,
    smoothing)
{
  num_samples <- NROW(pred)
  num_classes<- NCOL(pred)
  # Let’s make some assertions:
  # label should be a 1-D array.one-hot encoded label is not necessary
  stopifnot(NCOL(label)==1)
  # smoothing hyperparameter in allowed range
  stopifnot((smoothing>=0 & smoothing <= 1))
  # Same amount of rows in labels and predictions
  stopifnot((NROW(label)== num_samples))
  # Predicted probabilities must have as many columns as labels
  stopifnot(length(unique(label)) == num_classes)
  #Calculate the base level
  smoothing_per_class <- smoothing / num_classes
  # build the label matrix. Shape = [ num_samples, num_classes]
  # Start with the base level
  smoothed_labels_matrix = matrix(smoothing_per_class,
                                  nrow=num_samples,ncol=num_classes)
  # Add the smoothed correct labels
  true_labels_loc=cbind(1:num_samples, label)
  smoothed_labels_matrix[true_labels_loc]= 1 - smoothing + smoothing_per_class
  cat("Labels matrix:\n")
  print(smoothed_labels_matrix)
  # Calculate the loss
  cat("Loss for each sample:\n ",
      rowSums(- smoothed_labels_matrix * log(pred)))
  loss <- mean(rowSums(- smoothed_labels_matrix * log(pred)))
  cat("\n Loss:\n",loss)
  return (loss)
}

# Let’s build a "confident model", the model has very high predicted
#probabilities for one of the labels
label= c(1,2,2,3,1)
pred= rbind(
  c(0.85,0.10,0.05),
  c(0.05,0.9,0.05),
  c(0.02,0.95,0.03),
  c(0.13,0.02,0.85),
  c(0.86,0.04,0.1))

# cross entropy means smoothing=0
smoothing=0
loss<-smoothed_ce_loss(label,pred,smoothing)
