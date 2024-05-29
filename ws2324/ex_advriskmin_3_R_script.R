#’ @param X the feature input matrix X
#’ @param y the outcome vector y
#’ @param theta parameter vector for the model (2-dimensional)
# Load MASS and data set forbes
library(MASS)
data(forbes)
attach(forbes)

# initialize the data set
X = cbind(rep(1,17),bp)
y = pres
#’ function to represent your models via the parameter vector theta = c(theta_1, theta_2)
#’ @return a predicted label y_hat for x
f <- function(x, theta){
  return((exp(theta[2]*x[2])*theta[1]*x[1]))
}
#’ @return a vector consisting of the optimal parameter vector
optim_coeff <- function(X,y){
  #’ @return the empirical risk of a parameter vector theta
  emp_risk <- function(theta){
    sum( (log(y) - log(apply(X,1,f,theta)))^2 )
  }
  return(
    optim(c(0.4,0.5),
          emp_risk,
          method = "L-BFGS-B",
          lower=c(0,-Inf),
          upper=c(Inf,Inf))$par)
  # note that c(0.4,0.5) can be replaced by any other theta vector
  # satisfying the constraint theta[1]>0
}

# optimal coefficients
hat_theta = optim_coeff(X,y)
print(hat_theta)
## [1] 0.38050940 0.02059961
# Checking Forbes’ model visually
f_x <- function(x, theta){
  return((exp(theta[2]*x)*theta[1]))
}
curve(f_x(x,theta = hat_theta),min(bp),max(bp),xlab="x (bp)",ylab="y (pres)")
points(pres~bp,col=2)

# Alternative solution
hat_theta_2 = cov(bp,log(pres))/(var(bp))
hat_theta_1 = exp(mean(log(pres))-hat_theta_2*mean(bp))
curve(f_x(x,theta = hat_theta),min(bp),max(bp),xlab="x (bp)",ylab="y (pres)")
curve(f_x(x,theta = c(hat_theta_1,hat_theta_2)),min(bp),max(bp),add=T,col=2)
points(pres~bp)
