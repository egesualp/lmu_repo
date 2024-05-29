library(ggplot2)
set.seed(123)
# simulate 50 binary observations with noisy linear decision boundary
n = 50
X = matrix(runif(2*n), ncol = 2)
X_model = cbind(1, X)
y = -((X_model %*% c(0.3, -1, 1) + rnorm(n, 0, 0.3) < 0) - 1)
df = as.data.frame(X)
df$type = as.character(y)
ggplot(df) +
  geom_point(aes(x = V1, y = V2, color=type)) +
  xlab(expression(x[1])) +
  ylab(expression(x[2]))

theta = c(0, 0, 0)
remps = NULL
thetas = NULL
for(i in 1:30){
  exp_f = exp(X_model %*% theta)
  remps = rbind(remps, sum((y - 1/(1+exp_f))^2))
  hess = t(X_model) %*%
    (c((2 * exp_f*(2 * exp_f - y*(exp_f^2 - 1) - 1))/(exp_f + 1)^4 ) * X_model)
  grad = c(t(2*(y * exp_f - (1 + exp_f^-1)^-1) / (exp_f + 1)^2) %*% X_model)
  theta = theta + solve(hess, -grad)
  thetas = rbind(thetas, theta)
}

ggplot(data.frame(remps, t=1:nrow(remps)), aes(x=t, y=remps)) +
  geom_line() + ylab(expression(R[emp]))

ggplot(data.frame(theta = c(thetas), t=rep(1:nrow(thetas),3),
                  id = as.factor(rep(c(0, 1, 2), each= nrow(thetas)))),
       aes(x = t, y=theta)) +
  geom_line(aes(color = id)) + ylab(expression(theta))
