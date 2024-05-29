rm(list = ls())
# Plotting
library(ggplot2)
library(ggExtra)
library(ggdensity)
library(GGally)
library(cowplot)
# Copula Estimation
library(copula)
library(rvinecopulib)
set.seed(1234)

# Create a Gaussian copula with correlation 0.5 and one archimedean copula (Frank, Gumble,
# Clayton) with parameter θ = 2 (functions normalCopula, etc.). You can sample from these
# copulas using the rCopula function and get densities with dCopula

# Copula constructor
norm_cop = normalCopula(0.5)
clay_cop = claytonCopula(2)

# Sampling from copulas
sample_norm = rCopula(1e4, norm_cop)
sample_clay = rCopula(1e4, clay_cop)

# Proof of precious exercise
pCopula(u = c(0.5, 0.5), clay_cop)

#  Finally, create bivariate
# distributions with the mvdc function (always assuming N(0, 1) marginals)

# Multivariate distribution constructor
margin_params = list(list(mean = 0, sd = 1), list(mean = 0, sd = 1))
binorm = mvdc(norm_cop, margins = c("norm", "norm"), paramMargins = margin_params)
biclay = mvdc(clay_cop, margins = c("norm", "norm"), paramMargins = margin_params)

# Sampling from MV distributions
samples_binorm = rMvdc(500, binorm)
samples_biclay = rMvdc(500, biclay)

?mvdc
# Get sample densities 
dens_binom = dMvdc(samples_binorm, binorm)
dens_biclay = dMvdc(samples_biclay, biclay)
dMvdc(x= c(0,0), biclay)
dMvdc(x= c(0,0), binorm)

# Create scatter plots for both the copulas and the multivariate distributions, adding contours to
# visualize the densities. Another way of visualizing bivariate distribution densities are 3D plots:
# try those out as well.

# Hint: to visualize the copula density you may need to span a grid over the bivariate margins using
# expand.grid and evaluate the copula/multivariate density at the given equidistant grid points.


# Create data for sample and grid
get_cop_data = function(obj, sample_size = 500, is_log = TRUE, g_size = 100) {
  samples = if(class(obj) == "mvdc") rMvdc(sample_size, obj) else rCopula(sample_size, obj)
  colnames(samples) = c("x", "y")
  dens_cop = if(class(obj) == "mvdc") {
    dMvdc(samples, obj, log = is_log)
  } else dCopula(samples, obj, log = is_log)
  samples = cbind.data.frame(samples, density = dens_cop)
  grid = as.matrix(expand.grid(x = seq(max(samples$x), min(samples$x),length.out = g_size),
                               y = seq(max(samples$y), min(samples$y), length.out = g_size)))
  dens_grid = if(class(obj) == "mvdc") {
    dMvdc(grid, obj, log = is_log)
  } else dCopula(grid, obj, log = is_log)
  grid = cbind.data.frame(grid, density = dens_grid)
  list(grid = grid, samples = samples)
}

norm_cop_dta = get_cop_data(norm_cop)
clay_cop_dta = get_cop_data(clay_cop)
norm_mvd_dta = get_cop_data(binorm)
clay_mvd_dta = get_cop_data(biclay)

plot_cop_data = function(dta, title_chr) {
  ggplot(dta$grid, aes(x = x, y = y, z=density)) +
    geom_contour_filled(show.legend = FALSE) +
    geom_point(data = dta$samples, colour = "white", alpha = 0.5) +
    ggtitle(title_chr) +
    theme_minimal()
}
plot_grid(plot_cop_data(norm_cop_dta, "Gaussian Copula"),
          plot_cop_data(clay_cop_dta, "Clayton Copula"), nrow = 1)

plot_mvd_data = function(dta, title_chr) {
  plot_gcopMVD = ggplot(dta$grid, aes(x = x, y = y, z=density)) +
    geom_contour_filled(show.legend = FALSE) +
    geom_point(data = dta$samples, colour = "white", alpha = 0.5) +
    ggtitle(title_chr) +
    theme_minimal()
  ggMarginal(plot_gcopMVD, type="density")
}
plot_grid(
  plot_mvd_data(norm_mvd_dta, "2D-Dist. with Gaussian Copula"),
  plot_mvd_data(clay_mvd_dta, "2D-Dist. with Clayton Copula"),
  nrow = 1
)

# When dealing with copulas, association measures that are often used are Spearman’s Rho and
# Kendalls Tau. What is are the advantages of those correlation coefficients and why do we use
# them when dealing with copulas? In which cases might those rank correlations fail?

# They use non linear relationships, more specificly
# they work with the ranks of the data, similarly as copulas
# Spearman's rho
# more flexible than Pearsons correlation coefficient, as it does not
# revolve around the linearity of the relationship.

x_1 = rnorm(500)
cor_data = data.frame(
  x_1 = x_1,
  x_2 = (x_1)^2,
  x_exp = exp(x_1)
)
p1 = ggplot(cor_data, aes(x=x_1, y=x_exp)) +
  geom_point()
p2 = ggplot(cor_data, aes(x=x_1, y=x_2)) +
  geom_point()
plot_grid(p1, p2, nrow=1)

# kendals tau und spearmans rho measure the monotony of x in y = f(x)
# if we are given a monoton, non-linear gradient, kendall und spearman are one 1,
# even though the transformation mighzt be non linear.

c("pearson exp" = cor(x = cor_data$x_1, cor_data$x_exp, method = "pearson"),
  "spearman exp" = cor(x = cor_data$x_1, cor_data$x_exp, method = "spearman"))

# and an example for non-monotonic relationship
c("pearson squared" = cor(x = cor_data$x_1, cor_data$x_2, method = "pearson"),
  "spearman squared" = cor(x = cor_data$x_1, cor_data$x_2, method = "spearman"))

data(uranium, package="copula")
ggpairs(uranium, upper = list(continuous = wrap(ggally_cor,method = "spearman")))

# From now on we will limit our analysis to Cobalt (Co) and Titanium (Ti).

uranium_CoTi = subset(uranium, select = c("Co", "Ti"))
ml_estimates = apply(uranium_CoTi, 2, function(x) c(mean = mean(x), sd = sd(x)), simplify = FALSE)
plot_md_co = ggplot(uranium_CoTi, aes(x = Co)) +
  geom_histogram(aes(y=..density..), colour="grey", fill="white", bins = 25) +
  stat_function(fun = dnorm, colour = "purple",
                args = ml_estimates$Co) +
  ggtitle("Marginal Distr. Cobalt")
plot_md_ti = ggplot(uranium_CoTi, aes(x = Ti)) +
  geom_histogram(aes(y=..density..), colour="grey", fill="white", bins = 25) +
  stat_function(fun = dnorm, colour = "purple",
                args = ml_estimates$Ti)+
  ggtitle("Marginal Distr. Titanium")
p = ggplot(uranium_CoTi, aes(x = Co, y = Ti)) + stat_density_2d() +
  geom_point(alpha = 0.5, colour = "grey")
ggm12<-ggMarginal(p, type="density")

plot_grid(plot_grid(plot_md_co, plot_md_ti, nrow=1), ggm12, nrow=2)

# A priori, it is hard to choose a copula which models the dependency structure properly. In practice,
# model selection routines are often used to find the best fit from a pool of available copulas.

# Use the bicop function in the rvinecopulib package to estimate most suitable one parametric
# copula by AIC criterion. In terms of modelling the margins

  # Use the normal distributed margins from the previous exercise (see pnorm) and
  # use empiric marginals (see pseudo_obs)

pnorm_CoTi = cbind.data.frame(
  Co = pnorm(uranium_CoTi$Co, mean = ml_estimates$Co["mean"], ml_estimates$Co["sd"]),
  Ti = pnorm(uranium_CoTi$Ti, mean = ml_estimates$Ti["mean"], ml_estimates$Ti["sd"])
)

ml_cop = bicop(pnorm_CoTi, family_set = "onepar", selcrit = "aic")
summary(ml_cop)

pit_CoTi = pseudo_obs(uranium_CoTi)
emp_cop = bicop(pit_CoTi, family_set = "onepar", selcrit = "aic")
summary(emp_cop)

# The Loglikelihood is almost identical. In practice we mostly take the empirical or kernel density
# estimates of the marginals to maximize the likelihood of the copula and to fit θ. They often work
# better than distributional assumptions.
p1 = ggplot(pit_CoTi, aes(x = Co, width = 10)) +
  geom_histogram(binwidth = 0.1) + ggtitle("Empirical CDF Co")
p2 = ggplot(pit_CoTi, aes(x = Ti, width = 10)) +
  geom_histogram(binwidth = 0.1) + ggtitle("Empirical CDF Ti")
p3 = ggplot(pnorm_CoTi, aes(x = Co, width = 10)) +
  geom_histogram(binwidth = 0.1) + ggtitle("Norm CDF Co")
p4 = ggplot(pnorm_CoTi, aes(x = Ti, width = 10)) +
  geom_histogram(binwidth = 0.1) + ggtitle("Norm CDF Co")
plot_grid(p1, p2, p3, p4, nrow = 2)

# Finally, visualize the data with a scatter plot, and add the multivariate distributions built in the
# previous exercise. Discuss the fit.

# However, we can still choose weather we use empirical margins or
# known univariate (e.g. normal) margins for the MVD
ml_frank = mvdc(frankCopula(param = as.vector(ml_cop$parameters), dim = 2),
                margins = c("norm","norm"),
                paramMargins = list(as.list(ml_estimates$Co),
                                    as.list(ml_estimates$Ti)))
emp_frank = fitCopula(copula = frankCopula(param = as.vector(ml_cop$parameters), dim = 2),
                      pobs(uranium_CoTi))
# sample from an empirical copula
u_sim <- rCopula(1e4, emp_frank@copula)
x_sim <- sapply(1:ncol(uranium_CoTi), function(j) quantile(uranium_CoTi[, j],
                                                           probs = u_sim[, j]))
plot_2 <- ggplot(data = uranium, aes(x = Co, y = Ti)) +
  stat_density_2d(data = as.data.frame(rMvdc(10000, ml_frank)),
                  aes(x = V1, y = V2, color = after_stat(level)), show.legend = FALSE) +
  geom_point(alpha = 0.4, colour = "grey") +
  ggtitle("Frank Copula with Normal Margins") +
  theme_minimal()
plot_3 <- ggplot(data = uranium, aes(x = Co, y = Ti)) +
  stat_density_2d(data = as.data.frame(x_sim),
                  aes(x = V1, y = V2, color = after_stat(level)), show.legend = FALSE) +
  geom_point(alpha = 0.4, colour = "grey") +
  ggtitle("Frank Copula with Empirical Margins") +
  theme_minimal()
plot_grid(plot_2, plot_3, nrow=1)

# Advantage empirical marginal: no distributional assumptions
# Disadvantage empirical marginal: Sampling limited to the domain of the quantile function
# (roughly the Cartesian product of the marginals plus in between pts,
# won't exceed the domain of the marginals)
# Advantage parametric marginal: if the distributional assumptions hold,
# we can draw samples and estimate the probability density of singular new observations
# Disadvantage parametric marginal: distributional assumptions on the margins


