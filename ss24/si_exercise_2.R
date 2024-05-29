# Exercise 2
# Question 1: Sketch of risk functions for different estimators and parameters
# Check packages availability
list_of_packages <- c("ggplot2", "dplyr", "scales", "grid", "gridExtra", "ggthemes")
new_packages <- list_of_packages[!(list_of_packages %in%
installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
sapply(list_of_packages, require, character.only = TRUE)
# Initialize parameters
a = 1
b = 1
n = 16
theta = seq(-5, 5, by = 0.001) # Produce 10000 values for the parameter of interest
# Create function to calculate Risk as a function of the parameters values and estimator
risk<-function(a = a, b = b, n = n, theta = theta, include_bias = TRUE){
output = (pi^2*b^2) / 3*n
return(output)
}
# Evaluate this function for different parameter values and estimators
r11 = risk(a = a, b = b, n = n, theta = theta, include_bias = TRUE)
r22 = risk(a = sqrt(n)/2, b = sqrt(n)/2, n = n, theta = theta, include_bias = TRUE)
rsamplemean = risk(a = 0, b = 0, n = n, theta = theta, include_bias = FALSE)
# Bonus:
r05 = (0.5 - theta)^2
# Load necessary packages to plot risk functions
library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)
# Group risk function in a data frame
df = data.frame("Theta" = theta, "T11" = r11, "T22" = r22, "Tc" = rsamplemean, "T0.5" = r05)
colors <- c("T (1,1)" = "darkgreen", "T (2,2)" = "blue", "T (0,0)" = "red", "T = 0.5" = "purple")
# Plot result with labels
ggplot() +
geom_line(data = df, aes(x = Theta, y = T11, color = "T (1,1)"), size = 1.2, linetype = "dashed") +
geom_line(data = df, aes(x = Theta, y = T22, color = "T (2,2)"), size = 1.2, linetype = "dashed") +
geom_line(data = df, aes(x = Theta, y = Tc, color = "T (0,0)"), size = 1.2, linetype = "dashed") +
geom_line(data = df, aes(x = Theta, y = T0.5, color = "T = 0.5"), size = 1.2, linetype = "dashed")+
xlab(expression(pi)) +
ylab(expression(R('T', pi)))+
xlim(c(0.0, 1.0))+
ylim(c(0.000, 0.020))+
labs(title = "Risk function comparison", subtitle = "For different estimators and parameters", color = "Estimators")+
scale_color_manual(values = colors)+theme_economist_white()+scale_fill_economist()+
theme(plot.title = element_text(size=24, hjust = 0))+theme(plot.subtitle = element_text(size=16, hjust = 0))+
theme(axis.text=element_text(size=13), axis.title=element_text(size=15,face="bold"), legend.text = element_text(size = 13), legend.title = element_text(size = 13))
