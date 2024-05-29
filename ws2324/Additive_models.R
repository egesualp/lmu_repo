sim_Train <- runif(250, 0, 1987200)
X_Train <- sim_Train / 1987200
Y_Train <- sin(2*(4*X_Train - 2)) + 2*exp(-(16)^2*(X_Train-0.5)^2) + rnorm(250, 0, (0.3)^2)

library(ggplot2)
df_train <- data.frame(Y_Train, X_Train)
colnames(df_train) <- c("Y", "X")
# Function to fit a polynomial model and return the fitted values
fit_polynomial_model <- function(degree, data) {
       poly_terms <- poly(data$X, degree, raw = TRUE)
       model <- lm(Y ~ poly_terms, data = data)
       fitted_values <- predict(model, newdata = data)
       return(fitted_values)
   }
# Create a data frame to store results
results <- data.frame(X = df_train$X, Y = df_train$Y)
# Specify degrees for the polynomial models
degrees <- c(1, 2, 3, 10, 50, 250)

for (degree in degrees) {
       results[paste("Degree", degree)] <- fit_polynomial_model(degree, df_train)
}

results_long <- tidyr::gather(results, key = "Model", value = "FittedValues", -X, -Y)

# Create the scatter plot with fitted polynomial models
ggplot(results_long, aes(x = X, y = Y)) +
       geom_point() +
       geom_line(aes(y = FittedValues, color = Model)) +
       labs(title = "Scatter Plot with Fitted Polynomial Models",
                      x = "X",
                      y = "Y") +
       theme_minimal()


# Adding Test data
sim_Test <- runif(500, 0, 1987200)
X_Test <- sim_Test / 1987200
Y_Test <- sin(2*(4*X_Test - 2)) + 2*exp(-(16)^2*(X_Test-0.5)^2) + rnorm(500, 0, (0.3)^2)

df_test <- data.frame(Y_Test, X_Test)
colnames(df_test) <- c("Y", "X")

# Function to fit a polynomial model and return the fitted values
fit_polynomial_model <- function(degree, train, test) {
  # Fit the model using the training data
  poly_terms_train <- poly(train$X, degree, raw = TRUE)
  model <- lm(Y ~ poly_terms_train, data = train)
  
  # Create polynomial terms for the test data
  poly_terms_test <- poly(test$X, degree, raw = TRUE)
  
  # Make predictions using the model and the polynomial terms of the test data
  fitted_values <- predict(model, newdata = poly_terms_test)
  
  mse <- mean((fitted_values - test$Y)^2)
  
  return(list(FittedValues = fitted_values, MSE = mse))
}


# Create a data frame to store results
results_test <- data.frame(X = df_test$X, Y = df_test$Y)

# Specify degrees for the polynomial models
degrees <- c(1, 2, 3, 10, 50, 250)

# Loop through each degree, fit the model, and store fitted values in the results data frame
for (degree in degrees) {
  model_results <- fit_polynomial_model(degree, df_train, df_test)
  results_test[paste("Degree", degree)] <- model_results$FittedValues
  
  # Print MSE for each model
  cat(sprintf("Degree %d MSE: %.4f\n", degree, model_results$MSE))
}

# Reshape the data for ggplot
results_long <- tidyr::gather(results_test, key = "Model", value = "FittedValues", -X)

# Create the scatter plot with fitted polynomial models
ggplot() +
  geom_point(data = df_test, aes(x = X, y = Y), color = "black") +
  geom_line(aes(x = results_long$X, y = results_long$FittedValues, color = results_long$Model)) +
  labs(title = "Scatter Plot with Fitted Polynomial Models",
       x = "X",
       y = "Y") +
  theme_minimal()

