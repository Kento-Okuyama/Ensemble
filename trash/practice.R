# AR-1 Model Simulation

# Set parameters
phi <- 0.5 # Autoregressive coefficient
sigma <- 1 # Standard deviation of noise
n <- 100 # Sample size

# Generate white noise
set.seed(123) # Set random seed for reproducibility
epsilon <- rnorm(n, mean = 0, sd = sigma)

# Generate AR-1 process
y <- rep(0, n)
for (t in 2:n) {
  y[t] <- phi * y[t-1] + epsilon[t]
}

# Plot the result
plot(y, type = 'o', main = "AR-1 Process Simulation")


# Fit an AR-1 model to the simulated data to estimate parameters

# Use the arima function to fit an AR(1) model
# y is your time series data
fit <- arima(y, order = c(1, 0, 0))

# Display the estimated parameters
print(fit)

# The output includes estimates for phi (ar1) and the noise standard deviation


