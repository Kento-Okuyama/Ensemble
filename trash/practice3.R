# ARMA(2,2) Model Simulation

# Set parameters for ARMA(2,2)
phi1 <- 0.5
phi2 <- -0.25
theta1 <- 0.4
theta2 <- -0.3
sigma <- 1e-3 # Standard deviation of noise
n <- 100 # Sample size

# Generate white noise
set.seed(123)
epsilon <- rnorm(n, mean = 0, sd = sigma)

# Initialize vectors
y <- rep(0, n) # Vector to store the ARMA(2,2) process
errors <- rep(0, n) # Vector to store errors (MA part)

# Generate ARMA(2,2) process
for (t in 3:n) {
  errors[t] <- epsilon[t] + theta1 * epsilon[t-1] + theta2 * epsilon[t-2]
  y[t] <- phi1 * y[t-1] + phi2 * y[t-2] + errors[t]
}

# Plot the result
plot(y, type = 'o', main = "ARMA(2,2) Process Simulation")

########################################################
fit1 <- arima(y, order = c(1, 0, 0))
print(fit1)

fit2 <- arima(y, order = c(2, 0, 0))
print(fit2)

fit3 <- arima(y, order = c(3, 0, 0))
print(fit3)

fit4 <- arima(y, order = c(4, 0, 0))
print(fit4)

fit5 <- arima(y, order = c(5, 0, 0))
print(fit5)

fit6 <- arima(y, order = c(0, 0, 1))
print(fit6)

fit7 <- arima(y, order = c(0, 0, 2))
print(fit7)

fit8 <- arima(y, order = c(0, 0, 3))
print(fit8)

fit9 <- arima(y, order = c(0, 0, 4))
print(fit9)

fit10 <- arima(y, order = c(0, 0, 5))
print(fit10)

#####################################################
# Load the necessary library
library(forecast)

# Fit an ARMA(2,2) model to the time series data 'y'
fit <- arima(y, order=c(2,0,2))
print(fit)

# Forecast future values for 20 periods
future <- forecast(fit, h=20)

# Plot the original time series and the forecasts
plot(future)

