# AR-1 Model Simulation

# Set parameters
phi <- 0.5 # Autoregressive coefficient
sigma <- 1e-4 # Standard deviation of noise
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

