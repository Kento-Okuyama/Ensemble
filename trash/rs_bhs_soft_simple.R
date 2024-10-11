# Load necessary packages
library(ggplot2)
library(rstan)

# 1. Generate synthetic data with probabilistic regime switching
set.seed(123)
n <- 100
X <- numeric(n)
Y <- numeric(n)
X[1] <- rnorm(1, 0, 1)

# Generate data with regime switching
for (t in 2:n) {
  X[t] <- if (X[t-1] > 0) 0.5 * X[t-1] + rnorm(1, 0, 1) else -0.8 * X[t-1] + rnorm(1, 0, 1)
  Y[t] <- 1.0 * X[t] + rnorm(1, 0, 0.5)
}

# Prepare data for Stan
stan_data <- list(N = n, Y = Y)

# 2. Stan model for probabilistic regime switching
stan_code_prob <- "
data { int<lower=1> N; vector[N] Y; }
parameters { vector[N] X; real<lower=0> sigma; real alpha; real beta; real<lower=0> tau; real threshold; }
model {
  alpha ~ normal(0, 1); beta ~ normal(0, 1); sigma ~ normal(0, 1); tau ~ normal(0, 1); threshold ~ normal(0, 1);
  X[1] ~ normal(0, 1);
  for (t in 2:N) {
    real p = inv_logit(X[t-1] - threshold);
    X[t] ~ normal((p * alpha + (1 - p) * beta) * X[t-1], tau);
  }
  Y ~ normal(X, sigma);
}
"

# 3. Stan model for simple linear model (ignoring regime switching)
stan_code_simple <- "
data { int<lower=1> N; vector[N] Y; }
parameters { vector[N] X; real<lower=0> sigma; real alpha; real<lower=0> tau; }
model {
  alpha ~ normal(0, 1); sigma ~ normal(0, 1); tau ~ normal(0, 1);
  X[1] ~ normal(0, 1);
  for (t in 2:N) X[t] ~ normal(alpha * X[t-1], tau);
  Y ~ normal(X, sigma);
}
"

# Compile and sample from the models
fit_prob <- sampling(stan_model(model_code = stan_code_prob), data = stan_data, iter = 2000, chains = 4)
fit_simple <- sampling(stan_model(model_code = stan_code_simple), data = stan_data, iter = 2000, chains = 4)

# 4. RMSE calculation function
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))

# Get the predicted X values for each model
X_est_prob <- apply(extract(fit_prob)$X, 2, mean)
X_est_simple <- apply(extract(fit_simple)$X, 2, mean)

# Print RMSE values
cat("RMSE (Probabilistic Model):", rmse(X, X_est_prob), "\n")
cat("RMSE (Simple Model):", rmse(X, X_est_simple), "\n")

# 5. Plot actual vs predicted values from both models
df <- data.frame(time = 1:n, X = X, X_est_prob = X_est_prob, X_est_simple = X_est_simple)

ggplot(df, aes(x = time)) +
  geom_line(aes(y = X, color = "Actual X"), size = 1) +
  geom_line(aes(y = X_est_prob, color = "Probabilistic Model"), size = 0.7) +
  geom_line(aes(y = X_est_simple, color = "Simple Model"), size = 0.7, linetype = "dashed") +
  labs(title = "Comparison of Actual and Predicted X", x = "Time", y = "X") +
  theme_minimal()
