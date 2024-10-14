# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(rstan)

# 1. Generate synthetic data with deterministic regime switching
set.seed(123)
n <- 30  # Number of time points
X <- numeric(n)  # Hidden state
Y <- numeric(n)  # Observed data
state <- numeric(n)  # Regime (1 for regime 1, 2 for regime 2)
threshold <- 0  # Threshold to determine regime
state[1] <- 1
X[1] <- rnorm(1, 0, 1)  # Initial state

# Generate data with regime switching
for (t in 2:n) {
  if (X[t-1] > threshold) {
    state[t] <- 1
    X[t] <- 0.5 * X[t-1] + rnorm(1, 0, 1)
  } else {
    state[t] <- 2
    X[t] <- -0.8 * X[t-1] + rnorm(1, 0, 1)
  }
  Y[t] <- 1.0 * X[t] + rnorm(1, 0, 0.5)
}

# 2. Prepare data for Stan models
stan_data <- list(N = n, Y = Y)

# 3. Stan model for deterministic regime switching
stan_code_regime <- "
data { 
  int<lower=1> N;
  vector[N] Y;
}
parameters { 
  vector[N] X; 
  real<lower=0> sigma;
  real alpha;
  real beta;
  real<lower=0> tau;
  real threshold;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  tau ~ normal(0, 1);
  threshold ~ normal(0, 1);
  X[1] ~ normal(0, 1);
  
  for (t in 2:N) {
    if (X[t-1] > threshold) {
      X[t] ~ normal(alpha * X[t-1], tau);
    } else {
      X[t] ~ normal(beta * X[t-1], tau);
    }
  }
  Y ~ normal(X, sigma);
}
"

# 4. Stan model ignoring regime switching
stan_code_simple <- "
data { 
  int<lower=1> N;
  vector[N] Y;
}
parameters { 
  vector[N] X; 
  real<lower=0> sigma;
  real alpha;
  real<lower=0> tau;
}
model {
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  tau ~ normal(0, 1);
  X[1] ~ normal(0, 1);
  
  for (t in 2:N) {
    X[t] ~ normal(alpha * X[t-1], tau);
  }
  Y ~ normal(X, sigma);
}
"

# Compile the Stan models
stan_model_regime <- stan_model(model_code = stan_code_regime)
stan_model_simple <- stan_model(model_code = stan_code_simple)

# Fit the models
fit_regime <- sampling(stan_model_regime, data = stan_data, iter = 2000, chains = 4)
fit_simple <- sampling(stan_model_simple, data = stan_data, iter = 2000, chains = 4)

# Extract and summarize the parameters
alpha_mean <- mean(extract(fit_regime)$alpha)
beta_mean <- mean(extract(fit_regime)$beta)
threshold_mean <- mean(extract(fit_regime)$threshold)
alpha_simple_mean <- mean(extract(fit_simple)$alpha)

# Print parameter summaries
cat("Alpha (Regime-Switching Model):", alpha_mean, "\n")
cat("Beta (Regime-Switching Model):", beta_mean, "\n")
cat("Threshold (Regime-Switching Model):", threshold_mean, "\n")
cat("Alpha (Simple Model):", alpha_simple_mean, "\n")

# 5. RMSE function
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Extract the estimated hidden states
hidden_states_regime <- extract(fit_regime)$X
hidden_states_simple <- extract(fit_simple)$X

# Create data frames for plotting
df_hidden_regime <- data.frame(time = 1:n, X_estimated = apply(hidden_states_regime, 2, mean))
df_hidden_simple <- data.frame(time = 1:n, X_estimated = apply(hidden_states_simple, 2, mean))
df_hidden_regime$state <- ifelse(df_hidden_regime$X_estimated > threshold_mean, 1, 2)

# Original data frame
df <- data.frame(time = 1:n, X = X, Y = Y, state = factor(state))

# 6. Plot comparison
plot_X_actual <- ggplot(df, aes(x = time, y = X, color = state, group = 1)) +
  geom_line(size = 1) +
  labs(title = "Actual X (Regime Switching)", x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")

plot_X_regime <- ggplot(df_hidden_regime, aes(x = time, y = X_estimated, color = factor(state), group = 1)) +
  geom_line(size = 1) +
  labs(title = "Predicted X (Regime-Switching)", x = "Time", y = "X_estimated") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")

plot_X_simple <- ggplot(df_hidden_simple, aes(x = time, y = X_estimated)) + 
  geom_line(size = 1, color = "green") + 
  labs(title = "Predicted X (Simple Model)", x = "Time", y = "X_estimated")

# Arrange plots in a grid
grid.arrange(plot_X_actual, plot_X_regime, plot_X_simple, ncol = 1)

# Calculate RMSE values
rmse_regime <- rmse(X, df_hidden_regime$X_estimated)
rmse_simple <- rmse(X, df_hidden_simple$X_estimated)

# Print RMSE
cat("RMSE (Regime-Switching Model):", rmse_regime, "\n")
cat("RMSE (Simple Model):", rmse_simple, "\n")

# Combined plot
ggplot() +
  geom_line(data = df, aes(x = time, y = X, color = "Actual X"), size = 1) +
  geom_line(data = df_hidden_regime, aes(x = time, y = X_estimated, color = "Regime-Switching Model"), size = 0.7) +
  geom_line(data = df_hidden_simple, aes(x = time, y = X_estimated, color = "Simple Model"), size = 0.7, linetype = "dashed") +
  labs(title = "Comparison of Actual and Predicted X", x = "Time", y = "X") +
  theme_minimal()

