library(ggplot2)
library(gridExtra)
library(rstan)

# 1. Generate synthetic data with deterministic regime switching based on X_t
set.seed(123)
n <- 100  # Number of time points
X <- numeric(n)  # Hidden state
Y <- numeric(n)  # Observed data
threshold <- 0  # Threshold to determine regime
state <- numeric(n)  # Hidden state (1 for regime 1, 2 for regime 2)
state[1] <- 1
X[1] <- rnorm(1, 0, 1)  # Initial state

# Generate data with regime switching based on X_t
for (t in 2:n) {
  if (X[t-1] > threshold) {
    state[t] <- 1  # Regime 1
    X[t] <- 0.5 * X[t-1] + rnorm(1, 0, 1)  # Dynamics in regime 1
  } else {
    state[t] <- 2  # Regime 2
    X[t] <- -0.8 * X[t-1] + rnorm(1, 0, 1)  # Dynamics in regime 2
  }
  Y[t] <- 1.0 * X[t] + rnorm(1, 0, 0.5)  # Observation noise
}

# 2. Prepare data for Stan models
stan_data <- list(N = n, Y = Y)

# 3. Stan model for deterministic regime switching with threshold estimation
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

# 4. Stan model ignoring regime switching (simple linear model)
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

# Extract 'alpha', 'beta', and 'threshold' from fit_regime
alpha_values <- extract(fit_regime)$alpha
beta_values <- extract(fit_regime)$beta
threshold_values <- extract(fit_regime)$threshold

# Display the mean values of each parameter
alpha_mean <- mean(alpha_values)
beta_mean <- mean(beta_values)
threshold_mean <- mean(threshold_values)

cat("Alpha (mean):", alpha_mean, "\n")
cat("Beta (mean):", beta_mean, "\n")
cat("Threshold (mean):", threshold_mean, "\n")

# Extract 'alpha' from fit_simple
alpha_simple_values <- extract(fit_simple)$alpha

# Calculate the mean of alpha from fit_simple
alpha_simple_mean <- mean(alpha_simple_values)

cat("Alpha from fit_simple (mean):", alpha_simple_mean, "\n")

# Extract the estimated hidden states
hidden_states_regime <- extract(fit_regime)$X
hidden_states_simple <- extract(fit_simple)$X
threshold_estimated <- extract(fit_regime)$threshold

# Create data frames for plotting
df_hidden_regime <- data.frame(time = 1:n, X_estimated = apply(hidden_states_regime, 2, mean))
df_hidden_simple <- data.frame(time = 1:n, X_estimated = apply(hidden_states_simple, 2, mean))
df_hidden_regime$state <- ifelse(df_hidden_regime$X_estimated > mean(threshold_estimated), 1, 2)

# Original data frame
df <- data.frame(time = 1:n, X = X, Y = Y, state = factor(state))

# 5. Plot comparison: actual X, regime-switching model, and simple model
plot_X_actual <- ggplot(df, aes(x = time, y = X, color = state, group = 1)) +
  geom_line(size = 1) +
  labs(title = "Time vs Actual X (Regime Switching)", x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")

plot_X_regime <- ggplot(df_hidden_regime, aes(x = time, y = X_estimated, color = factor(state), group = 1)) +
  geom_line(size = 1) +
  labs(title = "Time vs Predicted X (Regime Switching)", x = "Time", y = "X_estimated") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")

plot_X_simple <- ggplot(df_hidden_simple, aes(x = time, y = X_estimated)) + 
  geom_line(size = 1, color = "green") + 
  labs(title = "Time vs Predicted X (Simple Model)", x = "Time", y = "X_estimated")

# Arrange all plots
grid.arrange(plot_X_actual, plot_X_regime, plot_X_simple, ncol = 1)

# Calculate RMSE for each model
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# RMSE for the regime-switching model
rmse_regime <- rmse(X, df_hidden_regime$X_estimated)

# RMSE for the simple linear model
rmse_simple <- rmse(X, df_hidden_simple$X_estimated)

# Print RMSE values
cat("RMSE (Regime-Switching Model):", rmse_regime, "\n")
cat("RMSE (Simple Linear Model):", rmse_simple, "\n")

# Combined plot with actual X and predicted X from both models
ggplot() +
  geom_line(data = df, aes(x = time, y = X, color = "Actual X"), size = 1) +  # Actual X
  geom_line(data = df_hidden_regime, aes(x = time, y = X_estimated, color = "Regime-Switching Model"), size = 0.7) +  # Regime-Switching Model
  geom_line(data = df_hidden_simple, aes(x = time, y = X_estimated, color = "Simple Model"), size = 0.7, linetype = "dashed") +  # Simple Model (dashed line)
  labs(title = "Comparison of Actual and Predicted X", x = "Time", y = "X") +
  theme_minimal()



