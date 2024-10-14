# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(rstan)

# 1. Generate synthetic data with Markov regime switching
set.seed(123)
n <- 100  # Number of time points
X <- numeric(n)  # Hidden state
Y <- numeric(n)  # Observed data
state <- numeric(n)  # Regime (1 for regime 1, 2 for regime 2)

# Parameters for the two regimes
alpha_1 <- 0.8  # Autoregression coefficient for regime 1
beta_1 <- 1.0   # Slope for regime 1
alpha_2 <- -1.0  # Autoregression coefficient for regime 2
beta_2 <- 1.0   # Slope for regime 2

# Transition matrix for regimes
P <- matrix(c(0.8, 0.2,  # P(1->1), P(1->2)
              0.3, 0.7), # P(2->1), P(2->2)
            nrow = 2, byrow = TRUE)

# Initial regime probabilities
state[1] <- sample(1:2, 1, prob = c(0.5, 0.5))  # Random initial regime
X[1] <- rnorm(1, 0, 1)  # Initial state

# Generate data with Markov regime switching
for (t in 2:n) {
  # Determine current regime based on previous regime and transition matrix
  state[t] <- sample(1:2, 1, prob = P[state[t-1],])
  
  # Generate X based on the current regime
  if (state[t] == 1) {
    X[t] <- alpha_1 * X[t-1] + rnorm(1, 0, 1)
  } else {
    X[t] <- alpha_2 * X[t-1] + rnorm(1, 0, 1)
  }
  
  # Generate observed data Y based on X
  Y[t] <- beta_1 * X[t] + rnorm(1, 0, 0.5)
}

# Dataframe for visualization
df <- data.frame(time = 1:n, X = X, Y = Y, state = factor(state))

# Plot actual X and regimes
plot_X_actual <- ggplot(df, aes(x = time, y = X, color = state, group = 1)) +
  geom_line(size = 1) +
  labs(title = "Actual X with Markov Switching Regimes", x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")
print(plot_X_actual)

# 2. Prepare data for Stan models
stan_data <- list(N = n, Y = Y)

# 3. Stan model for deterministic regime switching
stan_code_regime <- "
data {
  int<lower=1> N;  // Number of time points
  vector[N] Y;     // Observed data
}

parameters {
  vector[N] X;  // Hidden state
  real<lower=0> sigma;  // Observation noise
  real alpha_1;  // Autoregressive coefficient for regime 1
  real beta_1;   // Slope for regime 1
  real alpha_2;  // Autoregressive coefficient for regime 2
  real beta_2;   // Slope for regime 2
  real<lower=0> tau;  // Process noise
  simplex[2] pi;  // Initial state probability distribution
  simplex[2] P[2];  // Transition matrix between regimes
}

transformed parameters {
  matrix[N, 2] regime_prob;  // Probability of being in each regime at each time
  real log_prob[N, 2];       // Log-probability for each regime at each time

  // Initial state
  log_prob[1, 1] = log(pi[1]) + normal_lpdf(X[1] | 0, 1);  // Regime 1
  log_prob[1, 2] = log(pi[2]) + normal_lpdf(X[1] | 0, 1);  // Regime 2
  
  for (t in 2:N) {
    log_prob[t, 1] = log_sum_exp(log_prob[t-1, 1] + log(P[1, 1]) + normal_lpdf(X[t] | alpha_1 * X[t-1], tau),
                                 log_prob[t-1, 2] + log(P[2, 1]) + normal_lpdf(X[t] | alpha_1 * X[t-1], tau));
    
    log_prob[t, 2] = log_sum_exp(log_prob[t-1, 1] + log(P[1, 2]) + normal_lpdf(X[t] | alpha_2 * X[t-1], tau),
                                 log_prob[t-1, 2] + log(P[2, 2]) + normal_lpdf(X[t] | alpha_2 * X[t-1], tau));
  }
  
  // Calculate posterior probabilities for each regime
  for (t in 1:N) {
    real norm_const = log_sum_exp(log_prob[t, 1], log_prob[t, 2]);
    regime_prob[t, 1] = exp(log_prob[t, 1] - norm_const);  // Probability for regime 1
    regime_prob[t, 2] = exp(log_prob[t, 2] - norm_const);  // Probability for regime 2
  }
}

model {
  // Priors for parameters
  alpha_1 ~ normal(0, 1);
  beta_1 ~ normal(0, 1);
  alpha_2 ~ normal(0, 1);
  beta_2 ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);  // Log-normal for strictly positive values;
  tau ~ normal(0, 1);
  pi ~ dirichlet(rep_vector(1.0, 2));
  for (i in 1:2) {
    P[i] ~ dirichlet(rep_vector(1.0, 2));
  }

  // Likelihood for the observed data
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

# Extract posterior probabilities from Stan
regime_prob <- extract(fit_regime)$regime_prob

# Extract and summarize the parameters
alpha_mean <- mean(extract(fit_regime)$alpha_1)
beta_mean <- mean(extract(fit_regime)$beta_1)
threshold_mean <- mean(extract(fit_regime)$alpha_2)
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

# Add regime probabilities to the dataframe
df_hidden_regime$regime_1_prob <- apply(regime_prob[,,1], 2, mean)  # Probability of regime 1
df_hidden_regime$regime_2_prob <- apply(regime_prob[,,2], 2, mean)  # Probability of regime 2

# Original data frame
df <- data.frame(time = 1:n, X = X, Y = Y, state = factor(state))

# 6. Plot comparison
plot_X_actual <- ggplot(df, aes(x = time, y = X, color = state, group = 1)) +
  geom_line(size = 1) +
  labs(title = "Actual X (Regime Switching)", x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"), name = "Regime") +
  theme(legend.position = "none")

plot_X_regime <- ggplot(df_hidden_regime, aes(x = time, y = X_estimated, group = 1)) +
  geom_line(size = 1, color = "green") +
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

# Visualize regime probabilities
ggplot(df_hidden_regime, aes(x = time)) +
  geom_line(aes(y = regime_1_prob, color = "Regime 1")) +
  geom_line(aes(y = regime_2_prob, color = "Regime 2")) +
  labs(title = "Regime Probabilities", x = "Time", y = "Probability")
