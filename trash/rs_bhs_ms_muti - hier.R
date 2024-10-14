# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(rstan)
library(dplyr)

# Hierarchical structure parameters (population-level parameters)
mu_alpha <- 0.5  # Mean of alpha (population-level)
sigma_alpha <- 0.2  # Standard deviation of alpha (population-level)
mu_tau <- 0.8  # Mean of tau (population-level)
sigma_tau <- 0.3  # Standard deviation of tau (population-level)

# Individual-specific parameters sampled from hierarchical distributions
alpha_ind <- rnorm(n_individuals, mean = mu_alpha, sd = sigma_alpha)
tau_ind <- rnorm(n_individuals, mean = mu_tau, sd = sigma_tau)

# Generate data for each individual
set.seed(123)
for (i in 1:n_individuals) {
  # Randomly initialize the regime for the first time point
  state[i, 1] <- sample(1:2, 1, prob = c(0.5, 0.5))
  X[i, 1] <- rnorm(1, 0, 1)  # Initial hidden state
  
  for (t in 2:n) {
    # Determine the current regime based on the previous regime and transition matrix
    state[i, t] <- sample(1:2, 1, prob = P[state[i, t-1],])
    
    # Generate X based on the current regime and individual-specific parameters
    if (state[i, t] == 1) {
      X[i, t] <- alpha_ind[i] * X[i, t-1] + rnorm(1, 0, tau_ind[i])
    } else {
      X[i, t] <- alpha_ind[i] * X[i, t-1] + rnorm(1, 0, tau_ind[i])
    }
    
    # Generate observed data Y based on X
    Y[i, t] <- beta_1 * X[i, t] + rnorm(1, 0, 0.5)
  }
}

# Create a data frame for visualization
df_list <- lapply(1:n_individuals, function(i) {
  data.frame(time = 1:n, X = X[i, ], Y = Y[i, ], state = factor(state[i, ]), individual = i)
})
df <- do.call(rbind, df_list)

# Plot actual X and regimes (example for multiple individuals)
ggplot(df, aes(x = time, y = X, group = individual, color = state)) +
  geom_line(size = 1) +
  facet_wrap(~ individual) +
  labs(title = "Actual X and Markov Regime Switching (Multiple Individuals)", 
       x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"), 
                     labels = c("Regime 1", "Regime 2"),  
                     name = "Regime") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Stan code for the regime-switching model
stan_code_regime <- "
data {
  int<lower=1> N;  // Number of time points per individual
  int<lower=1> I;  // Number of individuals
  matrix[I, N] Y;  // Observed data for each individual
}
parameters {
  matrix[I, N] X;  // Hidden states for each individual
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
  real regime_prob[I, N, 2];  // Probability of being in each regime for each individual at each time
  real log_prob[I, N, 2];     // Log-probabilities for each regime
  
  for (i in 1:I) {
    // Initial state
    log_prob[i, 1, 1] = log(pi[1]) + normal_lpdf(X[i, 1] | 0, 1);  // Regime 1
    log_prob[i, 1, 2] = log(pi[2]) + normal_lpdf(X[i, 1] | 0, 1);  // Regime 2

    for (t in 2:N) {
      log_prob[i, t, 1] = log_sum_exp(log_prob[i, t-1, 1] + log(P[1, 1]) + normal_lpdf(X[i, t] | alpha_1 * X[i, t-1], tau),
                                     log_prob[i, t-1, 2] + log(P[2, 1]) + normal_lpdf(X[i, t] | alpha_1 * X[i, t-1], tau));
      
      log_prob[i, t, 2] = log_sum_exp(log_prob[i, t-1, 1] + log(P[1, 2]) + normal_lpdf(X[i, t] | alpha_2 * X[i, t-1], tau),
                                     log_prob[i, t-1, 2] + log(P[2, 2]) + normal_lpdf(X[i, t] | alpha_2 * X[i, t-1], tau));
    }

    // Compute posterior probabilities for each regime
    for (t in 1:N) {
      real norm_const = log_sum_exp(log_prob[i, t, 1], log_prob[i, t, 2]);
      regime_prob[i, t, 1] = exp(log_prob[i, t, 1] - norm_const);
      regime_prob[i, t, 2] = exp(log_prob[i, t, 2] - norm_const);
    }
  }
}
model {
  // Priors for parameters
  alpha_1 ~ normal(0, 1);
  beta_1 ~ normal(0, 1);
  alpha_2 ~ normal(0, 1);
  beta_2 ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);
  tau ~ normal(0, 1);
  pi ~ dirichlet(rep_vector(1.0, 2));
  for (i in 1:2) {
    P[i] ~ dirichlet(rep_vector(1.0, 2));
  }

  // Likelihood for observed data
  for (i in 1:I) {
    Y[i] ~ normal(X[i], sigma);
  }
}
"

# Compile the simple AR model
stan_code_simple <- "
data {
  int<lower=1> N;  // Number of time points per individual
  int<lower=1> I;  // Number of individuals
  matrix[I, N] Y;  // Observed data for each individual
}
parameters {
  matrix[I, N] X;  // Hidden states for each individual
  real<lower=0> sigma;  // Observation noise
  real alpha;  // Autoregressive coefficient (same for all individuals)
  real<lower=0> tau;  // Process noise
}
model {
  // Priors
  alpha ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);
  tau ~ normal(0, 1);
  
  // Likelihood
  for (i in 1:I) {
    X[i, 1] ~ normal(0, 1);  // Initial state
    for (t in 2:N) {
      X[i, t] ~ normal(alpha * X[i, t-1], tau);  // Autoregressive process
    }
    Y[i] ~ normal(X[i], sigma);  // Observation equation
  }
}
"

# Prepare the data list for Stan model
stan_data <- list(N = n, I = n_individuals, Y = Y)

# Compile both models
stan_model_simple <- stan_model(model_code = stan_code_simple)
stan_model_regime <- stan_model(model_code = stan_code_regime)

# Fit both models
fit_simple <- sampling(stan_model_simple, data = stan_data, iter = 2000, chains = 4)
fit_regime_multi <- sampling(stan_model_regime, data = stan_data, iter = 2000, chains = 4)

# RMSE function
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Compute hidden states and RMSE for the simple model
X_simple <- extract(fit_simple)$X
X_simple_mean <- apply(X_simple, c(2, 3), mean)  # Take the mean
rmse_simple <- rmse(X, X_simple_mean)

# Compute hidden states and RMSE for the regime-switching model
X_regime <- extract(fit_regime_multi)$X
X_regime_mean <- apply(X_regime, c(2, 3), mean)  # Take the mean
rmse_regime <- rmse(X, X_regime_mean)

# Display RMSE comparison results
cat("RMSE (Simple Model):", rmse_simple, "\n")
cat("RMSE (Regime-Switching Model):", rmse_regime, "\n")

# Extract regime probabilities from the Stan model fit
regime_prob <- extract(fit_regime_multi)$regime_prob

# Create a data frame for plotting regime probabilities for each individual
df_prob_list <- lapply(1:n_individuals, function(i) {
  data.frame(
    time = 1:n,
    regime_1_prob = apply(regime_prob[, i, , 1], 2, mean),  # Average regime 1 probability per time for individual i
    regime_2_prob = apply(regime_prob[, i, , 2], 2, mean),  # Average regime 2 probability per time for individual i
    individual = i
  )
})

# Combine data into one data frame
df_prob <- do.call(rbind, df_prob_list)

# Plot the regime probabilities over time for each individual
ggplot(df_prob, aes(x = time)) +
  geom_line(aes(y = regime_1_prob, color = "Regime 1")) +  # Regime 1 probability
  geom_line(aes(y = regime_2_prob, color = "Regime 2")) +  # Regime 2 probability
  facet_wrap(~ individual) +  # One plot per individual
  labs(title = "Regime Probabilities Over Time for Each Individual", 
       x = "Time", y = "Probability") +
  scale_color_manual(values = c("red", "blue"), name = "Regime") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Create a list to store the data frames for each individual
df_combined_list <- list()

for (i in 1:n_individuals) {
  # Combine actual X and predicted values for each individual
  df_combined_list[[i]] <- data.frame(
    time = rep(1:n, 3),
    X = c(X[i,], X_simple_mean[i,], X_regime_mean[i,]),
    Model = rep(c("Actual X", "Simple Model", "Regime-Switching Model"), each = n),
    individual = i  # Add individual identifier
  )
}

# Combine all individuals into one data frame
df_combined_all <- do.call(rbind, df_combined_list)

# Plot each individual's comparison separately with all lines as solid
ggplot(df_combined_all, aes(x = time, y = X, color = Model)) +
  geom_line(size = 1, linetype = "solid") +  # Set all lines to be solid
  facet_wrap(~individual, ncol = 1) +  # One plot per individual, vertically stacked
  labs(title = "Comparison of Actual and Predicted X for Each Individual", 
       x = "Time", 
       y = "X") +
  scale_color_manual(values = c("red", "blue", "green")) +  # Custom colors for each line
  theme_minimal() +
  theme(legend.position = "right")