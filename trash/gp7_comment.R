# Clear the workspace
rm(list = ls())

# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(rstan)
library(dplyr)

# 1. Generate Markov regime-switching data for each individual
set.seed(123)
n <- 30  # Number of time points for each individual
n_individuals <- 5  # Number of individuals

# Create matrices to store data
X <- matrix(0, nrow = n_individuals, ncol = n)  # Hidden state (X)
Y <- matrix(0, nrow = n_individuals, ncol = n)  # Observed data (Y)
state <- matrix(0, nrow = n_individuals, ncol = n)  # Regime state (1 or 2)

# Parameters for regime 1 and regime 2
phi_1 <- 0.8  # Autoregressive coefficient for regime 1
mu_1 <- 0.5

# Transition matrix between regimes
P <- matrix(c(0.8, 0.2,  # P(1->1), P(1->2)
              0.3, 0.7), # P(2->1), P(2->2)
            nrow = 2, byrow = TRUE)

# Generate data for each individual
for (i in 1:n_individuals) {
  state[i, 1] <- sample(1:2, 1, prob = c(0.5, 0.5))  # Initial regime state
  X[i, 1] <- rnorm(1, 0, 1)  # Initial hidden state
  
  for (t in 2:n) {
    # Sample next regime state based on the previous state
    state[i, t] <- sample(1:2, 1, prob = P[state[i, t-1], ])
    
    # Generate the hidden state X based on the regime
    if (state[i, t] == 1) {
      X[i, t] <- phi_1 * (X[i, t-1] - mu_1) + rnorm(1, 0, 1)  # AR(1) process for regime 1
    } else {
      X[i, t] <- X[i, t-1] + rnorm(1, 0, 1)  # Random walk for regime 2
    }
    
    # Observed data Y is noisy version of the hidden state X
    Y[i, t] <- X[i, t] + rnorm(1, 0, 0.5)
  }
}

# Create a data frame for visualization
df_list <- lapply(1:n_individuals, function(i) {
  data.frame(time = 1:n, X = X[i, ], Y = Y[i, ], 
             state = factor(state[i, ]), individual = i)
})
df <- do.call(rbind, df_list)

# Visualize actual X and regime-switching for each individual
ggplot(df, aes(x = time, y = X, group = individual, color = state)) +
  geom_line(size = 1) +
  facet_wrap(~ individual) +
  labs(title = "Actual X and Markov Regime Switching (Multiple Individuals)",
       x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"),
                     labels = c("Regime 1", "Regime 2"), name = "Regime") +
  theme_minimal() +
  theme(legend.position = "bottom")

# 2. Define and compile Stan models
# AR(1) Hidden Markov Model
stan_model_AR1 <- stan_model(model_code = "
  data {
    int<lower=1> N_people;    // Number of individuals
    int<lower=1> N_timepoints; // Number of time points
    vector[N_timepoints] y[N_people];  // Observed data
  }
  parameters {
    real phi1;  // AR(1) coefficient for latent state
    real mu_x;  // Mean of latent state
    real<lower=0> sigma_x;  // Standard deviation for latent state process
    real<lower=0> sigma_y;  // Observation noise
    vector[N_timepoints] x[N_people];  // Latent states
  }
  model {
    // Priors
    phi1 ~ normal(0, 1);  // AR(1) coefficient prior
    mu_x ~ normal(0, 10);  // Latent state mean prior
    sigma_x ~ cauchy(0, 2.5);  // Latent state stddev prior
    sigma_y ~ cauchy(0, 2.5);  // Observation noise prior

    // Latent state model (AR(1) process)
    for (j in 1:N_people) {
      x[j][1] ~ normal(mu_x, sigma_x);  // Initial latent state
      for (n in 2:N_timepoints) {
        x[j][n] ~ normal(mu_x + phi1 * (x[j][n-1] - mu_x), sigma_x);  // AR(1) transition for latent state
      }
      // Observation model
      y[j] ~ normal(x[j], sigma_y);  // y depends on latent state x
    }
  }
  generated quantities {
    vector[N_timepoints] y_hat[N_people];  // Predicted values for y
    vector[N_timepoints] log_lik[N_people];  // Log-likelihood for each time point
    for (j in 1:N_people) {
      for (n in 1:N_timepoints) {
        y_hat[j][n] = normal_rng(x[j][n], sigma_y);  // Simulate y based on latent x
        log_lik[j][n] = normal_lpdf(y[j][n] | x[j][n], sigma_y);  // Log likelihood
      }
    }
  }
")

# Random Walk Hidden Markov Model (HMM)
stan_model_RW <- stan_model(model_code = "
  data {
    int<lower=1> N_people;    // Number of individuals
    int<lower=1> N_timepoints; // Number of time points
    vector[N_timepoints] y[N_people];  // Observed data
  }
  parameters {
    real mu_x;  // Mean of random walk increments
    real<lower=0> sigma_x;  // Standard deviation of random walk increments
    real<lower=0> sigma_y;  // Observation noise
    vector[N_timepoints] x[N_people];  // Latent states (the random walk states)
  }
  model {
    // Priors
    mu_x ~ normal(0, 10);  // Prior for mean of random walk increments
    sigma_x ~ cauchy(0, 2.5);  // Prior for random walk stddev
    sigma_y ~ cauchy(0, 2.5);  // Prior for observation noise

    // Latent state model (random walk process)
    for (j in 1:N_people) {
      x[j][1] ~ normal(0, sigma_x);  // Initial state of the random walk
      for (n in 2:N_timepoints) {
        x[j][n] ~ normal(x[j][n-1] + mu_x, sigma_x);  // Random walk step
      }

      // Observation model
      y[j] ~ normal(x[j], sigma_y);  // y depends on latent state x
    }
  }
  generated quantities {
    vector[N_timepoints] y_hat[N_people];  // Predicted values for y
    vector[N_timepoints] log_lik[N_people];  // Log-likelihood for each time point
    for (j in 1:N_people) {
      for (n in 1:N_timepoints) {
        y_hat[j][n] = normal_rng(x[j][n], sigma_y);  // Simulate y based on latent x
        log_lik[j][n] = normal_lpdf(y[j][n] | x[j][n], sigma_y);  // Log likelihood
      }
    }
  }
")

# 3. Prepare data for Stan model fitting and fit models
fit_RW <- sampling(stan_model_RW, data = list(N_people = n_individuals, N_timepoints = n, y = Y), iter = 1000)
fit_AR1 <- sampling(stan_model_AR1, data = list(N_people = n_individuals, N_timepoints = n, y = Y), iter = 1000)

# Extract predictions from AR(1) and Random Walk models
y_hat_RW <- apply(extract(fit_RW, pars = "y_hat")$y_hat, c(2, 3), mean)
y_hat_AR1 <- apply(extract(fit_AR1, pars = "y_hat")$y_hat, c(2, 3), mean)

# Create data list for hierarchical stacking
data_list <- list(N_people = n_individuals, N_timepoints = n, y = Y, y_hat_RW = y_hat_RW, y_hat_AR1 = y_hat_AR1)

# 4. Define and compile Gaussian process ensemble model
stan_model_gp_ensemble <- stan_model(model_code = "
  data {
    int<lower=1> N_people;
    int<lower=1> N_timepoints;
    vector[N_timepoints] y[N_people];
    vector[N_timepoints] y_hat_RW[N_people];
    vector[N_timepoints] y_hat_AR1[N_people];
  }
  parameters {
    real<lower=0> eta;
    real<lower=0> rho;
    matrix[N_timepoints, 2] w_raw[N_people];
    real<lower=0> sigma;
  }
  transformed parameters {
    matrix[N_timepoints, 2] w[N_people];
    {
      matrix[N_timepoints, N_timepoints] K;
      matrix[N_timepoints, N_timepoints] L_K;
      for (i in 1:N_timepoints) {
        for (j in 1:N_timepoints) {
          if (i == j) {
            K[i, j] = eta^2 + 1e-10;
          } else {
            K[i, j] = eta^2 * exp(-square(i - j) / (2 * rho^2));
          }
        }
      }
      L_K = cholesky_decompose(K);
      for (j in 1:N_people) {
        matrix[N_timepoints, 2] w_temp = L_K * w_raw[j];
        for (t in 1:N_timepoints) {
          w[j, t] = to_row_vector(softmax(to_vector(w_temp[t])));
        }
      }
    }
  }
  model {
    eta ~ normal(0, 1);
    rho ~ normal(0, 1);
    for (j in 1:N_people) {
      for (k in 1:2) {
        w_raw[j, , k] ~ normal(0, 1);
      }
    }
    sigma ~ cauchy(0, 2.5);
    for (j in 1:N_people) {
      for (t in 1:N_timepoints) {
        real y_pred = w[j, t, 1] * y_hat_RW[j, t] + w[j, t, 2] * y_hat_AR1[j, t];
        y[j, t] ~ normal(y_pred, sigma);
      }
    }
  }
  generated quantities {
    vector[N_timepoints] log_lik[N_people];
    vector[N_timepoints] y_hat_ens[N_people];
    for (j in 1:N_people) {
      for (t in 1:N_timepoints) {
        real y_pred = w[j, t, 1] * y_hat_RW[j, t] + w[j, t, 2] * y_hat_AR1[j, t];
        log_lik[j][t] = normal_lpdf(y[j][t] | y_pred, sigma);
        y_hat_ens[j][t] = normal_rng(y_pred, sigma);
      }
    }
  }
")

# 5. Fit the hierarchical stacking model
fit_gp_ensemble <- sampling(stan_model_gp_ensemble, data = data_list, iter = 1000, warmup = 500, chains = 4)

# 6. Extract predictions from the ensemble model
y_hat_ensemble <- apply(extract(fit_gp_ensemble, pars = "y_hat_ens")$y_hat_ens, c(2, 3), mean)
w_ensemble <- apply(extract(fit_gp_ensemble, pars = "w")$w[,,,1], c(2, 3), mean)

# Function to calculate Mean Squared Error (MSE)
calculate_mse_time <- function(y_hat, y_true, start_point = 1) {
  mse_values <- sapply(start_point:dim(y_hat)[2], function(t) {
    mse_per_person <- sapply(1:nrow(y_true), function(i) (y_hat[i, t] - y_true[i, t])^2)
    return(mean(mse_per_person))
  })
  return(mse_values)
}

# Calculate MSE over time for the ensemble model
mse_time_ensemble <- calculate_mse_time(y_hat_ensemble, Y)

# Prepare data for plotting MSE over time
mse_time_data <- data.frame(
  Time = 1:n,
  MSE = mse_time_ensemble,
  Model = "Ensemble"
)

# Plot MSE over time for the ensemble model
ggplot(mse_time_data, aes(x = Time, y = MSE, color = Model, group = Model)) +
  geom_line(size = 1.2) +
  scale_y_log10() +  # Logarithmic scale for better visualization
  labs(title = "MSE over Time", x = "Time", y = "Log(MSE)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("Ensemble" = "purple"))


# Prepare data for plotting MSE over time
w_time_data <- data.frame(
  Time = 1:n,
  w = w_ensemble[1,],
  Model = "Ensemble"
)

# Plot MSE over time for the ensemble model
ggplot(w_time_data, aes(x = Time, y = w, color = Model, group = Model)) +
  geom_line(size = 1.2) +
  scale_y_log10() +  # Logarithmic scale for better visualization
  labs(title = "w over Time", x = "Time", y = "Log(MSE)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("Ensemble" = "purple"))


# Directly extract the weights for all individuals from the Stan fit
w_ensemble_all <- extract(fit_gp_ensemble, pars = "w")$w

# w_ensemble_all has dimensions [iterations, N_people, N_timepoints, 2], where the last dimension refers to the weight for each model (RW or AR1)
# We'll take the mean across iterations for the first weight (index = 1) across all time points and individuals

# Calculate mean weights over iterations for the first model (RW) for all individuals and time points
w_mean_ensemble_all <- apply(w_ensemble_all[, , , 1], c(2, 3), mean)  # [N_people, N_timepoints]

# Prepare data for plotting weights over time for all individuals
w_time_data_all <- data.frame(
  Time = rep(1:n, times = n_individuals),               # Repeat time points for each individual
  w = as.vector(t(w_mean_ensemble_all)),                # Flatten the matrix (weights) for all individuals
  Individual = factor(rep(1:n_individuals, each = n))   # Add an identifier for each individual
)

# Plot weights over time for all individuals
ggplot(w_time_data_all, aes(x = Time, y = w, color = Individual, group = Individual)) +
  geom_line(size = 0.8) +
  labs(title = "Weights over Time for All Individuals", x = "Time", y = "Weight") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = rainbow(n_individuals))  # Assign different colors for each individual
