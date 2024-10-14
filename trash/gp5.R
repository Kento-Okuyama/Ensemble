rm(list = ls())

# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(rstan)
library(dplyr)

# 1. Generate Markov regime-switching data (for each individual)
set.seed(123)
n <- 30  # Number of time points for each individual
n_individuals <- 5  # Number of individuals

# Create matrices to store data
X <- matrix(0, nrow = n_individuals, ncol = n)  # Hidden state
Y <- matrix(0, nrow = n_individuals, ncol = n)  # Observed data
state <- matrix(0, nrow = n_individuals, ncol = n)  # Regime (1 or 2)

# Parameters for regime 1 and regime 2
alpha_1 <- 1.2  # Autoregressive coefficient for regime 1
beta_1 <- 1.0   # Slope for regime 1

# Transition matrix between regimes
P <- matrix(c(0.8, 0.2,  # P(1->1), P(1->2)
              0.3, 0.7), # P(2->1), P(2->2)
            nrow = 2, byrow = TRUE)

# Generate data for each individual
for (i in 1:n_individuals) {
  state[i, 1] <- sample(1:2, 1, prob = c(0.5, 0.5))
  X[i, 1] <- rnorm(1, 0, 1)  # Initial hidden state
  
  for (t in 2:n) {
    state[i, t] <- sample(1:2, 1, prob = P[state[i, t-1], ])
    
    if (state[i, t] == 1) {
      X[i, t] <- alpha_1 * X[i, t-1] + rnorm(1, 0, 1)
    } else {
      X[i, t] <- X[i, t-1] + rnorm(1, 0, 1)  # Random walk for regime 2
    }
    
    Y[i, t] <- beta_1 * X[i, t] + rnorm(1, 0, 0.5)
  }
}

# Create a data frame for visualization
df_list <- lapply(1:n_individuals, function(i) {
  data.frame(time = 1:n, X = X[i, ], Y = Y[i, ], 
             state = factor(state[i, ]), individual = i)
})
df <- do.call(rbind, df_list)

# Visualize actual X and regime-switching (for each individual)
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
# AR(0) model
stan_model_AR0 <- stan_model(model_code = "
  data {
    int<lower=1> N_people;
    int<lower=1> N_timepoints;
    vector[N_timepoints] y[N_people];
  }
  parameters {
    real mu;
    real<lower=0> sigma;
  }
  model {
    mu ~ normal(0, 10);
    sigma ~ cauchy(0, 2.5);
    for (j in 1:N_people) {
      for (n in 1:N_timepoints) {
        y[j][n] ~ normal(mu, sigma);
      }
    }
  }
  generated quantities {
    vector[N_timepoints] y_hat[N_people];
    vector[N_timepoints] log_lik[N_people];
    for (j in 1:N_people) {
      for (n in 1:N_timepoints) {
        y_hat[j][n] = normal_rng(mu, sigma);
        log_lik[j][n] = normal_lpdf(y[j][n] | mu, sigma);
      }
    }
  }
")

# AR(1) model
stan_model_AR1 <- stan_model(model_code = "
  data {
    int<lower=1> N_people;
    int<lower=1> N_timepoints;
    vector[N_timepoints] y[N_people];
  }
  parameters {
    real phi1;
    real<lower=0> sigma;
    real mu;
  }
  model {
    phi1 ~ normal(0, 1);
    sigma ~ cauchy(0, 2.5);
    mu ~ normal(0, 10);
    for (j in 1:N_people) {
      for (n in 2:N_timepoints) {
        y[j][n] ~ normal(mu + phi1 * (y[j][n-1] - mu), sigma);
      }
    }
  }
  generated quantities {
    vector[N_timepoints] log_lik[N_people];
    vector[N_timepoints] y_hat[N_people];
    for (j in 1:N_people) {
      log_lik[j][1] = 0;
      y_hat[j][1] = y[j][1];
      for (n in 2:N_timepoints) {
        log_lik[j][n] = normal_lpdf(y[j][n] | mu + phi1 * (y[j][n-1] - mu), sigma);
        y_hat[j][n] = normal_rng(mu + phi1 * (y[j][n-1] - mu), sigma);
      }
    }
  }
")

# 3. Prepare data for Stan model fitting and fit models
fit_AR0 <- sampling(stan_model_AR0, data = list(N_people = n_individuals, N_timepoints = n, y = Y), iter = 1000)
fit_AR1 <- sampling(stan_model_AR1, data = list(N_people = n_individuals, N_timepoints = n, y = Y), iter = 1000)

# Extract predictions from AR(0) and AR(1) models
y_hat_AR0 <- apply(extract(fit_AR0, pars = "y_hat")$y_hat, c(2, 3), mean)
y_hat_AR1 <- apply(extract(fit_AR1, pars = "y_hat")$y_hat, c(2, 3), mean)

# Create data list for hierarchical stacking
data_list <- list(N_people = n_individuals, N_timepoints = n, y = Y, y_hat_AR0 = y_hat_AR0, y_hat_AR1 = y_hat_AR1)

# 4. Define and compile Gaussian process ensemble model
stan_model_gp_ensemble <- stan_model(model_code = "
  data {
    int<lower=1> N_people;
    int<lower=1> N_timepoints;
    vector[N_timepoints] y[N_people];
    vector[N_timepoints] y_hat_AR0[N_people];
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
        real y_pred = w[j, t, 1] * y_hat_AR0[j, t] + w[j, t, 2] * y_hat_AR1[j, t];
        y[j, t] ~ normal(y_pred, sigma);
      }
    }
  }
  generated quantities {
    vector[N_timepoints] log_lik[N_people];
    vector[N_timepoints] y_hat_ens[N_people];
    for (j in 1:N_people) {
      for (t in 1:N_timepoints) {
        real y_pred = w[j, t, 1] * y_hat_AR0[j, t] + w[j, t, 2] * y_hat_AR1[j, t];
        log_lik[j][t] = normal_lpdf(y[j][t] | y_pred, sigma);
        y_hat_ens[j][t] = normal_rng(y_pred, sigma);
      }
    }
  }
")

# 5. Fit the hierarchical stacking model
fit_gp_ensemble <- sampling(stan_model_gp_ensemble, data = data_list, iter = 2000, warmup = 1000, chains = 4)

# 6. Extract predictions from the ensemble model
y_hat_ensemble <- apply(extract(fit_gp_ensemble, pars = "y_hat_ens")$y_hat_ens, c(2, 3), mean)

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
