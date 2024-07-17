##########################################
# Data Generation and Stan Model Setup   #
##########################################

# Load required libraries
library(forecast)
library(rstan)
library(loo)

# Define parameters for data generation
n_people <- 10
n_burnins <- 25
n_timepoints <- 50

# Initialize list to store time series data for each person
time_series_list <- list()

# Generate AR(2) time series data for each person
set.seed(123) 
for (i in 1:n_people) {
  ar <- c(0.5, -0.3)  # Autoregressive coefficients
  mu <- 0             # Mean of the process
  sigma <- 1          # Standard deviation of the process
  e <- rnorm(n_burnins + n_timepoints, mean = mu, sd = sigma) # White noise
  data <- numeric(n_burnins + n_timepoints) 
  data[1:2] <- e[1:2]                        
  
  # Generate AR(2) process
  for (t in 3:(n_burnins + n_timepoints)) {
    data[t] <- mu + ar[1] * (data[t - 1] - mu) + ar[2] * (data[t - 2] - mu) + e[t]  
  }
  
  time_series_list[[i]] <- data[(n_burnins + 1):(n_burnins + n_timepoints)]
}

# Visualize the generated time series data (now AR(2))
plot(1:n_timepoints, type = 'n', 
     xlim = c(1, n_timepoints), ylim = range(unlist(time_series_list)), 
     main = "AR(2) Time Series for All Individuals", xlab = "Time", ylab = "Value")
colors <- rainbow(n_people)
for (i in 1:n_people) {
  lines(1:n_timepoints, time_series_list[[i]], col = colors[i], lwd = 1)
}

# Prepare the data for Stan
data_list <- list(N_people = n_people, N_timepoints = n_timepoints, y = lapply(time_series_list, function(x) x[1:n_timepoints]))

# Set rstan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Define the Stan models
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

stan_model_AR2 <- stan_model(model_code = "
data {
  int<lower=1> N_people;
  int<lower=1> N_timepoints;
  vector[N_timepoints] y[N_people];
}
parameters {
  real phi2;
  real<lower=0> sigma;
  real mu;
}
model {
  phi2 ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  mu ~ normal(0, 10);
  for (j in 1:N_people) {
    for (n in 3:N_timepoints) {
      y[j][n] ~ normal(mu + phi2 * (y[j][n-2] - mu), sigma);
    }
  }
}
generated quantities {
  vector[N_timepoints] log_lik[N_people];
  vector[N_timepoints] y_hat[N_people];
  for (j in 1:N_people) {
    log_lik[j][1] = 0;
    log_lik[j][2] = 0;
    y_hat[j][1] = y[j][1];
    y_hat[j][2] = y[j][2];
    for (n in 3:N_timepoints) {
      log_lik[j][n] = normal_lpdf(y[j][n] | mu + phi2 * (y[j][n-2] - mu), sigma);
      y_hat[j][n] = normal_rng(mu + phi2 * (y[j][n-2] - mu), sigma);
    }
  }
}
")

stan_model_AR3 <- stan_model(model_code = "
data {
  int<lower=1> N_people;
  int<lower=1> N_timepoints;
  vector[N_timepoints] y[N_people];
}
parameters {
  real phi3;
  real<lower=0> sigma;
  real mu;
}
model {
  phi3 ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  mu ~ normal(0, 10);
  for (j in 1:N_people) {
    for (n in 4:N_timepoints) {
      y[j][n] ~ normal(mu + phi3 * (y[j][n-3] - mu), sigma);
    }
  }
}
generated quantities {
  vector[N_timepoints] log_lik[N_people];
  vector[N_timepoints] y_hat[N_people];
  for (j in 1:N_people) {
    log_lik[j][1] = 0;
    log_lik[j][2] = 0;
    log_lik[j][3] = 0;
    y_hat[j][1] = y[j][1];
    y_hat[j][2] = y[j][2];
    y_hat[j][3] = y[j][3];
    for (n in 4:N_timepoints) {
      log_lik[j][n] = normal_lpdf(y[j][n] | mu + phi3 * (y[j][n-3] - mu), sigma);
      y_hat[j][n] = normal_rng(mu + phi3 * (y[j][n-3] - mu), sigma);
    }
  }
}
")

###############################################
# Fitting the Stan Models and Calculating ICs #
###############################################

# Fit the Stan models
fit_combined_AR1 <- sampling(stan_model_AR1, data = data_list, iter = 2000, chains = 4)
fit_combined_AR2 <- sampling(stan_model_AR2, data = data_list, iter = 2000, chains = 4)
fit_combined_AR3 <- sampling(stan_model_AR3, data = data_list, iter = 2000, chains = 4)

# Extract posterior means of the parameters
posterior_mean_AR1 <- summary(fit_combined_AR1)$summary
posterior_mean_AR2 <- summary(fit_combined_AR2)$summary
posterior_mean_AR3 <- summary(fit_combined_AR3)$summary

# Extract log-likelihoods
log_lik_AR1 <- extract_log_lik(fit_combined_AR1, merge_chains = TRUE)
log_lik_AR2 <- extract_log_lik(fit_combined_AR2, merge_chains = TRUE)
log_lik_AR3 <- extract_log_lik(fit_combined_AR3, merge_chains = TRUE)

# Reshape log-likelihood matrix to group by individual and time point
reshape_log_lik <- function(log_lik, n_people, n_timepoints) {
  n_samples <- nrow(log_lik)
  reshaped_log_lik <- array(0, dim = c(n_samples, n_people, n_timepoints))
  for (i in 1:n_people) {
    reshaped_log_lik[, i, ] <- log_lik[, seq(i, by = n_people, length.out = n_timepoints)]
  }
  return(reshaped_log_lik)
}

log_lik_AR1_reshaped <- reshape_log_lik(log_lik_AR1, n_people, n_timepoints)
log_lik_AR2_reshaped <- reshape_log_lik(log_lik_AR2, n_people, n_timepoints)
log_lik_AR3_reshaped <- reshape_log_lik(log_lik_AR3, n_people, n_timepoints)

# Calculate total log-likelihood for each person and chain combination
total_log_lik <- function(log_lik) {
  apply(log_lik, c(1, 2), sum)  # Sum log-likelihoods across time points
}

log_lik_AR1_total <- total_log_lik(log_lik_AR1_reshaped)
log_lik_AR2_total <- total_log_lik(log_lik_AR2_reshaped)
log_lik_AR3_total <- total_log_lik(log_lik_AR3_reshaped)

# Calculate WAIC and LOOIC using total log-likelihood
loo_AR1_total <- loo(log_lik_AR1_total)
loo_AR2_total <- loo(log_lik_AR2_total)
loo_AR3_total <- loo(log_lik_AR3_total)

# Extract posterior samples of y_hat for each model
y_hat_AR1 <- extract(fit_combined_AR1, pars = "y_hat")$y_hat
y_hat_AR2 <- extract(fit_combined_AR2, pars = "y_hat")$y_hat
y_hat_AR3 <- extract(fit_combined_AR3, pars = "y_hat")$y_hat

# Calculate MSE for each model (corrected)
calculate_mse <- function(y_hat, y_true) {
  mse_values <- sapply(1:dim(y_hat)[1], function(s) {  # Iterate over MCMC samples
    mse_per_person <- sapply(1:length(y_true), function(i) mean((y_hat[s, i, ] - y_true[[i]])^2))
    return(mean(mse_per_person)) # Average MSE across individuals for this sample
  })
  return(mean(mse_values)) # Average MSE across all MCMC samples
}

mse_AR1 <- calculate_mse(y_hat_AR1, data_list$y)
mse_AR2 <- calculate_mse(y_hat_AR2, data_list$y)
mse_AR3 <- calculate_mse(y_hat_AR3, data_list$y)

# Calculate total log-likelihood for each chain
total_log_lik2 <- function(log_lik) {
  apply(log_lik, 1, sum)  # Sum all log-likelihood values
}

log_lik_AR1_total2 <- mean(total_log_lik2(log_lik_AR1))
log_lik_AR2_total2 <- mean(total_log_lik2(log_lik_AR2))
log_lik_AR3_total2 <- mean(total_log_lik2(log_lik_AR3))

# Create a data frame to store the results
weights_df <- data.frame(
  Model = c("AR(1)", "AR(2)", "AR(3)"),
  LOOIC = c(loo_AR1_total$estimates["looic", "Estimate"], loo_AR2_total$estimates["looic", "Estimate"], loo_AR3_total$estimates["looic", "Estimate"]),
  p_eff_LOOIC = c(loo_AR1_total$estimates["p_loo", "Estimate"], loo_AR2_total$estimates["p_loo", "Estimate"], loo_AR3_total$estimates["p_loo", "Estimate"]),
  MSE = c(mse_AR1, mse_AR2, mse_AR3),
  LL = c(log_lik_AR1_total2, log_lik_AR2_total2, log_lik_AR3_total2)
)

# Print the results in the desired format
print(weights_df)
