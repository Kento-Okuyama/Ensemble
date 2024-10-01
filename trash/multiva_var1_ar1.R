###########################################################
# Definition of Latent Variable Model and Data Generation #
###########################################################

# Remove all objects
rm(list = ls())

# Load necessary libraries
library(forecast)
library(rstan)
library(loo)
library(ggplot2)
library(bayesplot)

# Define parameters for data generation
n_people <- 10
n_burnins <- 25
n_timepoints <- 50

# Data generation for the latent variable model
set.seed(123)
F <- array(0, dim = c(n_people, 2, n_timepoints + n_burnins))
Y1 <- array(0, dim = c(n_people, 3, n_timepoints))
Y2 <- array(0, dim = c(n_people, 3, n_timepoints))

# Coefficients for VAR(1) model
phi <- matrix(c(0.5, 0.2, -0.3, 0.4), nrow = 2, ncol = 2)

for (i in 1:n_people) {
  # Initial values for F
  F[i, , 1:2] <- matrix(rnorm(4, 0, 1), nrow = 2) # Initialize the first two timepoints
  
  # Generate latent variables F using VAR(1) model
  for (t in 3:(n_burnins + n_timepoints)) {
    F[i, , t] <- phi %*% F[i, , t-1] + rnorm(2, 0, 0.1)
  }
  
  # Generate observed data Y1 and Y2 from latent variables F
  for (j in 1:3) {
    Y1[i, j, ] <- F[i, 1, (n_burnins + 1):(n_burnins + n_timepoints)] + rnorm(n_timepoints, 0, 0.1)
    Y2[i, j, ] <- F[i, 2, (n_burnins + 1):(n_burnins + n_timepoints)] + rnorm(n_timepoints, 0, 0.1)
  }
}

# Prepare data for Stan
data_list <- list(
  N_people = n_people,
  N_timepoints = n_timepoints,
  Y1 = Y1,
  Y2 = Y2
)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#########################
# Stan Model Definition #
#########################

# Stan code for AR(1) model
stan_code <- "
data {
  int<lower=1> N_people;
  int<lower=1> N_timepoints;
  matrix[3, N_timepoints] Y1[N_people];
  matrix[3, N_timepoints] Y2[N_people];
}
parameters {
  matrix[N_timepoints, 2] F[N_people]; // Latent variables F1 and F2
  real<lower=-1, upper=1> phi1;        // AR(1) coefficient for F1
  real<lower=-1, upper=1> phi2;        // AR(1) coefficient for F2
  real<lower=0> sigma[2, 3];           // Standard deviation of observation noise
}
model {
  phi1 ~ uniform(-1, 1); // Prior for AR(1) coefficient for F1
  phi2 ~ uniform(-1, 1); // Prior for AR(1) coefficient for F2

  for (i in 1:N_people) {
    for (t in 2:N_timepoints) { // AR(1) model
      F[i, t, 1] ~ normal(phi1 * F[i, t-1, 1], 1);
      F[i, t, 2] ~ normal(phi2 * F[i, t-1, 2], 1);
    }
    F[i, 1, 1] ~ normal(0, 1); // Initial state for F1
    F[i, 1, 2] ~ normal(0, 1); // Initial state for F2

    for (t in 1:N_timepoints) {
      for (k in 1:3) {
        Y1[i, k, t] ~ normal(F[i, t, 1], sigma[1, k]);
        Y2[i, k, t] ~ normal(F[i, t, 2], sigma[2, k]);
      }
    }
  }
}
generated quantities {
  vector[N_timepoints] log_lik[N_people];
  matrix[3, N_timepoints] y_hat1[N_people];
  matrix[3, N_timepoints] y_hat2[N_people];
  
  for (i in 1:N_people) {
    for (t in 1:N_timepoints) {
      log_lik[i, t] = 0;
      for (k in 1:3) {
        y_hat1[i, k, t] = normal_rng(F[i, t, 1], sigma[1, k]);
        y_hat2[i, k, t] = normal_rng(F[i, t, 2], sigma[2, k]);
        log_lik[i, t] += normal_lpdf(Y1[i, k, t] | F[i, t, 1], sigma[1, k]) +
                         normal_lpdf(Y2[i, k, t] | F[i, t, 2], sigma[2, k]);
      }
    }
  }
}
"

# Compile Stan model
stan_model <- stan_model(model_code = stan_code)

#################
# Model Fitting #
#################

# Set initial values based on data
init_values <- function() {
  F_init <- array(0, dim = c(n_people, n_timepoints, 2))
  for (i in 1:n_people) {
    for (t in 1:n_timepoints) {
      F_init[i, t, 1] <- mean(data_list$Y1[i, , t])
      F_init[i, t, 2] <- mean(data_list$Y2[i, , t])
    }
  }
  list(
    F = F_init,
    phi1 = 0.5,
    phi2 = 0.5,
    sigma = matrix(runif(6, 0.1, 1), nrow = 2)
  )
}

# Fit the model with better initial values
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4, init = init_values)

# Extract samples
samples <- extract(fit)

############################
# Visualization of Results #
############################

# Obtain the mean of the posterior predictive distribution
y_hat1_mean <- apply(samples$y_hat1, c(2, 3, 4), mean) # Result is (n_people, 3, n_timepoints)
y_hat2_mean <- apply(samples$y_hat2, c(2, 3, 4), mean) # Result is (n_people, 3, n_timepoints)

# Define function to calculate MSE
calculate_mse <- function(y_hat, y_true) {
  return(mean((y_hat - y_true)^2))
}

# Calculate MSE
mse1 <- array(0, dim = c(n_people, n_timepoints))
mse2 <- array(0, dim = c(n_people, n_timepoints))
for (i in 1:n_people) {
  for (t in 1:n_timepoints) {
    mse1[i, t] <- calculate_mse(y_hat1_mean[i, , t], data_list$Y1[i, , t])
    mse2[i, t] <- calculate_mse(y_hat2_mean[i, , t], data_list$Y2[i, , t])
  }
}

# Plot MSE
mse_df1 <- data.frame(
  Time = rep(1:n_timepoints, n_people),
  MSE = as.vector(mse1),
  Person = rep(1:n_people, each = n_timepoints)
)

mse_df2 <- data.frame(
  Time = rep(1:n_timepoints, n_people),
  MSE = as.vector(mse2),
  Person = rep(1:n_people, each = n_timepoints)
)

# Plot posterior distributions of AR coefficients
posterior <- as.array(fit)
mcmc_areas(posterior, pars = c("phi1", "phi2"))

# Plot predicted vs observed values for the first person
person_index <- 1
for (k in 1:3) {
  plot(1:n_timepoints, data_list$Y1[person_index, k, ], type = "l", col = "blue", ylim = range(c(data_list$Y1[person_index, k, ], y_hat1_mean[person_index, k, ])), ylab = "Y1", xlab = "Time")
  lines(1:n_timepoints, y_hat1_mean[person_index, k, ], col = "red")
  title(paste("Observed vs Predicted Y1 for Person", person_index, "Variable", k))
  legend("topright", legend = c("Observed", "Predicted"), col = c("blue", "red"), lty = 1)
}

# Extract log likelihood
log_lik <- extract_log_lik(fit)

# Check the structure of log_lik
str(log_lik)  # Should show a matrix with dimensions (iterations, 10 * 50)

# Reshape log_lik to (iterations, n_people, n_timepoints)
iterations <- dim(log_lik)[1]
log_lik_reshaped <- array(log_lik, dim = c(iterations, n_people, n_timepoints))

# Average log likelihood over iterations
log_lik_mean <- apply(log_lik_reshaped, c(2, 3), mean)  # Average over iterations, keeping (n_people, n_timepoints)

# Optionally, average over people to get a single time series
log_lik_mean_over_people <- apply(log_lik_mean, 2, mean)

# Plot log likelihood over time
plot(1:n_timepoints, log_lik_mean_over_people, type = "l", col = "black", ylab = "Log Likelihood", xlab = "Time")
title("Log Likelihood over Time")

# Calculate 95% prediction intervals
y_hat1_lower <- apply(samples$y_hat1, c(2, 3, 4), quantile, probs = 0.025)
y_hat1_upper <- apply(samples$y_hat1, c(2, 3, 4), quantile, probs = 0.975)

# Plot prediction intervals vs observed values for the first person
for (k in 1:3) {
  plot(1:n_timepoints, data_list$Y1[person_index, k, ], type = "l", col = "blue", ylim = range(c(data_list$Y1[person_index, k, ], y_hat1_lower[person_index, k, ], y_hat1_upper[person_index, k, ])), ylab = "Y1", xlab = "Time")
  lines(1:n_timepoints, y_hat1_mean[person_index, k, ], col = "red")
  lines(1:n_timepoints, y_hat1_lower[person_index, k, ], col = "grey", lty = 2)
  lines(1:n_timepoints, y_hat1_upper[person_index, k, ], col = "grey", lty = 2)
  title(paste("Observed vs Predicted Y1 with Prediction Interval for Person", person_index, "Variable", k))
  legend("topright", legend = c("Observed", "Predicted", "95% Interval"), col = c("blue", "red", "grey"), lty = c(1, 1, 2))
}
