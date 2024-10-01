# Load required libraries
library(forecast)
library(rstan)
library(loo)
library(ggplot2)

# Define parameters for data generation
n_people <- 10 
n_burnins <- 25 
n_timepoints <- 30 

# Initialize list to store time series data for each person
time_series_list <- list()

# Generate ARMA(2,2) time series data for each person
set.seed(123)  # Set random seed for reproducibility
for (i in 1:n_people) {
  ar <- c(0.5, -0.3)  # Autoregressive coefficients
  ma <- c(0.4, -0.2)  # Moving average coefficients
  mu <- 0             # Mean of the process
  sigma <- 1          # Standard deviation of the process
  e <- rnorm(n_burnins + n_timepoints, mean = mu, sd = sigma) # White noise
  data <- numeric(n_burnins + n_timepoints) # Initialize data vector
  data[1:2] <- e[1:2]                        # Set initial values
  
  # Generate ARMA(2,2) process
  for (t in 3:(n_burnins + n_timepoints)) {
    data[t] <- mu + ar[1] * (data[t - 1] - mu) + 
      ar[2] * (data[t - 2] - mu) + 
      e[t] + ma[1] * e[t - 1] + ma[2] * e[t - 2] 
  }
  
  # Store time series data after discarding burn-in period
  time_series_list[[i]] <- data[(n_burnins + 1):(n_burnins + n_timepoints)]
}

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
  real phi1;
  real phi2;
  real<lower=0> sigma;
  real mu;
}
model {
  phi1 ~ normal(0, 1);
  phi2 ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  mu ~ normal(0, 10);
  for (j in 1:N_people) {
    for (n in 3:N_timepoints) {
      y[j][n] ~ normal(mu + phi1 * (y[j][n-1] - mu) + phi2 * (y[j][n-2] - mu), sigma);
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
      log_lik[j][n] = normal_lpdf(y[j][n] | mu + phi1 * (y[j][n-1] - mu) + phi2 * (y[j][n-2] - mu), sigma);
      y_hat[j][n] = normal_rng(mu + phi1 * (y[j][n-1] - mu) + phi2 * (y[j][n-2] - mu), sigma);
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
  real phi1;
  real phi2;
  real phi3;
  real<lower=0> sigma;
  real mu;
}
model {
  phi1 ~ normal(0, 1);
  phi2 ~ normal(0, 1);
  phi3 ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  mu ~ normal(0, 10);
  for (j in 1:N_people) {
    for (n in 4:N_timepoints) {
      y[j][n] ~ normal(mu + phi1 * (y[j][n-1] - mu) + phi2 * (y[j][n-2] - mu) + phi3 * (y[j][n-3] - mu), sigma);
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
      log_lik[j][n] = normal_lpdf(y[j][n] | mu + phi1 * (y[j][n-1] - mu) + phi2 * (y[j][n-2] - mu) + phi3 * (y[j][n-3] - mu), sigma);
      y_hat[j][n] = normal_rng(mu + phi1 * (y[j][n-1] - mu) + phi2 * (y[j][n-2] - mu) + phi3 * (y[j][n-3] - mu), sigma);
    }
  }
}
")

# Prepare the data for Stan
data_list <- list(
  N_people = n_people,
  N_timepoints = n_timepoints,
  y = do.call(rbind, lapply(1:n_people, function(i) time_series_list[[i]])),
  loo_scores = array(c(loo_AR1$pointwise[,1], loo_AR2$pointwise[,1], loo_AR3$pointwise[,1]), 
                     dim = c(n_people, n_timepoints, 3))
)

# Fit the Stan models
fit_AR1 <- sampling(stan_model_AR1, data = data_list, iter = 2000, chains = 4)
fit_AR2 <- sampling(stan_model_AR2, data = data_list, iter = 2000, chains = 4)
fit_AR3 <- sampling(stan_model_AR3, data = data_list, iter = 2000, chains = 4)

# Extract log-likelihood and posterior predictive samples (y_hat) for each model
log_lik_AR1 <- extract_log_lik(fit_AR1)
log_lik_AR2 <- extract_log_lik(fit_AR2)
log_lik_AR3 <- extract_log_lik(fit_AR3)

# Calculate LOO scores
loo_AR1 <- loo(log_lik_AR1)
loo_AR2 <- loo(log_lik_AR2)
loo_AR3 <- loo(log_lik_AR3)

# Store LOO scores in an array
loo_scores <- array(c(loo_AR1$pointwise[,1], loo_AR2$pointwise[,1], loo_AR3$pointwise[,1]), 
                    dim = c(n_people, n_timepoints, 3))

# Define the Stan model for dynamic weights using GP and LOO scores
stan_model_ens <- stan_model(model_code = "
  data {
    int<lower=1> N_people;
    int<lower=1> N_timepoints;
    vector[N_timepoints] y[N_people];
    vector[3] loo_scores[N_people, N_timepoints];
  }
  parameters {
    real<lower=0> eta;
    real<lower=0> rho;
    real<lower=0> sigma;
    matrix[N_timepoints, 3] w_raw[N_people];
  }
  transformed parameters {
    matrix[N_timepoints, 3] w[N_people];
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
      matrix[N_timepoints, 3] w_temp = L_K * w_raw[j];
      for (t in 1:N_timepoints) {
        vector[3] w_t = softmax(to_vector(w_temp[t]));
        w[j, t] = to_row_vector(w_t);
      }
    }
  }
  model {
    eta ~ normal(0, 1);
    rho ~ normal(0, 1);
    for (j in 1:N_people) {
      for (k in 1:3) {
        w_raw[j, , k] ~ normal(0, 1);
      }
    }
    sigma ~ cauchy(0, 2.5);
  
    for (j in 1:N_people) {
      for (t in 1:N_timepoints) {
        vector[3] loo_t = loo_scores[j, t];
        vector[3] w_t = to_vector(w[j, t]);
        y[j][t] ~ normal(dot_product(loo_t, w_t), 1e-10);
      }
    }
  }
  generated quantities {
    vector[N_timepoints] y_hat_ens[N_people];
    for (j in 1:N_people) {
      for (t in 1:N_timepoints) {
        vector[3] w_t = to_vector(w[j, t]); 
        y_hat_ens[j][t] = dot_product(loo_scores[j, t], w_t);
      }
    }
  }
")

# Set rstan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Fit the ensemble model
fit_combined <- sampling(
  stan_model_ens, 
  data = data_list, 
  iter = 4000, 
  warmup = 2000, 
  chains = 4, 
  control = list(max_treedepth = 20, adapt_delta = 0.999)
)

# Extract samples
samples <- extract(fit_combined)

# Extract weights from the samples
w_samples <- samples$w

# Check the weights
for (j in 1:1) {
  print(paste("Person", j))
  for (t in n_burnins:n_burnins) {
    print(paste("Time", t))
    # Extract the weights for person j at time t across all models
    weights <- w_samples[, j, t, ]
    print(head(weights)) # Print the weights (average over samples can be done)
    print(mean(weights)) # Print the mean of weights over samples
  }
}

# Posterior predictive distributions
y_hat_AR1 <- extract(fit_AR1, pars = "y_hat")$y_hat
y_hat_AR2 <- extract(fit_AR2, pars = "y_hat")$y_hat
y_hat_AR3 <- extract(fit_AR3, pars = "y_hat")$y_hat
y_hat_ensemble <- samples$y_hat_ens

# Ensure y_hat arrays have correct dimensions
y_hat_AR1 <- array(y_hat_AR1, dim = c(dim(y_hat_AR1)[1], n_people, n_timepoints))
y_hat_AR2 <- array(y_hat_AR2, dim = c(dim(y_hat_AR2)[1], n_people, n_timepoints))
y_hat_AR3 <- array(y_hat_AR3, dim = c(dim(y_hat_AR3)[1], n_people, n_timepoints))
y_hat_ensemble <- array(y_hat_ensemble, dim = c(dim(y_hat_ensemble)[1], n_people, n_timepoints))

print(y_hat_AR1[1,1,n_burnins])
print(y_hat_AR2[1,1,n_burnins])
print(y_hat_AR3[1,1,n_burnins])
print(y_hat_ensemble[1,1,n_burnins])

# Function to calculate mean squared error (MSE) at each time point starting from the 3rd data point
calculate_mse_time <- function(y_hat, y_true, start_point = n_burnins) {
  y_hat_mean <- apply(y_hat, c(2, 3), mean)  # Calculate the mean of predictions across samples
  mse_values <- sapply(start_point:ncol(y_true), function(t) {
    mse_per_person <- sapply(1:nrow(y_true), function(i) (y_hat_mean[i, t] - y_true[i, t])^2)
    return(mean(mse_per_person))  # Compute the average MSE across individuals for this time point
  })
  return(mse_values)
}

# Calculate MSE for each model at each time point starting from the 4th data point
mse_time_AR1 <- calculate_mse_time(y_hat_AR1, data_list$y)
mse_time_AR2 <- calculate_mse_time(y_hat_AR2, data_list$y)
mse_time_AR3 <- calculate_mse_time(y_hat_AR3, data_list$y)
mse_time_ensemble <- calculate_mse_time(y_hat_ensemble, data_list$y)

# Define x-axis for the plot (starting from the 3rd data point)
x_axis <- n_burnins:n_timepoints

# Prepare data for plotting
mse_time_data <- data.frame(
  Time = rep(x_axis, 4),
  MSE = c(mse_time_AR1, mse_time_AR2, mse_time_AR3, mse_time_ensemble),
  Model = factor(rep(c("AR(1)", "AR(2)", "AR(3)", "Ensemble"), each = length(x_axis)))
)

# Plot MSE over time for each model and ensemble with y-axis on log scale
ggplot(mse_time_data, aes(x = Time, y = MSE, color = Model, group = Model)) +
  geom_line(size = 1.2) +
  scale_y_log10() +
  labs(title = "MSE over Time", x = "Time", y = "Log(MSE)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("AR(1)" = "gray", "AR(2)" = "gray", "AR(3)" = "gray", "Ensemble" = "purple"))

# Prepare data for density plot
error_data <- data.frame(
  Error = c(mse_time_AR1, mse_time_AR2, mse_time_AR3, mse_time_ensemble),
  Model = factor(rep(c("AR(1)", "AR(2)", "AR(3)", "Ensemble"), each = length(mse_time_AR1)))
)

# Plot error distribution as density plot
ggplot(error_data, aes(x = Error, color = Model, fill = Model)) +
  geom_density(alpha = 0.4) +
  scale_x_log10() +
  labs(title = "Error Distribution by Model", x = "Log(Error)", y = "Density") +
  theme_minimal() +
  theme(legend.position = "top")
