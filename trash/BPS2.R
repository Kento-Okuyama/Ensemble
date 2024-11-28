# Load necessary libraries
library(rstan)
library(dplyr)
library(ggplot2)
library(loo)

# Data generation parameters (same as before)
set.seed(123)
N <- 10    # Number of time series
Nt <- 50   # Length of each time series
lambda <- 0.5
mu <- c(2, -2, 1, 0)
ar_pos <- 0.7
ar_neg <- -0.7
ma_coef <- 0.5
sigma <- c(0.5, 0.7, 0.6, 0.4)

# Transition probability matrix (same as before)
transition_matrix <- matrix(c(
  0.8, 0.05, 0.1, 0.05,
  0.1, 0.8, 0.05, 0.05,
  0.1, 0.05, 0.8, 0.05,
  0.1, 0.05, 0.1, 0.75
), nrow = 4, byrow = TRUE)

# Data generation function (same as before)
generate_data <- function(N, Nt, mu, sigma, transition_matrix, lambda, ar_pos, ar_neg, ma_coef) {
  y <- matrix(NA, nrow = N, ncol = Nt)
  regimes <- matrix(NA, nrow = N, ncol = Nt)
  
  for (n in 1:N) {
    current_regime <- 1
    y[n, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])
    regimes[n, 1] <- current_regime
    
    for (t in 2:Nt) {
      current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
      regimes[n, t] <- current_regime
      
      target_mean <- lambda * mu[current_regime] + (1 - lambda) * y[n, t - 1]
      if (current_regime == 1) {
        y[n, t] <- (1 - ar_pos) * target_mean + ar_pos * y[n, t-1] + rnorm(1, sd = sigma[current_regime])
      } else if (current_regime == 2) {
        y[n, t] <- (1 - ar_neg) * target_mean + ar_neg * y[n, t-1] + rnorm(1, sd = sigma[current_regime])
      } else if (current_regime == 3) {
        y[n, t] <- target_mean + ma_coef * (y[n, t - 1] - target_mean) + rnorm(1, sd = sigma[current_regime])
      } else {
        y[n, t] <- rnorm(1, mean = target_mean, sd = sigma[current_regime])
      }
    }
  }
  
  list(y = y, regimes = regimes)
}

# Generate data
data <- generate_data(N, Nt, mu, sigma, transition_matrix, lambda, ar_pos, ar_neg, ma_coef)
y <- data$y

# Prepare data for Stan
stan_data <- list(
  N = N,
  Nt = Nt,
  y = y
)

# Stan model code for individual model fitting
stan_individual_code <- "
data {
  int<lower=1> N;          // Number of time series
  int<lower=1> Nt;         // Length of each time series
  real y[N, Nt];           // Observed data
}
parameters {
  real<lower=0, upper=1> ar_coef;     // AR coefficient
  real<lower=-1, upper=1> ma_coef;    // MA coefficient
  real mu;                            // Mean for the white noise model
  real<lower=0> sigma;                // Standard deviation for all models
}
model {
  // Priors
  ar_coef ~ beta(2, 2);       // Prior for AR coefficient
  ma_coef ~ normal(0, 0.5);   // Prior for MA coefficient
  mu ~ normal(0, 5);          // Prior for mean
  sigma ~ cauchy(0, 2);       // Prior for standard deviation

  // Likelihood
  for (n in 1:N) {
    for (t in 2:Nt) {
      real prediction_ar = ar_coef * y[n, t - 1];
      real prediction_ma = mu + ma_coef * (y[n, t - 1] - mu);
      real prediction_wn = mu;

      // Add likelihoods for each model
      y[n, t] ~ normal(prediction_ar, sigma);  // AR(1)
      y[n, t] ~ normal(prediction_ma, sigma);  // MA(1)
      y[n, t] ~ normal(prediction_wn, sigma);  // White Noise
    }
  }
}
generated quantities {
  real y_pred_ar[N, Nt];
  real y_pred_ma[N, Nt];
  real y_pred_wn[N, Nt];

  for (n in 1:N) {
    y_pred_ar[n, 1] = mu;  // Initialize predictions
    y_pred_ma[n, 1] = mu;
    y_pred_wn[n, 1] = mu;

    for (t in 2:Nt) {
      y_pred_ar[n, t] = normal_rng(ar_coef * y[n, t - 1], sigma);
      y_pred_ma[n, t] = normal_rng(mu + ma_coef * (y[n, t - 1] - mu), sigma);
      y_pred_wn[n, t] = normal_rng(mu, sigma);
    }
  }
}
"

# Compile and fit individual models
fit_individual <- stan(model_code = stan_individual_code, data = stan_data, iter = 2000, chains = 4)

# Extract predictions from generated quantities
y_pred_ar <- extract(fit_individual, pars = "y_pred_ar")$y_pred_ar
y_pred_ma <- extract(fit_individual, pars = "y_pred_ma")$y_pred_ma
y_pred_wn <- extract(fit_individual, pars = "y_pred_wn")$y_pred_wn

# Compute mean predictions for each model
f1 <- apply(y_pred_ar, c(2, 3), mean)  # AR predictions
f2 <- apply(y_pred_ma, c(2, 3), mean)  # MA predictions
f3 <- apply(y_pred_wn, c(2, 3), mean)  # White noise predictions

# Combine predictions into a single array for BPS
f <- array(c(f1, f2, f3), dim = c(N, Nt, 3))

# Prepare data for BPS
stan_bps_data <- list(
  N = N,
  Nt = Nt,
  J = 3,   # Number of models
  y = y,
  f = f
)

# Stan model code for BPS
stan_bps_code <- "
data {
  int<lower=1> N;          // Number of time series
  int<lower=1> Nt;         // Length of each time series
  int<lower=1> J;          // Number of models
  real y[N, Nt];           // Observed data
  real f[N, Nt, J];        // Model predictions
}
parameters {
  vector[Nt] alpha;         // Time-varying intercept
  simplex[J] beta[Nt];      // Time-varying weights for each time step
  real<lower=0> sigma;      // Observation noise standard deviation
}
model {
  // Priors
  alpha ~ normal(0, 5);
  for (t in 1:Nt) {
    beta[t] ~ dirichlet(rep_vector(1.0, J));  // Dirichlet prior for weights
  }
  sigma ~ cauchy(0, 2);

  // Likelihood
  for (n in 1:N) {
    for (t in 1:Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = f[n, t, j];
      }
      y[n, t] ~ normal(alpha[t] + dot_product(beta[t], model_predictions), sigma);
    }
  }
}
generated quantities {
  matrix[N, Nt-1] log_lik;  // Log-likelihood for PSIS-LOO
  for (n in 1:N) {
    for (t in 2:Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = f[n, t, j];
      }
      log_lik[n, t-1] = normal_lpdf(y[n, t] | alpha[t] + dot_product(beta[t], model_predictions), sigma);
    }
  }
}
"

# Compile and fit BPS model
fit_bps <- stan(model_code = stan_bps_code, data = stan_bps_data, iter = 2000, chains = 4)

# Display results
print(fit_bps, pars = c("alpha", "beta", "sigma"))

# Extract log-likelihood for PSIS-LOO cross-validation
log_lik <- extract_log_lik(fit_bps, parameter_name = "log_lik", merge_chains = FALSE)
loo_result <- loo(log_lik, moment_match = TRUE)
print(loo_result)
