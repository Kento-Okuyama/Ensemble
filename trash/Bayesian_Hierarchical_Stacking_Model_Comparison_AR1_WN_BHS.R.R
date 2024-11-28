# Load necessary libraries
library(rstan)
library(loo)

# Set parameters and data
N <- 10   # Number of individuals
Nt <- 25  # Number of time points per individual
phi <- 0.8                   # AR(1) coefficient
sigma_AR <- 1                # AR(1) process variance
sigma_WN <- 0.5              # White noise process variance
beta <- 2                    # Coefficient for the logistic function
X2 <- rnorm(N, 0, 1)         # Cross-sectional variable

# Function to generate time-series data
generate_time_series <- function(N, Nt, phi, sigma_AR, sigma_WN, beta, X2) {
  y <- matrix(0, nrow=N, ncol=Nt)  # Store time-series data
  omega <- 1 / (1 + exp(-beta * X2))  # Mixture weights
  
  for (i in 1:N) {
    for (t in 2:Nt) {
      # Generate AR(1) component and white noise component
      ar1_component <- phi * y[i, t-1] + rnorm(1, 0, sigma_AR)
      wn_component <- rnorm(1, 0, sigma_WN)
      # Combine the two components based on the mixture weight
      y[i, t] <- omega[i] * ar1_component + (1 - omega[i]) * wn_component
    }
  }
  return(list(y=y, omega=omega))
}

# Simulate data
sim_data <- generate_time_series(N, Nt, phi, sigma_AR, sigma_WN, beta, X2)
y <- sim_data$y
omega <- sim_data$omega

# 1. AR(1) Model Stan Code
ar1_code <- "
data {
  int<lower=1> N;        // Number of individuals
  int<lower=1> Nt;       // Number of time points
  matrix[N, Nt] y;       // Time-series data
}
parameters {
  real<lower=0, upper=1> phi;    // AR(1) coefficient with prior
  real<lower=0> sigma_AR;        // AR(1) process standard deviation
}
model {
  // Priors
  phi ~ beta(2, 2);               // Prior for AR(1) coefficient (values between 0 and 1)
  sigma_AR ~ normal(0, 1);        // Prior for standard deviation, positive values only

  for (i in 1:N) {
    for (t in 2:Nt) {
      y[i, t] ~ normal(phi * y[i, t-1], sigma_AR);
    }
  }
}
generated quantities {
  real log_lik[N, Nt];  // Log-likelihood for each observation
  real y_pred[N, Nt];   // Predicted values
  
  for (i in 1:N) {
    log_lik[i, 1] = normal_lpdf(y[i, 1] | 0, 1);
    y_pred[i, 1] = y[i, 1];
    
    for (t in 2:Nt) {
      log_lik[i, t] = normal_lpdf(y[i, t] | phi * y[i, t-1], sigma_AR);
      y_pred[i, t] = normal_rng(phi * y[i, t-1], sigma_AR);
    }
  }
}
"

# 2. White Noise Model Stan Code
white_noise_code <- "
data {
  int<lower=1> N;        // Number of individuals
  int<lower=1> Nt;       // Number of time points
  matrix[N, Nt] y;       // Time-series data
}
parameters {
  real<lower=0> sigma_WN;        // White noise standard deviation
}
model {
  // Prior
  sigma_WN ~ normal(0, 1);       // Prior for standard deviation, positive values only

  for (i in 1:N) {
    for (t in 2:Nt) {
      y[i, t] ~ normal(0, sigma_WN);
    }
  }
}
generated quantities {
  real log_lik[N, Nt];  // Log-likelihood for each observation
  real y_pred[N, Nt];   // Predicted values
  
  for (i in 1:N) {
    log_lik[i, 1] = normal_lpdf(y[i, 1] | 0, 1);
    y_pred[i, 1] = y[i, 1];
    
    for (t in 2:Nt) {
      log_lik[i, t] = normal_lpdf(y[i, t] | 0, sigma_WN);
      y_pred[i, t] = normal_rng(0, sigma_WN);
    }
  }
}
"

# 3. Bayesian Hierarchical Stacking (BHS) Model Stan Code
bhs_code <- "
data {
  int<lower=1> N;        // Number of individuals
  int<lower=1> Nt;       // Number of time points
  matrix[N, Nt] y;       // Time-series data
  vector[N] X2;          // Cross-sectional variable
}
parameters {
  real<lower=0, upper=1> phi;    // AR(1) coefficient with prior
  real<lower=0> sigma_AR;        // AR(1) process standard deviation
  real<lower=0> sigma_WN;        // White noise standard deviation
  real beta;                     // Logistic regression coefficient
}
model {
  vector[N] omega;

  // Priors
  phi ~ beta(2, 2);              // Prior for AR(1) coefficient (values between 0 and 1)
  sigma_AR ~ normal(0, 1);        // Prior for AR(1) standard deviation
  sigma_WN ~ normal(0, 1);        // Prior for white noise standard deviation
  beta ~ normal(0, 1);            // Prior for logistic regression coefficient

  // Logistic function for omega based on X2
  for (i in 1:N) {
    omega[i] = inv_logit(beta * X2[i]);
  }

  // Likelihood combining AR(1) and White noise
  for (i in 1:N) {
    for (t in 2:Nt) {
      target += log_mix(omega[i],
        normal_lpdf(y[i,t] | phi * y[i,t-1], sigma_AR),
        normal_lpdf(y[i,t] | 0, sigma_WN));
    }
  }
}
generated quantities {
  real log_lik[N, Nt];  // Log-likelihood for each observation
  real y_pred[N, Nt];   // Predicted values
  vector[N] omega;

  // Logistic function for omega based on X2
  for (i in 1:N) {
    omega[i] = inv_logit(beta * X2[i]);
    log_lik[i, 1] = normal_lpdf(y[i, 1] | 0, 1);
    y_pred[i, 1] = y[i, 1];

    for (t in 2:Nt) {
      log_lik[i, t] = log_mix(omega[i],
        normal_lpdf(y[i, t] | phi * y[i, t-1], sigma_AR),
        normal_lpdf(y[i, t] | 0, sigma_WN));
      y_pred[i, t] = log_mix(omega[i],
        normal_rng(phi * y[i, t-1], sigma_AR),
        normal_rng(0, sigma_WN));
    }
  }
}
"

# Compile Stan models
ar1_model <- stan_model(model_code = ar1_code)
wn_model <- stan_model(model_code = white_noise_code)
bhs_model <- stan_model(model_code = bhs_code)

# Prepare data for Stan
stan_data <- list(N = N, Nt = Nt, y = y, X2 = X2)

# Fit the models
fit_ar1 <- sampling(ar1_model, data = stan_data, chains = 4, iter = 2000, warmup = 500)
fit_wn <- sampling(wn_model, data = stan_data, chains = 4, iter = 2000, warmup = 500)
fit_bhs <- sampling(bhs_model, data = stan_data, chains = 4, iter = 2000, warmup = 500,
                    control = list(max_treedepth = 10, adapt_delta = 0.8))

# Extract log-likelihood
log_lik_ar1 <- extract_log_lik(fit_ar1, merge_chains = TRUE)
log_lik_wn <- extract_log_lik(fit_wn, merge_chains = TRUE)
log_lik_bhs <- extract_log_lik(fit_bhs, merge_chains = TRUE)

# Perform LOO-CV
loo_ar1 <- loo(log_lik_ar1)
loo_wn <- loo(log_lik_wn)
loo_bhs <- loo(log_lik_bhs)

# Compare models
compare_models <- compare(loo_ar1, loo_wn, loo_bhs)
print(compare_models)

# Display the results
print("AR(1) model:")
print(fit_ar1)
print("White Noise model:")
print(fit_wn)
print("BHS model:")
print(fit_bhs)
