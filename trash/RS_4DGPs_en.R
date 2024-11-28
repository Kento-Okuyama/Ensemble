# Load necessary packages
library(rstan)
library(bayesplot)
library(loo)

# Define Stan code
stan_code <- "
data {
  int<lower=1> N;                  // Number of time series
  int<lower=1> Nt;                 // Length of each time series
  int<lower=1, upper=4> regime[N, Nt];  // Regime index at each time point
  real y[N, Nt];                    // Observation data matrix (each time series in a row)
}
parameters {
  vector[4] mu;                    // Mean for each regime
  real<lower=0, upper=1> lambda;   // Smoothing parameter (closer to 0 emphasizes the previous observation)
  real<lower=-1, upper=1> ar_pos;  // AR coefficient for Regime 1
  real<lower=-1, upper=1> ar_neg;  // AR coefficient for Regime 2
  real<lower=-1, upper=1> ma_coef; // MA coefficient for Regime 3
  vector<lower=0>[4] sigma;        // Standard deviation for each regime
}
model {
  // Setting prior distributions
  mu ~ normal(0, 5);             // Prior for regime means
  lambda ~ beta(2, 2);           // Prior for smoothing parameter
  ar_pos ~ normal(0, 0.5);       // Prior for AR+ coefficient (centered near 0)
  ar_neg ~ normal(0, 0.5);       // Prior for AR- coefficient (centered near 0)
  ma_coef ~ normal(0, 0.5);      // Prior for MA coefficient
  sigma ~ cauchy(0, 2);          // Weakly informative prior for standard deviations
  
  for (n in 1:N) {
    for (t in 2:Nt) {
      int k = regime[n, t];  // Current regime
      
      // Smoothing to gradually approach the mean of the next regime
      real target_mean = lambda * mu[k] + (1 - lambda) * y[n, t-1];
      
      if (k == 1) {
        y[n, t] ~ normal(target_mean + ar_pos * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 2) {
        y[n, t] ~ normal(target_mean + ar_neg * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 3) {
        y[n, t] ~ normal(target_mean + ma_coef * (y[n, t-1] - target_mean), sigma[k]);
      } else {
        y[n, t] ~ normal(target_mean, sigma[k]);
      }
    }
  }
}
generated quantities {
  real y_rep[N, Nt-1];           // Prediction generation
  real log_lik[N, Nt-1];         // Log-likelihood for each observation
  
  for (n in 1:N) {
    for (t in 2:Nt) {
      int k = regime[n, t];
      real target_mean = lambda * mu[k] + (1 - lambda) * y[n, t-1];
      
      if (k == 1) {
        y_rep[n, t-1] = normal_rng(target_mean + ar_pos * (y[n, t-1] - target_mean), sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean + ar_pos * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 2) {
        y_rep[n, t-1] = normal_rng(target_mean + ar_neg * (y[n, t-1] - target_mean), sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean + ar_neg * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 3) {
        y_rep[n, t-1] = normal_rng(target_mean + ma_coef * (y[n, t-1] - target_mean), sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean + ma_coef * (y[n, t-1] - target_mean), sigma[k]);
      } else {
        y_rep[n, t-1] = normal_rng(target_mean, sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean, sigma[k]);
      }
    }
  }
}
"

# Preparing data and running Stan model
set.seed(123)
N <- 5    # Number of time series
Nt <- 30   # Length of each time series
lambda <- 0.5  # Smoothing parameter (corresponds to lambda in Stan model)

# Parameter settings for each regime
mu <- c(2, -2, 1, 0)         # Mean for each regime
ar_pos <- 0.7                # AR coefficient for Regime 1
ar_neg <- -0.7               # AR coefficient for Regime 2
ma_coef <- 0.5               # MA coefficient for Regime 3
sigma <- c(0.5, 0.7, 0.6, 0.4)  # Standard deviation for each regime

# Define transition matrix (example setting with uniform probabilities)
transition_matrix <- matrix(c(
  0.8, 0.1, 0.05, 0.05,  
  0.1, 0.8, 0.05, 0.05,  
  0.05, 0.05, 0.8, 0.1,  
  0.05, 0.05, 0.1, 0.8   
), nrow = 4, byrow = TRUE)

# Generate data and regime indices for each time series
y <- matrix(NA, nrow = N, ncol = Nt)
regime <- matrix(NA, nrow = N, ncol = Nt)

# Data generation with regime transition matrix based on a Markov process
for (i in 1:N) {
  current_regime <- sample(1:4, 1)  
  y[i, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])  # Initial value
  regime[i, 1] <- current_regime
  
  for (t in 2:Nt) {
    # Transition of regime
    current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
    regime[i, t] <- current_regime
    
    # Calculate smoothed target mean
    target_mean <- lambda * mu[current_regime] + (1 - lambda) * y[i, t - 1]
    
    # Generate observation based on regime
    if (current_regime == 1) {  # AR(1) with positive coefficient
      y[i, t] <- target_mean + ar_pos * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else if (current_regime == 2) {  # AR(1) with negative coefficient
      y[i, t] <- target_mean + ar_neg * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else if (current_regime == 3) {  # MA(1) process
      y[i, t] <- target_mean + ma_coef * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else {  # White noise
      y[i, t] <- rnorm(1, mean = target_mean, sd = sigma[current_regime])
    }
  }
}

# Prepare data list for Stan
stan_data <- list(
  N = N,
  Nt = Nt,
  regime = regime,
  y = y
)

# Compile and sample from the Stan model
fit <- stan(model_code = stan_code, data = stan_data, iter = 2000, chains = 4)

# PSIS-LOO cross-validation
log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
loo_result <- loo(log_lik, moment_match = TRUE)
print(loo_result)

# Plot results
bayesplot::mcmc_areas(as.array(fit), pars = c("ar_pos", "ar_neg", "ma_coef"))
bayesplot::mcmc_areas(as.array(fit), pars = c("sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]"))
bayesplot::mcmc_areas(as.array(fit), pars = c("mu[1]", "mu[2]", "mu[3]", "mu[4]", "lambda"))
