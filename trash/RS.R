# Load necessary libraries
library(rstan)
library(bayesplot)
library(loo)
library(ggplot2)
library(dplyr)

# Set the seed for reproducibility
set.seed(123)

# Parameters for data simulation
N <- 5    # Number of time series
Nt <- 30  # Length of each time series
lambda <- 0.5  # Smoothing parameter for target_mean
mu <- c(2, -2, 1, 0)         # Regime means
ar_pos <- 0.7                # Positive AR coefficient
ar_neg <- -0.7               # Negative AR coefficient
ma_coef <- 0.5               # MA coefficient
sigma <- c(0.5, 0.7, 0.6, 0.4)  # Standard deviations for each regime

# Define transition matrix for regime switching
transition_matrix <- matrix(c(
  0.8, 0.05, 0.1, 0.05,  
  0.1, 0.8, 0.05, 0.05,  
  0.1, 0.05, 0.8, 0.05,  
  0.1, 0.05, 0.1, 0.75   
), nrow = 4, byrow = TRUE)

# Initialize matrices to store simulated data
y <- matrix(NA, nrow = N, ncol = Nt)
regime <- matrix(NA, nrow = N, ncol = Nt)

# Simulate data with regime transitions
for (i in 1:N) {
  current_regime <- 1 # sample(1:4, 1)  
  y[i, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])  # Initial value
  regime[i, 1] <- current_regime
  
  for (t in 2:Nt) {
    current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
    regime[i, t] <- current_regime
    
    target_mean <- lambda * mu[current_regime] + (1 - lambda) * y[i, t - 1]
    
    if (current_regime == 1) {
      y[i, t] <- target_mean + ar_pos * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else if (current_regime == 2) {
      y[i, t] <- target_mean + ar_neg * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else if (current_regime == 3) {
      y[i, t] <- target_mean + ma_coef * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else {
      y[i, t] <- rnorm(1, mean = target_mean, sd = sigma[current_regime])
    }
  }
}

# Convert to data frame for ggplot visualization
y_df <- data.frame(
  time = rep(1:Nt, times = N),
  value = as.vector(y),
  series = rep(1:N, each = Nt),
  regime = as.vector(regime)
)

# Plot each individual's time series with regime-based coloring
ggplot(y_df, aes(x = time, y = value, group = series, color = as.factor(regime))) +
  geom_line(aes(color = as.factor(regime))) +  # Line changes color based on regime
  geom_point(size = 1) +  # Add points to emphasize regime switches
  facet_wrap(~ series, ncol = 1, scales = "free_y") +  # Separate plots for each individual
  labs(title = "Time Series with Regime Switching for Each Individual",
       color = "Regime") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("1" = "blue", "2" = "red", "3" = "green", "4" = "purple"))

# Count occurrences of each regime in the simulated data
regime_counts <- y_df %>%
  group_by(regime) %>%
  summarize(count = n())

# Plot the distribution of regimes
ggplot(regime_counts, aes(x = as.factor(regime), y = count, fill = as.factor(regime))) +
  geom_bar(stat = "identity") +
  labs(title = "Frequency of Each Regime", x = "Regime", y = "Count") +
  theme_minimal()

# Stan model code definition
stan_code <- "
data {
  int<lower=1> N;                  // Number of time series
  int<lower=1> Nt;                 // Length of each time series
  int<lower=1, upper=4> regime[N, Nt];  // Regime index at each time point
  real y[N, Nt];                    // Observation matrix (each time series is a row)
}
parameters {
  vector[4] mu;                    // Mean for each regime
  real<lower=0, upper=1> lambda;   // Smoothing parameter (closer to 0 gives more weight to previous observation)
  real<lower=0, upper=1> ar_pos;  // AR coefficient for regime 1
  real<lower=-1, upper=0> ar_neg;  // AR coefficient for regime 2
  real<lower=-1, upper=1> ma_coef; // MA coefficient for regime 3
  vector<lower=0>[4] sigma;        // Standard deviation for each regime
}
model {
  // Priors for parameters
  mu ~ normal(0, 5);
  lambda ~ beta(2, 2);
  ar_pos ~ normal(0.7, 0.3);
  ar_neg ~ normal(-0.7, 0.3);
  ma_coef ~ normal(0, 0.5);
  sigma ~ cauchy(0, 2);

  // Time series model with regime-dependent AR/MA processes
  for (n in 1:N) {
    for (t in 2:Nt) {
      int k = regime[n, t];
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
  real y_rep[N, Nt-1];
  real log_lik[N, Nt-1];

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

# Compile and sample from the Stan model
fit <- stan(model_code = stan_code, data = stan_data, iter = 2000, chains = 4)

# Extract log-likelihood for PSIS-LOO cross-validation
log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
loo_result <- loo(log_lik, moment_match = TRUE)
print(loo_result)

# Posterior analysis
# Plot posterior distributions of AR and MA coefficients
bayesplot::mcmc_areas(as.array(fit), pars = c("ar_pos", "ar_neg", "ma_coef")) +
  labs(title = "Posterior Distributions of AR and MA Coefficients")

# Plot posterior distributions of sigma (standard deviations for each regime)
bayesplot::mcmc_areas(as.array(fit), pars = c("sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]")) +
  labs(title = "Posterior Distributions of Sigma (Standard Deviation) per Regime")

# Plot posterior distributions of mu (mean of each regime) and lambda (smoothing parameter)
bayesplot::mcmc_areas(as.array(fit), pars = c("mu[1]", "mu[2]", "mu[3]", "mu[4]", "lambda")) +
  labs(title = "Posterior Distributions of Regime Means and Lambda")
