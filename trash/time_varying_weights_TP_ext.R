# Load necessary libraries
library(rstan)
library(bayesplot)
library(loo)
library(ggplot2)
library(dplyr)

# Set the seed for reproducibility
set.seed(123)

# Parameters for data simulation
N <- 10    # Number of time series
Nt <- 50  # Length of each time series
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
for (n in 1:N) {
  current_regime <- 1 # sample(1:4, 1)  
  y[n, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])  # Initial value
  regime[n, 1] <- current_regime
  
  for (t in 2:Nt) {
    current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
    regime[n, t] <- current_regime
    
    target_mean <- lambda * mu[current_regime] + (1 - lambda) * y[n, t - 1]
    
    if (current_regime == 1) {
      y[n, t] <- (1 - ar_pos) * target_mean + ar_pos * y[n, t-1] + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else if (current_regime == 2) {
      y[n, t] <- (1 - ar_neg) * target_mean + ar_neg * y[n, t-1] + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else if (current_regime == 3) {
      y[n, t] <- target_mean + ma_coef * (y[n, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
    } else {
      y[n, t] <- rnorm(1, mean = target_mean, sd = sigma[current_regime])
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

# Prepare data list for Stan
stan_data <- list(
  N = N,
  Nt = Nt,
  regime = regime,
  y = y
)

# Stan model code
stan_code <- "
data {
  int<lower=1> N;                   // Number of individuals
  int<lower=1> Nt;                  // Time series length for each individual
  real y[N, Nt];                    // Observed data
}
parameters {
  vector[4] mu;                     // Mean for each state
  real<lower=0, upper=1> lambda;    // Smoothing parameter for target mean
  real<lower=0, upper=1> ar_pos;    // AR coefficient for state 1
  real<lower=-1, upper=0> ar_neg;   // AR coefficient for state 2
  real<lower=-1, upper=1> ma_coef;  // MA coefficient for state 3
  vector<lower=0>[4] sigma;         // Standard deviation for each state
  simplex[4] T[4];                  // Transition probability matrix
  simplex[4] w1;                    // Initial weights (to be sampled)
}
transformed parameters {
  simplex[4] w[Nt];               // Dynamic weights
  
  // Set the initial value of w
  w[1] = w1;

  // Dynamically update weights based on transition probabilities
  for (t in 2:Nt) {
    for (j in 1:4) {
      w[t][j] = T[1][j] * w[t - 1][1] +
                T[2][j] * w[t - 1][2] +
                T[3][j] * w[t - 1][3] +
                T[4][j] * w[t - 1][4];
    }
    // Normalize to maintain simplex properties
    w[t] = w[t] / sum(w[t]);
  }
}
model {
  // Priors for parameters
  mu ~ normal(0, 5);
  lambda ~ beta(2, 2);
  ar_pos ~ normal(0.7, 0.3);
  ar_neg ~ normal(-0.7, 0.3);
  ma_coef ~ normal(0, 0.5);
  sigma ~ cauchy(0, 2);

  // Prior for transition matrix
  for (i in 1:4) {
    vector[4] T_prior = rep_vector(0.1, 4);
    T_prior[i] = 0.8;
    T[i] ~ dirichlet(T_prior);
  }

  // Prior for initial weights
  vector[4] w_prior = rep_vector(0.25, 4);  // Example prior (balanced weights)
  w1 ~ dirichlet(w_prior);

  // Likelihood for observed data
  for (n in 1:N) {
    for (t in 2:Nt) {
      real target_mean = lambda * dot_product(w[t], mu) + (1 - lambda) * y[n, t - 1];
      vector[4] y_pred;

      // State-specific predictions
      y_pred[1] = (1 - ar_pos) * target_mean + ar_pos * y[n, t-1];
      y_pred[2] = (1 - ar_neg) * target_mean + ar_neg * y[n, t-1];
      y_pred[3] = target_mean + ma_coef * (y[n, t - 1] - target_mean);
      y_pred[4] = target_mean;

      // Weighted prediction based on dynamic weights
      real weighted_prediction = dot_product(w[t], y_pred);
      real variance_weighted = dot_product(w[t], sigma);

      // Observation model
      y[n, t] ~ normal(weighted_prediction, variance_weighted);
    }
  }
}
generated quantities {
  real weighted_prediction;
  real variance_weighted;
  matrix[N, Nt-1] y_rep;             // Posterior predictive samples
  matrix[N, Nt-1] log_lik;           // Log-likelihood for LOO
  for (n in 1:N) {
    for (t in 2:Nt) {
      real target_mean = lambda * dot_product(w[t], mu) + (1 - lambda) * y[n, t - 1];
      vector[4] y_pred;
      y_pred[1] = (1 - ar_pos) * target_mean + ar_pos * y[n, t-1];
      y_pred[2] = (1 - ar_neg) * target_mean + ar_neg * y[n, t-1];
      y_pred[3] = target_mean + ma_coef * (y[n, t-1] - target_mean);
      y_pred[4] = target_mean;

      weighted_prediction = dot_product(w[t], y_pred);
      variance_weighted = dot_product(w[t], sigma);
      
      y_rep[n, t-1] = normal_rng(weighted_prediction, variance_weighted);
      log_lik[n, t-1] = normal_lpdf(y[n, t] | weighted_prediction, variance_weighted);
    }
  }
}
"
# Compile and sample from the Stan model
fit <- stan(model_code = stan_code, data = stan_data, iter = 2000, chains = 4)

# Compile and sample the Stan model
# fit <- stan(
#   model_code = stan_code,
#   data = stan_data,
#   iter = 4000,             # Increase number of iterations to improve convergence and ESS
#   chains = 4,
#   control = list(
#     adapt_delta = 0.99,    # Set adapt_delta to 0.99 to reduce divergent transitions
#     max_treedepth = 15     # Increase max_treedepth to 15 for better exploration of parameter space
#   )
# )

# Extract log-likelihood for PSIS-LOO cross-validation
log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
loo_result <- loo(log_lik, moment_match = TRUE)
print(loo_result)

# Posterior analysis of dynamic weights
bayesplot::mcmc_areas(as.array(fit), pars = c("ar_pos", "ar_neg", "ma_coef")) +
  labs(title = "Posterior distributions of AR and MA coefficients")

bayesplot::mcmc_areas(as.array(fit), pars = c("sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]")) +
  labs(title = "Posterior distributions of state standard deviations")

bayesplot::mcmc_areas(as.array(fit), pars = c("mu[1]", "mu[2]", "mu[3]", "mu[4]", "lambda")) +
  labs(title = "Posterior distributions of state means and smoothing parameter")

bayesplot::mcmc_areas(as.array(fit), pars = c("T[1,1]", "T[2,2]", "T[3,3]", "T[4,4]")) +
  labs(title = "Posterior distributions of transition matrix diagonal elements")

# Check posterior distribution of transition matrix
print(fit, pars = "T")
