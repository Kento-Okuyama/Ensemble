# Load necessary libraries
library(rstan)
library(loo)

# Data generation parameters
set.seed(123)
N <- 10    # Number of time series
Nt <- 50   # Length of each time series
mu <- c(2, -2, 1, 0)
sigma <- c(0.5, 0.7, 0.6, 0.4)
transition_matrix <- matrix(c(
  0.8, 0.05, 0.1, 0.05,
  0.1, 0.8, 0.05, 0.05,
  0.1, 0.05, 0.8, 0.05,
  0.1, 0.05, 0.1, 0.75
), nrow = 4, byrow = TRUE)

# Generate data
generate_data <- function(N, Nt, mu, sigma, transition_matrix) {
  y <- matrix(NA, nrow = N, ncol = Nt)
  regimes <- matrix(NA, nrow = N, ncol = Nt)
  for (n in 1:N) {
    current_regime <- 1
    y[n, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])
    regimes[n, 1] <- current_regime
    for (t in 2:Nt) {
      current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
      regimes[n, t] <- current_regime
      y[n, t] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])
    }
  }
  list(y = y, regimes = regimes)
}

data <- generate_data(N, Nt, mu, sigma, transition_matrix)
y <- data$y

# Prepare data for Stan
stan_data <- list(
  N = N,
  Nt = Nt,
  y = y
)

# Stan model for AR(1), MA(1), White Noise
stan_model_code <- "
data {
  int<lower=1> N;          // Number of time series
  int<lower=1> Nt;         // Length of each time series
  real y[N, Nt];           // Observed data
}
parameters {
  real<lower=0, upper=1> ar_coef;  // AR(1) coefficient
  real<lower=-1, upper=1> ma_coef; // MA(1) coefficient
  real mu;                         // Mean for White Noise
  real<lower=0> sigma;             // Standard deviation
}
model {
  // Priors
  ar_coef ~ beta(2, 2);
  ma_coef ~ normal(0, 0.5);
  mu ~ normal(0, 5);
  sigma ~ cauchy(0, 2);

  // Likelihood
  for (n in 1:N) {
    for (t in 2:Nt) {
      real prediction_ar = ar_coef * y[n, t - 1];
      real prediction_ma = mu + ma_coef * (y[n, t - 1] - mu);
      real prediction_wn = mu;

      // Fit each model separately
      y[n, t] ~ normal(prediction_ar, sigma);  // AR(1)
      y[n, t] ~ normal(prediction_ma, sigma);  // MA(1)
      y[n, t] ~ normal(prediction_wn, sigma);  // White Noise
    }
  }
}
generated quantities {
  real y_pred_ar[N, Nt-1];
  real y_pred_ma[N, Nt-1];
  real y_pred_wn[N, Nt-1];
  for (n in 1:N) {
    for (t in 2:Nt) {
      y_pred_ar[n, t-1] = normal_rng(ar_coef * y[n, t - 1], sigma);
      y_pred_ma[n, t-1] = normal_rng(mu + ma_coef * (y[n, t - 1] - mu), sigma);
      y_pred_wn[n, t-1] = normal_rng(mu, sigma);
    }
  }
}
"

# Fit the model
fit <- stan(model_code = stan_model_code, data = stan_data, iter = 2000, chains = 4)

# Extract predictions
y_pred_ar <- extract(fit, pars = "y_pred_ar")$y_pred_ar
y_pred_ma <- extract(fit, pars = "y_pred_ma")$y_pred_ma
y_pred_wn <- extract(fit, pars = "y_pred_wn")$y_pred_wn

# Combine model predictions
f1 <- apply(y_pred_ar, c(2, 3), mean)
f2 <- apply(y_pred_ma, c(2, 3), mean)
f3 <- apply(y_pred_wn, c(2, 3), mean)
f <- array(c(f1, f2, f3), dim = c(N, Nt, 3))

# Prepare data for the transition model
stan_transition_data <- list(
  N = N,
  Nt = Nt,
  J = 3,
  f = f,
  y = y
)

# Stan model for transition matrix and weights
stan_transition_code <- "
data {
  int<lower=1> N;                   // Number of individuals
  int<lower=1> Nt;                  // Time series length for each individual
  int<lower=1> J;                   // Number of models
  real y[N, Nt];                    // Observed data
  real f[N, Nt, J];                 // Model predictions
}
parameters {
  simplex[J] w1;                    // Initial weights
  simplex[J] T[J];                  // Transition probability matrix
  real<lower=0> sigma;              // Observation noise
}
transformed parameters {
  simplex[J] w[Nt];                 // Dynamic weights for each time step

  w[1] = w1;

  // Dynamically update weights based on transition probabilities
  for (t in 2:Nt) {
    w[t] = rep_vector(0.0, J);      // Initialize weights
    for (j in 1:J) {
      for (k in 1:J) {
        w[t][j] += T[k][j] * w[t - 1][k];
      }
    }
    // Normalize to maintain simplex properties
    w[t] = w[t] / sum(w[t]);
  }
}
model {
  // Priors
  for (i in 1:J) {
    vector[J] T_prior = rep_vector(0.1, J);
    T_prior[i] = 0.8;
    T[i] ~ dirichlet(T_prior);
  }
  w1 ~ dirichlet(rep_vector(1.0, J));
  sigma ~ cauchy(0, 2);

  // Likelihood
  for (n in 1:N) {
    for (t in 1:Nt) {
      real weighted_prediction = 0;
      for (j in 1:J) {
        weighted_prediction += w[t][j] * f[n, t, j];
      }
      y[n, t] ~ normal(weighted_prediction, sigma);
    }
  }
}
generated quantities {
  matrix[N, Nt-1] log_lik;
  for (n in 1:N) {
    for (t in 2:Nt) {
      real weighted_prediction = 0;
      for (j in 1:J) {
        weighted_prediction += w[t][j] * f[n, t, j];
      }
      log_lik[n, t-1] = normal_lpdf(y[n, t] | weighted_prediction, sigma);
    }
  }
}
"

# Fit the transition model
fit_transition <- stan(model_code = stan_transition_code, data = stan_transition_data, iter = 2000, chains = 4)

# Extract log-likelihood and compute PSIS-LOO
log_lik <- extract_log_lik(fit_transition, parameter_name = "log_lik", merge_chains = FALSE)
loo_result <- loo(log_lik, moment_match = TRUE)

# Display results
print(fit_transition, pars = c("T", "w1", "sigma"))
print(loo_result)
