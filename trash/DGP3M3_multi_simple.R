# Install and load required packages
if (!require("rstan")) install.packages("rstan", dependencies=TRUE)
if (!require("loo")) install.packages("loo", dependencies=TRUE)
library(rstan)
library(loo)

# Set Stan options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Data preparation
set.seed(123)
Nt <- 30  # Length of the time series
Ni <- 15  # Number of individuals
Y <- matrix(NA, Ni, Nt)  # Time series data for each individual
C <- sample(1:3, Ni, replace = TRUE)  # Latent category for each individual

# Simulation parameters
phi1_true <- 0.8
phi2_true <- 0.5
theta1_true <- 0.3
sigma_true <- 1.0

# Generate time series data
for (i in 1:Ni) {
  Y[i, 1:2] <- rnorm(2)  # Initial values
  for (t in 3:Nt) {
    if (C[i] == 1) { # MA(1) model
      Y[i, t] <- theta1_true * (Y[i, t-1] - mean(Y[i, 1:(t-1)])) + rnorm(1, 0, sigma_true)
    } else if (C[i] == 2) { # AR(2) model
      Y[i, t] <- phi1_true * Y[i, t-1] + phi2_true * Y[i, t-2] + rnorm(1, 0, sigma_true)
    } else if (C[i] == 3) { # ARMA(2,2) model
      Y[i, t] <- phi1_true * Y[i, t-1] + phi2_true * Y[i, t-2] +
        theta1_true * (Y[i, t-1] - mean(Y[i, 1:(t-1)])) + rnorm(1, 0, sigma_true)
    }
  }
}

# Prepare data for Stan
stan_data <- list(Nt = Nt, Ni = Ni, Y = Y)

# Stan model code for MA(1)
ma1_stan_code <- "
data {
  int<lower=0> Nt;
  int<lower=0> Ni;
  vector[Nt] Y[Ni];
}
parameters {
  real<lower=0> sigma;
  real theta1;
}
model {
  sigma ~ normal(0, 2);
  theta1 ~ normal(0, 1);
  for (i in 1:Ni) {
    for (t in 2:Nt) {
      Y[i][t] ~ normal(theta1 * (Y[i][t-1] - mean(Y[i][1:(t-1)])), sigma);
    }
  }
}
generated quantities {
  vector[Nt-2] log_lik[Ni];
  for (i in 1:Ni) {
    for (t in 3:Nt) {
      log_lik[i, t-2] = normal_lpdf(Y[i][t] | theta1 * (Y[i][t-1] - mean(Y[i][1:(t-1)])), sigma);
    }
  }
}
"

# Stan model code for AR(2)
ar2_stan_code <- "
data {
  int<lower=0> Nt;
  int<lower=0> Ni;
  vector[Nt] Y[Ni];
}
parameters {
  real<lower=0> sigma;
  real phi1;
  real phi2;
}
model {
  sigma ~ normal(0, 2);
  phi1 ~ normal(0, 1);
  phi2 ~ normal(0, 1);
  for (i in 1:Ni) {
    for (t in 3:Nt) {
      Y[i][t] ~ normal(phi1 * Y[i][t-1] + phi2 * Y[i][t-2], sigma);
    }
  }
}
generated quantities {
  vector[Nt-2] log_lik[Ni];
  for (i in 1:Ni) {
    for (t in 3:Nt) {
      log_lik[i, t-2] = normal_lpdf(Y[i][t] | phi1 * Y[i][t-1] + phi2 * Y[i][t-2], sigma);
    }
  }
}
"

# Stan model code for ARMA(2,2)
arma22_stan_code <- "
data {
  int<lower=0> Nt;
  int<lower=0> Ni;
  vector[Nt] Y[Ni];
}
parameters {
  real<lower=0> sigma;
  real phi1;
  real phi2;
  real theta1;
}
model {
  sigma ~ normal(0, 2);
  phi1 ~ normal(0, 1);
  phi2 ~ normal(0, 1);
  theta1 ~ normal(0, 1);
  for (i in 1:Ni) {
    for (t in 3:Nt) {
      Y[i][t] ~ normal(phi1 * Y[i][t-1] + phi2 * Y[i][t-2] + theta1 * (Y[i][t-1] - mean(Y[i][1:(t-1)])), sigma);
    }
  }
}
generated quantities {
  vector[Nt-2] log_lik[Ni];
  for (i in 1:Ni) {
    for (t in 3:Nt) {
      log_lik[i, t-2] = normal_lpdf(Y[i][t] | phi1 * Y[i][t-1] + phi2 * Y[i][t-2] + theta1 * (Y[i][t-1] - mean(Y[i][1:(t-1)])), sigma);
    }
  }
}
"

# List of Stan model codes and model names
model_codes <- list(ma1_stan_code, ar2_stan_code, arma22_stan_code)
model_names <- c("MA(1)", "AR(2)", "ARMA(2,2)")
loo_results <- list()

# Fit each Stan model and calculate LOO-CV
for (i in 1:3) {
  fit_single <- stan(model_code = model_codes[[i]], data = stan_data,
                     iter = 2000, warmup = 1000, chains = 4,
                     control = list(adapt_delta = 0.8, max_treedepth = 10))
  
  # LOO-CV calculation
  log_lik <- extract_log_lik(fit_single)
  loo_result <- loo(log_lik)
  loo_results[[model_names[i]]] <- loo_result
  
  # Display results
  cat("\nLOO-CV for model", model_names[i], ":\n")
  print(loo_result)
}

# Compare LOO-CV results across models
loo_compare(loo_results)
