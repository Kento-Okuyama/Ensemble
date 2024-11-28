# Required packages
library(rstan)
library(loo)

# Stan options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Stan model code directly in R
stan_code <- "
data {
  int<lower=0> Nt;           // Number of time points
  vector[Nt] Y;              // Time series data
  int<lower=1,upper=3> C[Nt]; // Latent categorical variable (1=AR1, 2=AR2, 3=ARMA22)
}

parameters {
  real<lower=0> sigma;     // Error term standard deviation
  vector[3] alpha;         // Unbounded parameters for softmax (model weights)
  simplex[3] w;            // Model weights
  
  real phi1;               // AR(1), AR(2), ARMA(2,2) coefficient 1
  real phi2;               // AR(2), ARMA(2,2) coefficient 2
  real theta1;             // MA(1), ARMA(2,2) coefficient 1
  real theta2;             // ARMA(2,2) coefficient 2
}

model {
  vector[Nt] mu;           // Latent variable for the mean process

  // Priors
  alpha ~ normal(0, 5);    // Prior for alpha (softmax)
  phi1 ~ normal(0, 1);     // Prior for AR/ARMA coefficients
  phi2 ~ normal(0, 1);     // Prior for AR(2) and ARMA(2,2) coefficient
  theta1 ~ normal(0, 1);   // Prior for MA(1) and ARMA(2,2) coefficient
  theta2 ~ normal(0, 1);   // Prior for ARMA(2,2) coefficient
  sigma ~ normal(0, 2);    // Prior for error standard deviation

  // Initialize mu (first two values set to zero to avoid NaN)
  mu[1] = 0; 
  mu[2] = 0;

  // Latent process model depending on the latent categorical variable (C)
  for (t in 3:Nt) {
    if (C[t] == 1) {       // AR(1) model
      mu[t] = phi1 * Y[t-1];
    } else if (C[t] == 2) { // AR(2) model
      mu[t] = phi1 * Y[t-1] + phi2 * Y[t-2];
    } else if (C[t] == 3) { // ARMA(2,2) model
      mu[t] = phi1 * Y[t-1] + phi2 * Y[t-2] 
              + theta1 * (Y[t-1] - mu[t-1]) + theta2 * (Y[t-2] - mu[t-2]);
    }
  }

  // Likelihood of observed data given latent process
  Y[3:Nt] ~ normal(mu[3:Nt], sigma);
}

generated quantities {
  vector[Nt-2] log_lik;
  vector[Nt] mu;

  // Initialize mu safely
  mu[1] = 0;
  mu[2] = 0;

  for (t in 3:Nt) {
    if (C[t] == 1) {
      mu[t] = phi1 * Y[t-1];
      log_lik[t-2] = normal_lpdf(Y[t] | mu[t], sigma);
    } else if (C[t] == 2) {
      mu[t] = phi1 * Y[t-1] + phi2 * Y[t-2];
      log_lik[t-2] = normal_lpdf(Y[t] | mu[t], sigma);
    } else if (C[t] == 3) {
      mu[t] = phi1 * Y[t-1] + phi2 * Y[t-2] +
              theta1 * (Y[t-1] - mu[t-1]) + theta2 * (Y[t-2] - mu[t-2]);
      log_lik[t-2] = normal_lpdf(Y[t] | mu[t], sigma);
    }
  }
}
"

# Simulated time series data
set.seed(123)               # Set seed for reproducibility
Nt <- 100                   # Length of time series
Y <- numeric(Nt)            # Time series data
C <- sample(1:3, Nt, replace = TRUE)  # Latent categorical variable (1=AR1, 2=AR2, 3=ARMA22)

# Simulate data using AR(1), AR(2), and ARMA(2,2) models
phi1_true <- 0.8            # True AR(1) coefficient
sigma_true <- 1.0           # True noise standard deviation
Y[1:2] <- rnorm(2)          # Initial values

# Generate the time series based on the latent model (C)
for (t in 3:Nt) {
  if (C[t] == 1) {          # AR(1) model
    Y[t] <- phi1_true * Y[t-1] + rnorm(1, 0, sigma_true);
  } else if (C[t] == 2) {   # AR(2) model
    Y[t] <- phi1_true * Y[t-1] + 0.5 * Y[t-2] + rnorm(1, 0, sigma_true);
  } else if (C[t] == 3) {   # ARMA(2,2) model
    Y[t] <- phi1_true * Y[t-1] + 0.5 * Y[t-2] 
    + 0.3 * (Y[t-1] - Y[t-2]) + rnorm(1, 0, sigma_true);
  }
}

# Prepare data for Stan
stan_data <- list(Nt = Nt, Y = Y, C = C)

# Compile and fit the Stan model
fit <- stan(model_code = stan_code, data = stan_data, 
            iter = 1000, chains = 4, control = list(adapt_delta = 0.99, max_treedepth = 15))

# Print the results for key parameters
print(fit, pars = c("phi1", "phi2", "theta1", "theta2", "sigma", "w"))

# Posterior predictive checks
post_pred <- extract(fit, "log_lik")  # Extract log likelihood
loo_result <- loo(post_pred$log_lik)  # Compute LOO (leave-one-out cross-validation)
print(loo_result)

# Extract posterior samples
samples <- extract(fit)               # Extract all samples
weights <- samples$w                  # Extract model weights

# Visualize model weights using a boxplot
boxplot(weights, main="Posterior Model Weights", names=c("AR(1)", "AR(2)", "ARMA(2,2)"))
