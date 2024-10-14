stan_model_code <- "
data {
  int<lower=0> N;  // Number of time points
  int<lower=0> I;  // Number of individuals
  matrix[I, N] Y;  // Observed data (matrix of I individuals and N time points)
}
parameters {
  real phi_pos;  // AR(1) coefficient for the positive model
  real phi_neg;  // AR(1) coefficient for the negative model
  real<lower=0> sigma_eta;  // Variance of the latent state noise
  real<lower=0> sigma_eps;  // Observation noise
  real<lower=0, upper=1> p_switch;  // Probability of switching between regimes
  matrix[I, N] X;  // Latent states (to be estimated)
}
model {
  // Priors
  phi_pos ~ normal(0, 1) T[0, 1];  // Prior for positive AR(1) coefficient
  phi_neg ~ normal(0, 1) T[-1, 0];  // Prior for negative AR(1) coefficient
  sigma_eta ~ normal(0, 1);  // Prior for latent state noise variance
  sigma_eps ~ normal(0, 1);  // Prior for observation noise variance
  p_switch ~ beta(2, 2);  // Prior for switching probability

  // Likelihood for latent states
  for (i in 1:I) {
    X[i, 1] ~ normal(0, sigma_eta);  // Initial state
    for (t in 2:N) {
      // Mixture of AR(1) processes based on switching probability
      target += log_mix(p_switch,
                        normal_lpdf(X[i, t] | phi_neg * X[i, t-1], sigma_eta),
                        normal_lpdf(X[i, t] | phi_pos * X[i, t-1], sigma_eta));
    }
  }

  // Observation model
  for (i in 1:I) {
    for (t in 1:N) {
      Y[i, t] ~ normal(X[i, t], sigma_eps);
    }
  }
}
"

# Load necessary packages
library(rstan)

# Generate data (using existing simulation data)
set.seed(123)
n_individuals <- 10  # Number of individuals
n_time <- 30  # Length of the time series
phi_pos <- 0.8  # AR(1) coefficient (for the positive regime)
phi_neg <- -0.8  # AR(1) coefficient (for the negative regime)
sigma_eta <- 1  # Variance of the noise
sigma_eps <- 0.5  # Observation noise
p_switch <- 0.1  # Probability of switching between regimes

# Generate data
X <- matrix(0, n_individuals, n_time)
Y <- matrix(0, n_individuals, n_time)
S <- matrix(0, n_individuals, n_time)

for (i in 1:n_individuals) {
  S[i, 1] <- sample(c(0, 1), 1)  # Initial regime
  X[i, 1] <- rnorm(1, 0, sigma_eta)
  
  for (t in 2:n_time) {
    if (runif(1) < p_switch) {
      S[i, t] <- 1 - S[i, t-1]  # Regime switching
    } else {
      S[i, t] <- S[i, t-1]  # Maintain regime
    }
    
    if (S[i, t] == 0) {
      X[i, t] <- phi_pos * X[i, t-1] + rnorm(1, 0, sigma_eta)
    } else {
      X[i, t] <- phi_neg * X[i, t-1] + rnorm(1, 0, sigma_eta)
    }
  }
  
  Y[i, ] <- X[i, ] + rnorm(n_time, 0, sigma_eps)
}

# Ensure that S is included in the list passed to Stan
stan_data <- list(
  N = n_time,
  I = n_individuals,
  Y = Y
)

# Compile the Stan model
stan_model <- stan_model(model_code = stan_model_code)

# Run sampling in Stan
fit <- sampling(stan_model, data = stan_data, iter = 2000, chains = 4)

# Extract estimated parameters
posterior_samples <- rstan::extract(fit)

# Check the estimated AR(1) coefficients
phi_pos_est <- posterior_samples$phi_pos
phi_neg_est <- posterior_samples$phi_neg
sigma_eta_est <- posterior_samples$sigma_eta
sigma_eps_est <- posterior_samples$sigma_eps

# Visualize results
hist(phi_pos_est, main = "Posterior distribution of phi_pos", xlab = "phi_pos")
hist(phi_neg_est, main = "Posterior distribution of phi_neg", xlab = "phi_neg")
hist(sigma_eta_est, main = "Posterior distribution of sigma_eta", xlab = "sigma_eta")
hist(sigma_eps_est, main = "Posterior distribution of sigma_eps", xlab = "sigma_eps")

# Extract posterior samples of X from Stan results
X_samples <- posterior_samples$X

# Extract the posterior samples of the latent variable X at the last time point (N)
X_last_samples <- X_samples[, , n_time]

# True latent variable X at the last time point from the simulation
X_true_last <- X[, n_time]

# Plot histograms of posterior distributions for each individual, marking the true X
par(mfrow = c(2, 5))  # 2 rows and 5 columns for 10 graphs

for (i in 1:n_individuals) {
  # Plot the posterior distribution of the latent state at the last time point for individual i
  hist(X_last_samples[, i], breaks = 30, main = paste("Individual", i),
       xlab = "Latent state (X)", col = "lightblue", border = "white")
  
  # Mark the true X value with a red vertical line
  abline(v = X_true_last[i], col = "red", lwd = 2, lty = 2)
}

