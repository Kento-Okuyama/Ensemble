# Install and load rstan
library(rstan)

# Parameters for simulation
N <- 100   # Number of individuals
Nt <- 50   # Number of time points per individual

phi <- 0.8                   # AR(1) coefficient
sigma_AR <- 1                # AR(1) process variance
sigma_WN <- 0.5              # White noise process variance
beta <- 2                    # Coefficient for logistic function
X2 <- rnorm(N, 0, 1)         # Cross-sectional variable

# Function to generate time-series for each individual
generate_time_series <- function(N, Nt, phi, sigma_AR, sigma_WN, beta, X2) {
  y <- matrix(0, nrow=N, ncol=Nt)   # Store time-series data
  omega <- 1 / (1 + exp(-beta * X2))  # Mixture weights
  
  for (i in 1:N) {
    for (t in 2:Nt) {
      # Generate AR(1) process component
      ar1_component <- phi * y[i, t-1] + rnorm(1, 0, sigma_AR)
      
      # Generate White noise component
      wn_component <- rnorm(1, 0, sigma_WN)
      
      # Combine the two components based on the mixture weight omega[i]
      y[i, t] <- omega[i] * ar1_component + (1 - omega[i]) * wn_component
    }
  }
  return(list(y=y, omega=omega))
}

# Simulate data
sim_data <- generate_time_series(N, Nt, phi, sigma_AR, sigma_WN, beta, X2)
y <- sim_data$y
omega <- sim_data$omega


# Stan model code
stan_code <- "
data {
  int<lower=1> N;        // Number of individuals
  int<lower=1> Nt;       // Number of time points
  matrix[N, Nt] y;       // Time-series data
  vector[N] X2;          // Cross-sectional variable
}

parameters {
  real<lower=0> phi;             // AR(1) coefficient
  real<lower=0> sigma_AR;        // AR(1) process standard deviation
  real<lower=0> sigma_WN;        // White noise standard deviation
  real beta;                     // Logistic regression coefficient
}

model {
  vector[N] omega;
  
  // Logistic function for omega based on X2
  for (i in 1:N) {
    omega[i] = inv_logit(beta * X2[i]);
  }

  // Likelihood
  for (i in 1:N) {
    for (t in 2:Nt) {
      target += log_mix(omega[i],
        normal_lpdf(y[i,t] | phi * y[i,t-1], sigma_AR),
        normal_lpdf(y[i,t] | 0, sigma_WN));
    }
  }
}

generated quantities {
  // Posterior predictions or other quantities of interest
}
"

# Compile the model using rstan
stan_model <- stan_model(model_code = stan_code)

# Prepare data for Stan
stan_data <- list(
  N = N,
  Nt = Nt,
  y = y,
  X2 = X2
)

# Fit the model using rstan
fit <- sampling(stan_model, data = stan_data, chains = 4, iter=2000, warmup=500)

# Print the results
print(fit)
