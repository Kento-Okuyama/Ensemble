# Data for the Stan model
stan_data <- list(
  N = N,         # Number of samples
  T = Nt,        # Number of time steps
  y = y_1it,     # Observed data (y_1it)
  y2 = y_2i,     # Individual-specific covariates (y_2i)
  S0 = S_i0      # Initial state
)

# Stan model as a string
state_switching_model = "
data {
  int<lower=0> N;          // Number of samples (individuals)
  int<lower=0> T;          // Number of time steps
  matrix[N, T] y;          // Observed data (y_1it)
  vector[N] y2;            // Individual-specific covariates (y_2i)
  int S0[N];               // Initial state (S_i0)
}

parameters {
  real c[2];               // Intercepts for each state (c^0, c^1)
  real phi[2];             // Autoregressive coefficients for each state (φ^0, φ^1)
  real<lower=0> sigma[2];  // Standard deviation for each state (σ^0, σ^1)
  matrix[N, T] alpha;      // State transition parameters
}

model {
  // Priors
  c ~ normal(0, 1);
  phi ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  
  // Apply normal distribution to each element of alpha
  for (i in 1:N) {
    for (t in 1:T) {
      alpha[i, t] ~ normal(0, 1);
    }
  }

  for (i in 1:N) {
    for (t in 2:T) {
      // Compute the transition probability for the previous time step
      real p1 = inv_logit(alpha[i, t-1]);
      
      // Generate observations based on the state (using probabilities)
      if (S0[i] == 0) {
        y[i, t] ~ normal(c[1] + phi[1] * y[i, t-1], sigma[1]);
      } else {
        y[i, t] ~ normal(c[2] + phi[2] * y[i, t-1], sigma[2]);
      }
    }
  }
}

generated quantities {
  // Sampling S_it based on the generated data
  int S_it[N, T];
  
  for (i in 1:N) {
    for (t in 2:T) {
      real p1 = inv_logit(alpha[i, t-1]);
      S_it[i, t] = bernoulli_rng(p1);
    }
  }
}
"

# Compile the Stan model
stan_model <- stan_model(model_code = state_switching_model)

# Perform MCMC sampling
fit <- sampling(stan_model, data = stan_data, iter = 1000, chains = 4)

# Display the results
print(fit)

# Visualize the results
plot(fit)
