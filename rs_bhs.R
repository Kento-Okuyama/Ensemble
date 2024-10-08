# Load necessary packages
library(ggplot2)
library(gridExtra)
library(rstan)

# 1. Generate synthetic data with deterministic regime switching based on X_t
set.seed(123)
n <- 100  # Number of time points
X <- numeric(n)  # Hidden state
Y <- numeric(n)  # Observed data

# Threshold to determine regime (for data generation, we still use a fixed value here)
threshold <- 0  # Switch between regimes based on X_t

state <- numeric(n)  # Hidden state (1 for regime 1, 2 for regime 2)

# Initial state is regime 1
state[1] <- 1
X[1] <- rnorm(1, 0, 1)  # Initial state

# Generate data with regime switching based on X_t
for (t in 2:n) {
  if (X[t-1] > threshold) {
    state[t] <- 1  # Regime 1 when X_t > threshold
    X[t] <- 0.5 * X[t-1] + rnorm(1, 0, 1)  # Dynamics in regime 1
  } else {
    state[t] <- 2  # Regime 2 when X_t <= threshold
    X[t] <- -0.8 * X[t-1] + rnorm(1, 0, 1)  # Dynamics in regime 2
  }
  # Measurement model
  Y[t] <- 1.0 * X[t] + rnorm(1, 0, 0.5)  # Observation noise
}

# 2. Visualize both the actual X and predicted X (Stan results) over time with regime coloring

# Prepare data for Stan
stan_data <- list(N = n, Y = Y)

# 3. Stan model for deterministic regime switching with threshold estimation
stan_code <- "
data {
  int<lower=1> N;       // Number of observations
  vector[N] Y;          // Observed data
}
parameters {
  vector[N] X;          // Hidden states
  real<lower=0> sigma;  // Measurement noise standard deviation
  real alpha;           // Coefficient for regime 1
  real beta;            // Coefficient for regime 2
  real<lower=0> tau;    // Transition noise standard deviation
  real threshold;       // Threshold for regime switching (to be estimated)
}
model {
  // Priors
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  tau ~ normal(0, 1);
  threshold ~ normal(0, 1);  // Prior for the threshold

  // Initial state
  X[1] ~ normal(0, 1);

  // Regime-switching dynamics based on the estimated threshold
  for (t in 2:N) {
    if (X[t-1] > threshold) {  // Regime 1 when X[t-1] > threshold
      X[t] ~ normal(alpha * X[t-1], tau);  // Regime 1 dynamics
    } else {  // Regime 2 when X[t-1] <= threshold
      X[t] ~ normal(beta * X[t-1], tau);   // Regime 2 dynamics
    }
  }

  // Measurement model
  Y ~ normal(X, sigma);
}
"

# Compile the Stan model
stan_model <- stan_model(model_code = stan_code)

# Fit the model using Stan
fit <- sampling(stan_model, data = stan_data, iter = 2000, chains = 4)

# Extract the estimated hidden states and threshold from the Stan model
hidden_states <- extract(fit)$X
threshold_estimated <- extract(fit)$threshold  # Extract estimated threshold

# Create a data frame for hidden states and their estimated values
df_hidden <- data.frame(time = 1:n, X_estimated = apply(hidden_states, 2, mean))

# Assign regime information based on the estimated threshold
df_hidden$state <- ifelse(df_hidden$X_estimated > mean(threshold_estimated), 1, 2)

# 4. Plot both actual and predicted X with regime coloring
df <- data.frame(time = 1:n, X = X, Y = Y, state = factor(state))

# Plot for actual X with regime coloring
plot_X_actual <- ggplot(df, aes(x = time, y = X, color = state, group = 1)) +
  geom_line(size = 1) +  # Use one line for X
  labs(title = "Time vs Actual X (Regime Switching)", x = "Time", y = "X") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")  # Set legend name to "Regime"

# Plot for predicted X (Stan results) with regime coloring
plot_X_predicted <- ggplot(df_hidden, aes(x = time, y = X_estimated, color = factor(state), group = 1)) +
  geom_line(size = 1) +  # Use one line for predicted X
  labs(title = "Time vs Predicted X (Regime Switching)", x = "Time", y = "X_estimated") +
  scale_color_manual(values = c("red", "blue"), name = "Regime")  # Set legend name to "Regime"

# Arrange both plots in a grid
grid.arrange(plot_X_actual, plot_X_predicted, ncol = 1)
