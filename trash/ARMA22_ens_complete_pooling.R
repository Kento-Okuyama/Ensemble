# Load required libraries
library(forecast)
library(rstan)
library(loo)
library(ggplot2)

# Define parameters for data generation
n_people <- 10
n_burnins <- 25
n_timepoints <- 50

# Initialize list to store time series data for each person
time_series_list <- list()

# Generate ARMA(2,2) time series data for each person
set.seed(123)  # Set random seed for reproducibility
for (i in 1:n_people) {
  ar <- c(0.5, -0.3)  # Autoregressive coefficients
  ma <- c(0.4, -0.2)  # Moving average coefficients
  mu <- 0             # Mean of the process
  sigma <- 1          # Standard deviation of the process
  e <- rnorm(n_burnins + n_timepoints, mean = mu, sd = sigma)  # White noise
  data <- numeric(n_burnins + n_timepoints)  # Initialize data vector
  data[1:2] <- e[1:2]                # Set initial values
  
  # Generate ARMA(2,2) process
  for (t in 3:(n_burnins + n_timepoints)) {
    data[t] <- mu + ar[1] * (data[t - 1] - mu) + 
      ar[2] * (data[t - 2] - mu) + 
      e[t] + ma[1] * e[t - 1] + ma[2] * e[t - 2]
  }
  
  # Store time series data after discarding burn-in period
  time_series_list[[i]] <- data[(n_burnins + 1):(n_burnins + n_timepoints)]
}

# Visualize the generated time series data
plot(1:n_timepoints, type = 'n',
     xlim = c(1, n_timepoints), ylim = range(unlist(time_series_list)),
     main = "ARMA(2,2) Time Series for All Individuals",
     xlab = "Time", ylab = "Value")
colors <- rainbow(n_people)  # Create a color palette for each individual
for (i in 1:n_people) {
  lines(1:n_timepoints, time_series_list[[i]], col = colors[i], lwd = 1)
}

# Set rstan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# --- Prepare data for Stan ---
# Get AR(p) model predictions for each p = 1, 2, 3
models <- c(1, 2, 3)  
y_hat_models <- lapply(models, function(p) {
  do.call(rbind, lapply(time_series_list, function(ts) 
    fitted(Arima(ts, order = c(p, 0, 0)))))
})

# Prepare the data list for Stan
data_list <- list(
  N_people = n_people,
  N_timepoints = n_timepoints,
  y = do.call(rbind, time_series_list),
  N_models = length(models),
  y_hat_models = y_hat_models
)

# Define the Stan model for complete pooling (ensure it is named correctly)
stan_model_ens_complete_pooling <- stan_model(model_code = "
data {
    int<lower=1> N_people;          // Number of individuals
    int<lower=1> N_timepoints;      // Number of time points
    vector[N_timepoints] y[N_people];          // Observed data
    int<lower=1> N_models; //number of models
    vector[N_timepoints] y_hat_models[N_models, N_people]; // Predictions from each model
}
parameters {
    simplex[N_models] w;            // Single set of weights for all time points and individuals
    real<lower=0> sigma;          // Standard deviation of observation noise
}
model {
    // Prior distributions
    sigma ~ cauchy(0, 2.5);

    // Observation model
    for (j in 1:N_people) {
        for (t in 1:N_timepoints) {
            real y_pred = 0;
            for (m in 1:N_models){
                y_pred = y_pred + w[m] * y_hat_models[m, j, t]; // Summing over models
            }
            y[j, t] ~ normal(y_pred, sigma);
        }
    }
}
generated quantities {
    vector[N_timepoints] log_lik[N_people]; // Log-likelihood for each data point
    vector[N_timepoints] y_hat[N_people];   // Posterior predictive distributions
    for (j in 1:N_people) {
        for (t in 1:N_timepoints) {
            real y_pred = 0;
            for (m in 1:N_models){
                y_pred = y_pred + w[m] * y_hat_models[m, j, t];
            }
            log_lik[j][t] = normal_lpdf(y[j][t] | y_pred, sigma);
            y_hat[j][t] = normal_rng(y_pred, sigma);
        }
    }
}
")


# Fit the Stan models
fit_combined <- sampling(
  stan_model_ens_complete_pooling,
  data = data_list,
  iter = 2000,
  warmup = 1000,
  chains = 4,
  control = list(max_treedepth = 15, adapt_delta = 0.99), 
  init = list(
    list(w = rep(1/3, 3), sigma = 1),
    list(w = rep(1/3, 3), sigma = 1),
    list(w = rep(1/3, 3), sigma = 1),
    list(w = rep(1/3, 3), sigma = 1)
  )
)

# Extract samples, including the weight 'w' 
samples <- extract(fit_combined, pars = c("y_hat", "w"))

# Posterior predictive distributions
y_hat_ensemble <- samples$y_hat 

# Extract the estimated weights
weights <- samples$w

# Assuming weights is a matrix of 4000 rows (iterations) and 3 columns (one for each model)
mean_weights <- colMeans(weights)

# Calculate ensemble predictions using the mean weights
y_hat_ensemble_mean <- y_hat_AR1 * mean_weights[1] + y_hat_AR2 * mean_weights[2] + y_hat_AR3 * mean_weights[3]


# Function to calculate mean squared error (MSE)
calculate_mse <- function(y_hat, y_true, start_point = 4) {
  mse_values <- sapply(start_point:ncol(y_hat), function(t) {
    mse_per_person <- (y_hat[, t] - y_true[, t])^2
    mean(mse_per_person)  
  })
  mean(mse_values)  
}

mse_AR1 <- calculate_mse(y_hat_AR1, data_list$y)
mse_AR2 <- calculate_mse(y_hat_AR2, data_list$y)
mse_AR3 <- calculate_mse(y_hat_AR3, data_list$y)
mse_ensemble <- calculate_mse(y_hat_ensemble_mean, data_list$y)

# Display MSEs
print(mse_AR1)
print(mse_AR2)
print(mse_AR3)
print(mse_ensemble)

# Plot MSE over time for each model and ensemble 
mse_time_AR1 <- calculate_mse_time(y_hat_AR1, data_list$y)
mse_time_AR2 <- calculate_mse_time(y_hat_AR2, data_list$y)
mse_time_AR3 <- calculate_mse_time(y_hat_AR3, data_list$y)
mse_time_ensemble <- calculate_mse_time(y_hat_ensemble_mean, data_list$y)

x_axis <- 4:n_timepoints

# Prepare data for plotting
mse_time_data <- data.frame(
  Time = rep(x_axis, 4),
  MSE = c(mse_time_AR1, mse_time_AR2, mse_time_AR3, mse_time_ensemble),
  Model = factor(rep(c("AR(1)", "AR(2)", "AR(3)", "Ensemble"), each = length(x_axis)))
)

# Plot MSE over time for each model and ensemble with y-axis on log scale
ggplot(mse_time_data, aes(x = Time, y = MSE, color = Model, group = Model)) +
  geom_line(size = 1.2) +
  scale_y_log10() +
  labs(title = "MSE over Time (Starting from 4th Data Point)", x = "Time", y = "Log(MSE)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("AR(1)" = "gray", "AR(2)" = "gray", "AR(3)" = "gray", "Ensemble" = "purple"))

# Prepare data for density plot
error_data <- data.frame(
  Error = c(mse_time_AR1, mse_time_AR2, mse_time_AR3, mse_time_ensemble),
  Model = factor(rep(c("AR(1)", "AR(2)", "AR(3)", "Ensemble"), each = length(mse_time_AR1)))
)

# Plot error distribution as density plot
ggplot(error_data, aes(x = Error, color = Model, fill = Model)) +
  geom_density(alpha = 0.4) +
  scale_x_log10() +
  labs(title = "Error Distribution by Model", x = "Log(Error)", y = "Density") +
  theme_minimal() +
  theme(legend.position = "top")

