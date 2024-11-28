# 1. Simulating Data
# Load necessary libraries
library(brms)
library(rstan)
library(loo)  # For leave-one-out cross-validation
library(rstanarm)

# Simulate data
set.seed(42)
n <- 100 # number of data points
x <- runif(n, -3, 3)  # predictor

# Define two models with different strengths
y1 <- 1.5 * sin(x) + rnorm(n, sd = 0.3)  # Model 1 is good in the middle region
y2 <- 1.5 * cos(x) + rnorm(n, sd = 0.3)  # Model 2 is better in the outer regions

# Combine the two
y <- ifelse(abs(x) < 1, y1, y2) + rnorm(n, sd = 0.3)

# Create a data frame
data <- data.frame(x = x, y = y)

# 2. Fit individual models

# Fit two linear models
fit1 <- stan_glm(y ~ sin(x), data = data)
fit2 <- stan_glm(y ~ cos(x), data = data)

# 3. Compute leave-one-out cross-validation (LOO)
loo_fit1 <- loo(fit1)
loo_fit2 <- loo(fit2)

# 4. Implement Bayesian Hierarchical Stacking
# Define the log predictive densities (PSIS-LOO)
lpd1 <- loo_fit1$pointwise[, "elpd_loo"]
lpd2 <- loo_fit2$pointwise[, "elpd_loo"]

# Combine the log predictive densities with equal weights (stacking)
weights <- c(0.5, 0.5)  # equal weights for simplicity
stacked_lpd <- log(exp(lpd1) * weights[1] + exp(lpd2) * weights[2])

# Compute the sum of the log predictive densities for stacked model
sum(stacked_lpd)

# 5. Compare with hierarchical stacking
# Stan code for hierarchical stacking
stan_code <- "
data {
  int<lower=1> N;  // number of observations
  vector[N] x;  // predictor
  matrix[N, 2] lpd;  // log predictive densities of models
}
parameters {
  real alpha;  // intercept for logistic regression
  real beta;   // slope for logistic regression
}
model {
  vector[N] w;
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  
  for (n in 1:N) {
    w[n] = inv_logit(alpha + beta * x[n]);
    target += log_mix(w[n], lpd[n, 1], lpd[n, 2]);  // Log of weighted combination
  }
}
"

# Prepare data for Stan
stan_data <- list(
  N = n,
  x = data$x,
  lpd = cbind(lpd1, lpd2)
)

# Fit the model
fit_bhs <- stan(model_code = stan_code, data = stan_data)

# Extract and summarize the posterior
print(fit_bhs, pars = c("alpha", "beta"))

# 6. Results and comparison
# Extract posterior mean for alpha and beta
post <- rstan::extract(fit_bhs)
alpha_mean <- mean(post$alpha)
beta_mean <- mean(post$beta)

# Compute the weighted combination of models based on the estimated weights
weights_bhs <- plogis(alpha_mean + beta_mean * data$x)
stacked_lpd_bhs <- log(exp(lpd1) * weights_bhs + exp(lpd2) * (1 - weights_bhs))

# Sum of the log predictive densities for hierarchical stacking
sum(stacked_lpd_bhs)

# Compare with the complete-pooling result
sum(stacked_lpd)  # from earlier complete-pooling stacking

