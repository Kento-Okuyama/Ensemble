fit_BMA <- function(data, iter = 2000, chains = 4) {
  
  # Stan model code (referenced below as stan_code)
  stan_code <- "
  data {
    int<lower=1> N;                  // Number of time series
    int<lower=1> train_Nt;                 // Length of each time series
    real train_y[N, train_Nt];                    // Observation matrix (each time series is a row)
  }
  
  parameters {
    vector[4] mu;                    // Mean for each regime
    real<lower=0, upper=1> lambda;   // Smoothing parameter (closer to 0 gives more weight to previous observation)
    real<lower=0, upper=1> ar_pos;  // AR coefficient for regime 1
    real<lower=-1, upper=0> ar_neg;  // AR coefficient for regime 2
    real<lower=-1, upper=1> ma_coef; // MA coefficient for regime 3
    vector<lower=0>[4] sigma;        // Standard deviation for each regime
    simplex[4] w;                    // Stacking weights for model averaging
  }
  
  model {
    real weighted_prediction;  
    real variance_weighted;
    // Priors for parameters
    mu ~ normal(0, 5);
    lambda ~ beta(2, 2);
    ar_pos ~ normal(0.7, 0.3);
    ar_neg ~ normal(-0.7, 0.3);
    ma_coef ~ normal(0, 0.5);
    sigma ~ cauchy(0, 2);
    w ~ dirichlet(rep_vector(1.0, 4)); // Prior for model weights
  
    // Time series model with model-averaged AR/MA processes
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        real target_mean = lambda * dot_product(w, mu) + (1 - lambda) * train_y[n, t-1];
        vector[4] y_pred; // Define y_pred as a vector
  
        // Calculate regime-specific predictions
        y_pred[1] = (1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1];
        y_pred[2] = (1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1];
        y_pred[3] = target_mean + ma_coef * (train_y[n, t - 1] - target_mean);
        y_pred[4] = target_mean;
  
        // Compute weighted prediction across regimes
        weighted_prediction = dot_product(w, y_pred); // Use y_pred as vector
        variance_weighted = dot_product(w, sigma); // Use sigma as vector
        
        // Observation model
        train_y[n, t] ~ normal(weighted_prediction, variance_weighted);
      }
    }
  }
  
  generated quantities {
    real weighted_prediction;  
    real variance_weighted;
    real y_rep[N, train_Nt-1];             // Posterior predictive samples
    real log_lik[N, train_Nt-1];           // Log-likelihood values for LOO calculation
  
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        real target_mean = lambda * dot_product(w, mu) + (1 - lambda) * train_y[n, t-1];
        vector[4] y_pred;
  
        // Calculate regime-specific predictions
        y_pred[1] = (1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1];
        y_pred[2] = (1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1];
        y_pred[3] = target_mean + ma_coef * (train_y[n, t-1] - target_mean);
        y_pred[4] = target_mean;
  
        // Weighted prediction across regimes
        weighted_prediction = dot_product(w, y_pred);
        variance_weighted = dot_product(w, sigma);
  
        // Posterior predictive distribution and log likelihood
        y_rep[n, t-1] = normal_rng(weighted_prediction, variance_weighted);
        log_lik[n, t-1] = normal_lpdf(train_y[n, t] | weighted_prediction, variance_weighted);
      }
    }
  }
  "
  
  # Compile and sample from the Stan model
  fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
  
  # Extract log-likelihood for PSIS-LOO cross-validation
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  print(loo_result)

  # Plot posterior distributions of AR and MA coefficients
  post_coef <- bayesplot::mcmc_areas(as.array(fit), pars = c("ar_pos", "ar_neg", "ma_coef")) +
    labs(title = "Posterior Distributions of AR and MA Coefficients")
  
  # Plot posterior distributions of sigma (standard deviations for each regime)
  post_sigma <- bayesplot::mcmc_areas(as.array(fit), pars = c("sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]")) +
    labs(title = "Posterior Distributions of Sigma (Standard Deviation) per Regime")
  
  # Plot posterior distributions of mu (mean of each regime) and lambda (smoothing parameter)
  post_mu_lmd <- bayesplot::mcmc_areas(as.array(fit), pars = c("mu[1]", "mu[2]", "mu[3]", "mu[4]", "lambda")) +
    labs(title = "Posterior Distributions of Regime Means and Lambda")
  
  # Plot posterior distributions of stacking weights
  post_w <- bayesplot::mcmc_areas(as.array(fit), pars = c("w[1]", "w[2]", "w[3]", "w[4]")) +
    labs(title = "Posterior Distributions of Model Averaging Weights")
  
  post_plot = list(post_coef = post_coef, post_sigma = post_sigma, post_mu_lmd = post_mu_lmd, post_w = post_w)

  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, post_plot = post_plot))
}