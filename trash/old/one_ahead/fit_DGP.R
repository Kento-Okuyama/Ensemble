fit_DGP <- function(data, iter = 2000, chains = 4) {
  
  # Stan model code definition
  stan_code <- "
  data {
    int<lower=1> N;                  // Number of time series
    int<lower=1> train_Nt;                 // Length of each time series
    int<lower=1, upper=4> train_regime[N, train_Nt];  // Regime index at each time point
    real train_y[N, train_Nt];                    // Observation matrix (each time series is a row)
  }
  
  parameters {
    vector[4] mu;                    // Mean for each regime
    real<lower=0, upper=1> lambda;   // Smoothing parameter (closer to 0 gives more weight to previous observation)
    real<lower=0, upper=1> ar_pos;  // AR coefficient for regime 1
    real<lower=-1, upper=0> ar_neg;  // AR coefficient for regime 2
    real<lower=-1, upper=1> ma_coef; // MA coefficient for regime 3
    vector<lower=0>[4] sigma;        // Standard deviation for each regime
  }
  
  model {
    // Priors for parameters
    mu ~ normal(0, 5);
    lambda ~ beta(2, 2);
    ar_pos ~ normal(0.7, 0.3);
    ar_neg ~ normal(-0.7, 0.3);
    ma_coef ~ normal(0, 0.5);
    sigma ~ cauchy(0, 2);
  
    // Time series model with regime-dependent AR/MA processes
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        int k = train_regime[n, t];
        real target_mean = lambda * mu[k] + (1 - lambda) * train_y[n, t-1];
  
        if (k == 1) {
          train_y[n, t] ~ normal((1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1], sigma[k]);
        } else if (k == 2) {
          train_y[n, t] ~ normal((1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1], sigma[k]);
        } else if (k == 3) {
          train_y[n, t] ~ normal(target_mean + ma_coef * (train_y[n, t-1] - target_mean), sigma[k]);
        } else {
          train_y[n, t] ~ normal(target_mean, sigma[k]);
        }
      }
    }
  }
  generated quantities {
    real y_rep[N, train_Nt-1];
    real log_lik[N, train_Nt-1];
  
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        int k = train_regime[n, t];
        real target_mean = lambda * mu[k] + (1 - lambda) * train_y[n, t-1];
  
        if (k == 1) {
          y_rep[n, t-1] = normal_rng((1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1], sigma[k]);
          log_lik[n, t-1] = normal_lpdf(train_y[n, t] | (1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1], sigma[k]);
        } else if (k == 2) {
          y_rep[n, t-1] = normal_rng((1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1], sigma[k]);
          log_lik[n, t-1] = normal_lpdf(train_y[n, t] | (1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1], sigma[k]);
        } else if (k == 3) {
          y_rep[n, t-1] = normal_rng(target_mean + ma_coef * (train_y[n, t-1] - target_mean), sigma[k]);
          log_lik[n, t-1] = normal_lpdf(train_y[n, t] | target_mean + ma_coef * (train_y[n, t-1] - target_mean), sigma[k]);
        } else {
          y_rep[n, t-1] = normal_rng(target_mean, sigma[k]);
          log_lik[n, t-1] = normal_lpdf(train_y[n, t] | target_mean, sigma[k]);
        }
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
  post_coef <-  bayesplot::mcmc_areas(as.array(fit), pars = c("ar_pos", "ar_neg", "ma_coef")) +
    labs(title = "Posterior Distributions of AR and MA Coefficients")
  
  # Plot posterior distributions of sigma (standard deviations for each regime)
  post_sigma <- bayesplot::mcmc_areas(as.array(fit), pars = c("sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]")) +
    labs(title = "Posterior Distributions of Sigma (Standard Deviation) per Regime")
  
  # Plot posterior distributions of mu (mean of each regime) and lambda (smoothing parameter)
  post_mu_lmd <- bayesplot::mcmc_areas(as.array(fit), pars = c("mu[1]", "mu[2]", "mu[3]", "mu[4]", "lambda")) +
    labs(title = "Posterior Distributions of Regime Means and Lambda")
  
  post_plot = list(post_coef = post_coef, post_sigma = post_sigma, post_mu_lmd = post_mu_lmd)

  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, post_plot = post_plot))
}