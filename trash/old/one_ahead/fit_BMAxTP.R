fit_BMAxTP <- function(data, iter = 2000, chains = 4) {
  
  # Stan model code
  stan_code <- "
  data {
    int<lower=1> N;                   // Number of individuals
    int<lower=1> train_Nt;                  // Time series length for each individual
    real train_y[N, train_Nt];                    // Observed data
  }
  parameters {
    vector[4] mu;                     // Mean for each state
    real<lower=0, upper=1> lambda;    // Smoothing parameter for target mean
    real<lower=0, upper=1> ar_pos;    // AR coefficient for state 1
    real<lower=-1, upper=0> ar_neg;   // AR coefficient for state 2
    real<lower=-1, upper=1> ma_coef;  // MA coefficient for state 3
    vector<lower=0>[4] sigma;         // Standard deviation for each state
    simplex[4] T[4];                  // Transition probability matrix
  }
  
  transformed parameters {
    vector[4] w[train_Nt];               // Dynamic weights
    
    // Initialize weights
    w[1] = [1, 0, 0, 0]';
  
    // Update weights based on transition probabilities
    for (t in 2:train_Nt) {
      for (j in 1:4) { 
          w[t][j] = T[1][j] * w[t - 1][1] + T[2][j] * w[t - 1][2] + T[3][j] * w[t - 1][3] + T[4][j] * w[t - 1][4]; 
      }
    }
  }
  
  model {
    real weighted_prediction;
    real variance_weighted;
  
    // Prior distributions for parameters
    mu ~ normal(0, 5);
    lambda ~ beta(2, 2);
    ar_pos ~ normal(0.7, 0.3);
    ar_neg ~ normal(-0.7, 0.3);
    ma_coef ~ normal(0, 0.5);
    sigma ~ cauchy(0, 2);
  
    // Priors for transition matrix
    for (i in 1:4) {
      vector[4] alpha = rep_vector(0.1, 4);
      alpha[i] = 0.8;
      T[i] ~ dirichlet(alpha);
    }
  
    // State-dependent model for each time series
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        real target_mean = lambda * dot_product(w[t], mu) + (1 - lambda) * train_y[n, t - 1];
        vector[4] y_pred;
  
        // State-specific predictions
        y_pred[1] = (1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1];
        y_pred[2] = (1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1];
        y_pred[3] = target_mean + ma_coef * (train_y[n, t - 1] - target_mean);
        y_pred[4] = target_mean;
  
        // Weighted prediction based on dynamic weights
        weighted_prediction = dot_product(w[t], y_pred);
        variance_weighted = dot_product(w[t], sigma); 
  
        // Observation model
        train_y[n, t] ~ normal(weighted_prediction, variance_weighted);
      }
    }
  }
  generated quantities {
    real weighted_prediction;
    real variance_weighted;
    matrix[N, train_Nt-1] y_rep;             // Posterior predictive samples
    matrix[N, train_Nt-1] log_lik;           // Log-likelihood for LOO
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        real target_mean = lambda * dot_product(w[t], mu) + (1 - lambda) * train_y[n, t - 1];
        vector[4] y_pred;
        y_pred[1] = (1 - ar_pos) * target_mean + ar_pos * train_y[n, t-1];
        y_pred[2] = (1 - ar_neg) * target_mean + ar_neg * train_y[n, t-1];
        y_pred[3] = target_mean + ma_coef * (train_y[n, t-1] - target_mean);
        y_pred[4] = target_mean;
  
        weighted_prediction = dot_product(w[t], y_pred);
        variance_weighted = dot_product(w[t], sigma);
        
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
  
  # # Plot posterior distributions of stacking weights
  # post_w1 <- bayesplot::mcmc_areas(as.array(fit), pars = c("w1[1]", "w1[2]", "w1[3]", "w1[4]")) +
  #   labs(title = "Posterior Distributions of initial Weights")
  
  # Posterior analysis of transition probabilities
  post_T_stay <- bayesplot::mcmc_areas(as.array(fit), pars = c("T[1,1]", "T[2,2]", "T[3,3]", "T[4,4]")) +
    labs(title = "Posterior distributions of transition matrix diagonal elements")
  
  post_plot = list(post_coef = post_coef, post_sigma = post_sigma, post_mu_lmd = post_mu_lmd, post_T_stay = post_T_stay)

  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, post_plot = post_plot))
}