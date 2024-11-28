fit_apriori_latent <- function(data, iter = 2000, chains = 4) {
  # Helper function to define and fit Stan models
  fit_model <- function(stan_code, data, iter, chains) {
    # Compile and fit the Stan model
    fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
    return(fit)
  }
  
  # AR(1) Model Stan Code
  stan_code_ar <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
    real test_y[N, test_Nt];
  }
  parameters {
    real<lower=-1, upper=1> ar_coef;     // AR coefficient
    real mu_ar;                         // Mean for the AR model
    real<lower=0> sigma_ar;             // Standard deviation for AR
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[N, train_Nt];        // Latent states
  }
  model {
    // Priors
    ar_coef ~ normal(0, 1);
    mu_ar ~ normal(0, 5);
    sigma_ar ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    // AR(1) latent state model
    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ar, sigma_ar); // Initial latent state
      for (t in 2:train_Nt) {
        train_eta[n, t] ~ normal((1 - ar_coef) * mu_ar + ar_coef * train_eta[n, t - 1], sigma_ar);
      }
    }

    // Measurement model
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real y_pred_ar[N, train_Nt + test_Nt];
    real eta_pred_ar[N, train_Nt + test_Nt];
    real log_lik[N, train_Nt];  // Log-likelihood for PSIS-LOO
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE;    

    SSE = 0;
    for (n in 1:N) {
      eta_pred_ar[n, 1] = normal_rng(mu_ar, sigma_ar);
      y_pred_ar[n, 1] = normal_rng(eta_pred_ar[n, 1], sigma_m);

      for (t in 2:(train_Nt + test_Nt)) {
        eta_pred_ar[n, t] = normal_rng((1 - ar_coef) * mu_ar + ar_coef * eta_pred_ar[n, t-1], sigma_ar);
        y_pred_ar[n, t] = normal_rng(eta_pred_ar[n, t], sigma_m);
      }
      
      // Calculate log-likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | train_eta[n, t], sigma_m);
      }
      
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_ar[n, train_Nt + t];
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # MA(1) Model Stan Code
  stan_code_ma <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
    real test_y[N, test_Nt];
  }
  parameters {
    real ma_coef;    // MA coefficient
    real mu_ma;                         // Mean for the MA model
    real<lower=0> sigma_ma;             // Standard deviation for MA
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[N, train_Nt];        // Latent states
  }
  model {
    // Priors
    ma_coef ~ normal(0, 1);
    mu_ma ~ normal(0, 5);
    sigma_ma ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    // MA(1) latent state model
    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ma, sigma_ma); // Initial latent state
      for (t in 2:train_Nt) {
        train_eta[n, t] ~ normal(mu_ma + ma_coef * (train_eta[n, t-1] - mu_ma), sigma_ma);
      }
    }

    // Measurement model
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real y_pred_ma[N, train_Nt + test_Nt];
    real eta_pred_ma[N, train_Nt + test_Nt];
    real log_lik[N, train_Nt];  // Log-likelihood for PSIS-LOO
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE;   
    
    SSE = 0;
    for (n in 1:N) {
      eta_pred_ma[n, 1] = normal_rng(mu_ma, sigma_ma);
      y_pred_ma[n, 1] = normal_rng(eta_pred_ma[n, 1], sigma_m);

      for (t in 2:(train_Nt + test_Nt)) {
        eta_pred_ma[n, t] = normal_rng(mu_ma + ma_coef * (eta_pred_ma[n, t - 1] - mu_ma), sigma_ma);
        y_pred_ma[n, t] = normal_rng(eta_pred_ma[n, t], sigma_m);
      }
      
      // Calculate log-likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | train_eta[n, t], sigma_m);
      }
    
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_ma[n, train_Nt + t];
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }  
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # White Noise Model Stan Code
  stan_code_wn <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
    real test_y[N, test_Nt];
  }
  parameters {
    real mu_wn;                         // Mean for the WN model
    real<lower=0> sigma_wn;             // Standard deviation for WN
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[N, train_Nt];        // Latent states
  }
  model {
    // Priors
    mu_wn ~ normal(0, 5);
    sigma_wn ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    // White noise latent state model
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_eta[n, t] ~ normal(mu_wn, sigma_wn);
      }
    }

    // Measurement model
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real y_pred_wn[N, train_Nt + test_Nt];
    real eta_pred_wn[N, train_Nt + test_Nt];
    real log_lik[N, train_Nt];  // Log-likelihood for PSIS-LOO
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE;
    
    SSE = 0;
    for (n in 1:N) {
      eta_pred_wn[n, 1] = normal_rng(mu_wn, sigma_wn);
      y_pred_wn[n, 1] = normal_rng(eta_pred_wn[n, 1], sigma_m);

      for (t in 2:(train_Nt + test_Nt)) {
        eta_pred_wn[n, t] = normal_rng(mu_wn, sigma_wn);
        y_pred_wn[n, t] = normal_rng(eta_pred_wn[n, t], sigma_m);
      }
      
      // Calculate log-likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | train_eta[n, t], sigma_m);
      }
    
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_wn[n, train_Nt + t];
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }  
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # Fit individual models
  fit_ar <- fit_model(stan_code_ar, data, iter, chains)
  fit_ma <- fit_model(stan_code_ma, data, iter, chains)
  fit_wn <- fit_model(stan_code_wn, data, iter, chains)
  
  # Extract log-likelihood and calculate PSIS-LOO
  log_lik_ar <- extract_log_lik(fit_ar, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_ar <- loo(log_lik_ar, moment_match = TRUE)
  log_lik_ma <- extract_log_lik(fit_ma, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_ma <- loo(log_lik_ma, moment_match = TRUE)
  log_lik_wn <- extract_log_lik(fit_wn, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_wn <- loo(log_lik_wn, moment_match = TRUE)
  
  # Extract predictions
  eta_pred_ar <- extract(fit_ar, pars = "eta_pred_ar")$eta_pred_ar
  eta_pred_ma <- extract(fit_ma, pars = "eta_pred_ma")$eta_pred_ma
  eta_pred_wn <- extract(fit_wn, pars = "eta_pred_wn")$eta_pred_wn
  y_pred_ar <- extract(fit_ar, pars = "y_pred_ar")$y_pred_ar
  y_pred_ma <- extract(fit_ma, pars = "y_pred_ma")$y_pred_ma
  y_pred_wn <- extract(fit_wn, pars = "y_pred_wn")$y_pred_wn
  
  # Compute mean predictions
  e1 <- apply(eta_pred_ar, c(2, 3), mean)
  e2 <- apply(eta_pred_ma, c(2, 3), mean)
  e3 <- apply(eta_pred_wn, c(2, 3), mean)
  f1 <- apply(y_pred_ar, c(2, 3), mean)
  f2 <- apply(y_pred_ma, c(2, 3), mean)
  f3 <- apply(y_pred_wn, c(2, 3), mean)
  
  # Combine predictions into a single array
  train_e <- array(c(e1[, 1:data$train_Nt], e2[, 1:data$train_Nt], e3[, 1:data$train_Nt]), dim = c(data$N, data$train_Nt, 3))
  test_e <- array(c(e1[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], e2[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], e3[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)]), dim = c(data$N, data$test_Nt, 2))
  train_f <- array(c(f1[, 1:data$train_Nt], f2[, 1:data$train_Nt], f3[, 1:data$train_Nt]), dim = c(data$N, data$train_Nt, 3))
  test_f <- array(c(f1[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], f2[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], f3[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)]), dim = c(data$N, data$test_Nt, 3))
  
  # Extract RMSE
  test_rmse_ar <- mean(extract(fit_ar, "RMSE")$RMSE)
  test_rmse_ma <- mean(extract(fit_ma, "RMSE")$RMSE)
  test_rmse_wn <- mean(extract(fit_wn, "RMSE")$RMSE)
  
  # Prepare output data
  data_fit <- list(
    N = data$N,
    train_Nt = data$train_Nt,
    test_Nt = data$test_Nt,
    train_y = data$train_y,
    test_y = data$test_y,
    J = 3,
    train_e = train_e,
    test_e = test_e,
    train_f = train_f,
    test_f = test_f
  )
  
  # Plot posterior distributions of AR and MA coefficients
  post_coef <- bayesplot::mcmc_areas(as.array(fit_ar), pars = c("ar_coef", "mu_ar", "sigma_ar")) +
    labs(title = "Posterior Distributions of AR Coefficient and Mu (mean) and Sigma (standard deviation)")
  
  # Plot posterior distributions of mu (mean)
  post_mu <- bayesplot::mcmc_areas(as.array(fit_ma), pars = c("ma_coef", "mu_ma", "sigma_ma")) +
    labs(title = "Posterior Distributions of MA Coefficient and Mu (mean) and Sigma (standard deviation)")
  
  # Plot posterior distributions of sigma (standard deviation)
  post_sigma <- bayesplot::mcmc_areas(as.array(fit_wn), pars = c("mu_wn", "sigma_wn")) +
    labs(title = "Posterior Distributions of Mu (mean) and Sigma (standard deviation)")
  
  post_plot = list(post_coef = post_coef, post_mu = post_mu, post_sigma = post_sigma)
  
  return(list(data_fit = data_fit, 
              res_ar = list(fit = fit_ar, log_lik = log_lik_ar, loo_result = loo_result_ar, test_rmse = test_rmse_ar), 
              res_ma = list(fit = fit_ma, log_lik = log_lik_ma, loo_result = loo_result_ma, test_rmse = test_rmse_ma),
              res_wn = list(fit = fit_wn, log_lik = log_lik_wn, loo_result = loo_result_wn, test_rmse = test_rmse_wn),
              post_plot = post_plot))
}
