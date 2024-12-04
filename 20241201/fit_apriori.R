fit_apriori <- function(data, iter = 2000, chains = 4) {
  # Helper function to define and fit Stan models
  fit_model <- function(stan_code, data, iter, chains) {
    # Compile and fit the Stan model
    fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
    return(fit)
  }
  
  # AR(1) Model Stan Code
  stan_code_ar <- "
  data {
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[train_Nt];
    real test_y[test_Nt];
  }
  parameters {
    real<lower=-1, upper=1> ar_coef;     // AR coefficient
    real mu_ar;                         // Mean for the AR model
    real<lower=0> sigma_ar;             // Standard deviation for AR
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[train_Nt];        // Latent states
  }
  model {
    // Priors
    ar_coef ~ normal(0.7, 0.2);
    mu_ar ~ normal(0, 5);
    sigma_ar ~ cauchy(0, 1);
    sigma_m ~ cauchy(0, 3);

    // AR(1) latent state model
    train_eta[1] ~ normal(mu_ar, sigma_ar); // Initial latent state
    for (t in 2:train_Nt) {
      train_eta[t] ~ normal((1 - ar_coef) * mu_ar + ar_coef * train_eta[t - 1], sigma_ar);
    }
    

    // Measurement model
    for (t in 1:train_Nt) {
      train_y[t] ~ normal(train_eta[t], sigma_m);
    }
  }
  generated quantities {
    real y_pred_ar[train_Nt + test_Nt];
    real eta_pred_ar[train_Nt + test_Nt];
    real log_lik[train_Nt];  // Log-likelihood for PSIS-LOO
    real test_y_pred[test_Nt]; // Predicted values for test data
    real SSE;
    real RMSE;    

    SSE = 0;
    eta_pred_ar[1] = normal_rng(mu_ar, sigma_ar);
    y_pred_ar[1] = normal_rng(eta_pred_ar[1], sigma_m);

    for (t in 2:(train_Nt + test_Nt)) {
      eta_pred_ar[t] = normal_rng((1 - ar_coef) * mu_ar + ar_coef * eta_pred_ar[t-1], sigma_ar);
      y_pred_ar[t] = normal_rng(eta_pred_ar[t], sigma_m);
    }
    
    // Calculate log-likelihood
    for (t in 1:train_Nt) {
      log_lik[t] = normal_lpdf(train_y[t] | train_eta[t], sigma_m);
    }
    
    for (t in 1:test_Nt) {
      test_y_pred[t] = y_pred_ar[train_Nt + t];
      SSE += square(test_y[t] - test_y_pred[t]);
    }
    
    RMSE = sqrt(SSE / test_Nt);
  }
  "
  
  # MA(1) Model Stan Code
  stan_code_ma <- "
  data {
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[train_Nt];
    real test_y[test_Nt];
  }
  parameters {
    real ma_coef;    // MA coefficient
    real mu_ma;                         // Mean for the MA model
    real<lower=0> sigma_ma;             // Standard deviation for MA
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[train_Nt];        // Latent states
  }
  model {
    // Priors
    ma_coef ~ normal(0.5, 0.2);
    mu_ma ~ normal(0, 5);
    sigma_ma ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 3);

    // MA(1) latent state model
    train_eta[1] ~ normal(mu_ma, sigma_ma); // Initial latent state
    for (t in 2:train_Nt) {
      train_eta[t] ~ normal(mu_ma + ma_coef * (train_eta[t-1] - mu_ma), sigma_ma);
    }

    // Measurement model
    for (t in 1:train_Nt) {
      train_y[t] ~ normal(train_eta[t], sigma_m);
    }
  }
  generated quantities {
    real y_pred_ma[train_Nt + test_Nt];
    real eta_pred_ma[train_Nt + test_Nt];
    real log_lik[train_Nt];  // Log-likelihood for PSIS-LOO
    real test_y_pred[test_Nt]; // Predicted values for test data
    real SSE;
    real RMSE;   
    
    SSE = 0;
    eta_pred_ma[1] = normal_rng(mu_ma, sigma_ma);
    y_pred_ma[1] = normal_rng(eta_pred_ma[1], sigma_m);

    for (t in 2:(train_Nt + test_Nt)) {
      eta_pred_ma[t] = normal_rng(mu_ma + ma_coef * (eta_pred_ma[t - 1] - mu_ma), sigma_ma);
      y_pred_ma[t] = normal_rng(eta_pred_ma[t], sigma_m);
    }
    
    // Calculate log-likelihood
    for (t in 1:train_Nt) {
      log_lik[t] = normal_lpdf(train_y[t] | train_eta[t], sigma_m);
    }
  
    for (t in 1:test_Nt) {
      test_y_pred[t] = y_pred_ma[train_Nt + t];
      SSE += square(test_y[t] - test_y_pred[t]);
    }
    RMSE = sqrt(SSE / test_Nt);
  }
  "
  
  # Fit individual models
  fit_ar <- fit_model(stan_code_ar, data, iter, chains)
  fit_ma <- fit_model(stan_code_ma, data, iter, chains)
  
  # Extract log-likelihood and calculate PSIS-LOO
  log_lik_ar <- extract_log_lik(fit_ar, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_ar <- loo(log_lik_ar, moment_match = TRUE)
  log_lik_ma <- extract_log_lik(fit_ma, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_ma <- loo(log_lik_ma, moment_match = TRUE)
  
  # Extract predictions
  eta_pred_ar <- extract(fit_ar, pars = "eta_pred_ar")$eta_pred_ar
  eta_pred_ma <- extract(fit_ma, pars = "eta_pred_ma")$eta_pred_ma
  y_pred_ar <- extract(fit_ar, pars = "y_pred_ar")$y_pred_ar
  y_pred_ma <- extract(fit_ma, pars = "y_pred_ma")$y_pred_ma
  
  # Compute mean predictions
  e1 <- apply(eta_pred_ar, 2, mean)
  e2 <- apply(eta_pred_ma, 2, mean)
  f1 <- apply(y_pred_ar, 2, mean)
  f2 <- apply(y_pred_ma, 2, mean)
  
  # Combine predictions into a single array
  pred_e <- array(c(e1, e2), dim = c(data$train_Nt + data$test_Nt, 2))
  train_f <- array(c(f1[1:data$train_Nt], f2[1:data$train_Nt]), dim = c(data$train_Nt, 2))
  test_f <- array(c(f1[(data$train_Nt + 1):(data$train_Nt + data$test_Nt)], f2[(data$train_Nt + 1):(data$train_Nt + data$test_Nt)]), dim = c(data$test_Nt, 2))
  
  # Extract RMSE
  test_rmse_ar <- mean(extract(fit_ar, "RMSE")$RMSE)
  test_rmse_ma <- mean(extract(fit_ma, "RMSE")$RMSE)
  
  # Prepare output data
  data_fit <- list(
    train_Nt = data$train_Nt,
    test_Nt = data$test_Nt,
    train_y = data$train_y,
    test_y = data$test_y,
    J = 2,
    pred_e = pred_e,
    train_f = train_f,
    test_f = test_f
  )
  
  # Plot posterior distributions of AR and MA coefficients
  post_coef <- bayesplot::mcmc_areas(as.array(fit_ar), pars = c("ar_coef", "mu_ar", "sigma_ar")) +
    labs(title = "Posterior Distributions of AR Coefficient and Mu (mean) and Sigma (standard deviation)")
  
  # Plot posterior distributions of mu (mean)
  post_mu <- bayesplot::mcmc_areas(as.array(fit_ma), pars = c("ma_coef", "mu_ma", "sigma_ma")) +
    labs(title = "Posterior Distributions of MA Coefficient and Mu (mean) and Sigma (standard deviation)")
  
  post_plot = list(post_coef = post_coef, post_mu = post_mu)
  
  return(list(data_fit = data_fit, 
              res_ar = list(fit = fit_ar, log_lik = log_lik_ar, loo_result = loo_result_ar, test_rmse = test_rmse_ar), 
              res_ma = list(fit = fit_ma, log_lik = log_lik_ma, loo_result = loo_result_ma, test_rmse = test_rmse_ma),
              post_plot = post_plot))
}
