fit_apriori_latent <- function(data, iter = 2000, chains = 4) {

  # Helper function to define and fit Stan models
  fit_model <- function(stan_code, data, iter, chains) {
    # Compile and fit the Stan model
    fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
    return(fit)
  }
  
  # AR(2) Model Stan Code
  stan_code_ar2 <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
    real test_y[N, test_Nt];
  }
  parameters {
    real<lower=-1, upper=1> ar_coef1;    // AR(1) coefficient
    real<lower=-1, upper=1> ar_coef2;    // AR(2) coefficient
    real mu_ar;                         // Mean for the AR model
    real<lower=0> sigma_ar;             // Standard deviation for AR
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[N, train_Nt];        // Latent states
  }
  model {
    // Priors
    ar_coef1 ~ normal(0, 1);
    ar_coef2 ~ normal(0, 1);
    mu_ar ~ normal(0, 5);
    sigma_ar ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    // AR(2) latent state model
    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ar, sigma_ar); // Initial latent state
      train_eta[n, 2] ~ normal(mu_ar, sigma_ar); // Second latent state
      for (t in 3:train_Nt) {
        train_eta[n, t] ~ normal(
          (1 - ar_coef1 - ar_coef2) * mu_ar +
          ar_coef1 * train_eta[n, t - 1] +
          ar_coef2 * train_eta[n, t - 2], sigma_ar);
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
    real y_pred_ar2[N, train_Nt + test_Nt];
    real eta_pred_ar2[N, train_Nt + test_Nt];
    real log_lik[N, train_Nt];  // Log-likelihood for PSIS-LOO
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE;    

    SSE = 0;
    for (n in 1:N) {
      eta_pred_ar2[n, 1] = normal_rng(mu_ar, sigma_ar);
      eta_pred_ar2[n, 2] = normal_rng(mu_ar, sigma_ar);
      y_pred_ar2[n, 1] = normal_rng(eta_pred_ar2[n, 1], sigma_m);
      y_pred_ar2[n, 2] = normal_rng(eta_pred_ar2[n, 2], sigma_m);

      for (t in 3:(train_Nt + test_Nt)) {
        eta_pred_ar2[n, t] = normal_rng(
          (1 - ar_coef1 - ar_coef2) * mu_ar +
          ar_coef1 * eta_pred_ar2[n, t - 1] +
          ar_coef2 * eta_pred_ar2[n, t - 2], sigma_ar);
        y_pred_ar2[n, t] = normal_rng(eta_pred_ar2[n, t], sigma_m);
      }

      // Calculate log-likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | train_eta[n, t], sigma_m);
      }
      
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_ar2[n, train_Nt + t];
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }  
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # MA(2) Model Stan Code
  stan_code_ma2 <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
    real test_y[N, test_Nt];
  }
  parameters {
    real ma_coef1;   // MA(1) coefficient
    real ma_coef2;   // MA(2) coefficient
    real mu_ma;                         // Mean for the MA model
    real<lower=0> sigma_ma;             // Standard deviation for MA
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[N, train_Nt];        // Latent states
  }
  model {
    // Priors
    ma_coef1 ~ normal(0, 1);
    ma_coef2 ~ normal(0, 1);
    mu_ma ~ normal(0, 5);
    sigma_ma ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    // MA(2) latent state model
    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ma, sigma_ma); // Initial latent state
      train_eta[n, 2] ~ normal(mu_ma, sigma_ma); // Second latent state
      for (t in 3:train_Nt) {
        train_eta[n, t] ~ normal(
          mu_ma +
          ma_coef1 * (train_eta[n, t - 1] - mu_ma) +
          ma_coef2 * (train_eta[n, t - 2] - mu_ma), sigma_ma);
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
    real y_pred_ma2[N, train_Nt + test_Nt];
    real eta_pred_ma2[N, train_Nt + test_Nt];
    real log_lik[N, train_Nt];  // Log-likelihood for PSIS-LOO
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE;    

    SSE = 0;
    for (n in 1:N) {
      eta_pred_ma2[n, 1] = normal_rng(mu_ma, sigma_ma);
      eta_pred_ma2[n, 2] = normal_rng(mu_ma, sigma_ma);
      y_pred_ma2[n, 1] = normal_rng(eta_pred_ma2[n, 1], sigma_m);
      y_pred_ma2[n, 2] = normal_rng(eta_pred_ma2[n, 2], sigma_m);

      for (t in 3:(train_Nt + test_Nt)) {
        eta_pred_ma2[n, t] = normal_rng(
          mu_ma +
          ma_coef1 * (eta_pred_ma2[n, t - 1] - mu_ma) +
          ma_coef2 * (eta_pred_ma2[n, t - 2] - mu_ma), sigma_ma);
        y_pred_ma2[n, t] = normal_rng(eta_pred_ma2[n, t], sigma_m);
      }

      // Calculate log-likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | train_eta[n, t], sigma_m);
      }
    
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_ma2[n, train_Nt + t];
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }  
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # Fit individual models
  fit_ar2 <- fit_model(stan_code_ar2, data, iter, chains)
  fit_ma2 <- fit_model(stan_code_ma2, data, iter, chains)
  
  # Extract log-likelihood and calculate PSIS-LOO
  log_lik_ar2 <- extract_log_lik(fit_ar2, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_ar2 <- loo(log_lik_ar2, moment_match = TRUE)
  log_lik_ma2 <- extract_log_lik(fit_ma2, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result_ma2 <- loo(log_lik_ma2, moment_match = TRUE)
  
  # Extract predictions
  eta_pred_ar2 <- extract(fit_ar2, pars = "eta_pred_ar2")$eta_pred_ar2
  eta_pred_ma2 <- extract(fit_ma2, pars = "eta_pred_ma2")$eta_pred_ma2
  y_pred_ar2 <- extract(fit_ar2, pars = "y_pred_ar2")$y_pred_ar2
  y_pred_ma2 <- extract(fit_ma2, pars = "y_pred_ma2")$y_pred_ma2
  
  # Compute mean predictions
  e1 <- apply(eta_pred_ar2, c(2, 3), mean)
  e2 <- apply(eta_pred_ma2, c(2, 3), mean)
  f1 <- apply(y_pred_ar2, c(2, 3), mean)
  f2 <- apply(y_pred_ma2, c(2, 3), mean)
  
  # Combine predictions into a single array
  train_e <- array(c(e1[, 1:data$train_Nt], e2[, 1:data$train_Nt]), dim = c(data$N, data$train_Nt, 2))
  test_e <- array(c(e1[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], e2[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)]), dim = c(data$N, data$test_Nt, 2))
  train_f <- array(c(f1[, 1:data$train_Nt], f2[, 1:data$train_Nt]), dim = c(data$N, data$train_Nt, 2))
  test_f <- array(c(f1[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], f2[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)]), dim = c(data$N, data$test_Nt, 2))
  
  # Extract RMSE
  test_rmse_ar2 <- mean(extract(fit_ar2, "RMSE")$RMSE)
  test_rmse_ma2 <- mean(extract(fit_ma2, "RMSE")$RMSE)

  # Prepare output data
  data_fit <- list(
    N = data$N,
    train_Nt = data$train_Nt,
    test_Nt = data$test_Nt,
    train_y = data$train_y,
    test_y = data$test_y,
    J = 2,
    train_e = train_e,
    test_e = test_e,
    train_f = train_f,
    test_f = test_f
  )
  
  # Plot posterior distributions of AR(2) and MA(2) coefficients
  post_coef_ar <- bayesplot::mcmc_areas(as.array(fit_ar2), pars = c("ar_coef1", "ar_coef2", "mu_ar", "sigma_ar")) +
    labs(title = "Posterior Distributions of AR(2) Coefficients")
  
  post_coef_ma <- bayesplot::mcmc_areas(as.array(fit_ma2), pars = c("ma_coef1", "ma_coef2", "mu_ma", "sigma_ma")) +
    labs(title = "Posterior Distributions of MA(2) Coefficients")
  
  post_plot <- list(post_coef_ar = post_coef_ar, post_coef_ma = post_coef_ma)
  
  return(list(data_fit = data_fit, res_ar2 = list(fit = fit_ar2, log_lik = log_lik_ar2, loo_result = loo_result_ar2, test_rmse = test_rmse_ar2), res_ma2 = list(fit = fit_ma2, log_lik = log_lik_ma2, loo_result = loo_result_ma2, test_rmse = test_rmse_ma2), post_plot = post_plot))
}
