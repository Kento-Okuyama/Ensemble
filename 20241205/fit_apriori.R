fit_apriori <- function(data, iter = 2000, chains = 4) {
  # ===========================
  #    Helper Function
  # ===========================
  fit_model <- function(stan_code, data, iter, chains) {
    # Compile and fit the Stan model
    fit <- tryCatch({
      stan(model_code = stan_code, data = data, iter = iter, chains = chains)
    }, error = function(e) {
      stop("Stan model fitting failed: ", e)
    })
    return(fit)
  }
  
  # ===========================
  #    Stan Model Code
  # ===========================
  # AR(1) model Stan code
  stan_code_ar <- "
  data {
    int<lower=1> train_Nt;
    int<lower=1> val_Nt;
    int<lower=1> test_Nt;
    real train_y[train_Nt];
    real val_y[val_Nt];
    real test_y[test_Nt];
  }
  parameters {
    real<lower=-1, upper=1> ar_coef;    // AR coefficient
    real mu_ar;                         // Mean for the AR model
    real lambda;                        // Factor loading for measurement model
    real<lower=0> sigma_ar;             // Standard deviation for AR
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[train_Nt];           // Latent states
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
      train_y[t] ~ normal(lambda * train_eta[t], sigma_m);
    }
  }
  generated quantities {
    real y_pred_ar[train_Nt + val_Nt + test_Nt];
    real eta_pred_ar[train_Nt + val_Nt + test_Nt];
    real log_lik[train_Nt];  // Log-likelihood for PSIS-LOO
    real test_y_pred[test_Nt]; // Predicted values for test data
    real SSE;
    real RMSE;

    SSE = 0;
    eta_pred_ar[1] = normal_rng(mu_ar, sigma_ar);
    y_pred_ar[1] = normal_rng(eta_pred_ar[1], sigma_m);
    for (t in 2:(train_Nt + val_Nt + test_Nt)) {
      eta_pred_ar[t] = normal_rng((1 - ar_coef) * mu_ar + ar_coef * eta_pred_ar[t-1], sigma_ar);
      y_pred_ar[t] = normal_rng(lambda * eta_pred_ar[t], sigma_m);
    }

    // Calculate log-likelihood
    for (t in 1:train_Nt) {
      log_lik[t] = normal_lpdf(train_y[t] | lambda * train_eta[t], sigma_m);
    }

    // Compute test RMSE
    for (t in 1:test_Nt) {
      test_y_pred[t] = y_pred_ar[train_Nt + val_Nt + t];
      SSE += square(test_y[t] - test_y_pred[t]);
    }
    RMSE = sqrt(SSE / test_Nt);
  }
  "
  
  # MA(1) model Stan code
  stan_code_ma <- "
  data {
    int<lower=1> train_Nt;
    int<lower=1> val_Nt;
    int<lower=1> test_Nt;
    real train_y[train_Nt];
    real val_y[val_Nt];
    real test_y[test_Nt];
  }
  parameters {
    real ma_coef;                       // MA coefficient
    real mu_ma;                         // Mean for the MA model
    real lambda;                        // Factor loading for measurement model
    real<lower=0> sigma_ma;             // Standard deviation for MA
    real<lower=0> sigma_m;              // Standard deviation for measurement error
    real train_eta[train_Nt];           // Latent states
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
      train_y[t] ~ normal(lambda * train_eta[t], sigma_m);
    }
  }
  generated quantities {
    real y_pred_ma[train_Nt + val_Nt + test_Nt];
    real eta_pred_ma[train_Nt + val_Nt + test_Nt];
    real log_lik[train_Nt];  // Log-likelihood for PSIS-LOO
    real test_y_pred[test_Nt]; // Predicted values for test data
    real SSE;
    real RMSE;

    SSE = 0;
    eta_pred_ma[1] = normal_rng(mu_ma, sigma_ma);
    y_pred_ma[1] = normal_rng(eta_pred_ma[1], sigma_m);
    for (t in 2:(train_Nt + val_Nt + test_Nt)) {
      eta_pred_ma[t] = normal_rng(mu_ma + ma_coef * (eta_pred_ma[t - 1] - mu_ma), sigma_ma);
      y_pred_ma[t] = normal_rng(lambda * eta_pred_ma[t], sigma_m);
    }

    // Calculate log-likelihood
    for (t in 1:train_Nt) {
      log_lik[t] = normal_lpdf(train_y[t] | lambda * train_eta[t], sigma_m);
    }

    // Compute test RMSE
    for (t in 1:test_Nt) {
      test_y_pred[t] = y_pred_ma[train_Nt + val_Nt + t];
      SSE += square(test_y[t] - test_y_pred[t]);
    }
    RMSE = sqrt(SSE / test_Nt);
  }
  "
  
  # ===========================
  #    Fit Models
  # ===========================
  fit_ar <- fit_model(stan_code_ar, data, iter, chains)
  fit_ma <- fit_model(stan_code_ma, data, iter, chains)
  
  # ===========================
  #    Extract and Process Results
  # ===========================
  extract_results <- function(fit, suffix) {
    log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
    loo_result <- loo(log_lik, moment_match = TRUE)
    test_rmse <- mean(extract(fit, "RMSE")$RMSE)
    eta_pred <- extract(fit, pars = paste0("eta_pred_", suffix))[[1]]
    y_pred <- extract(fit, pars = paste0("y_pred_", suffix))[[1]]
    return(list(fit = fit, log_lik = log_lik, loo_result = loo_result, test_rmse = test_rmse,
                eta_pred = eta_pred, y_pred = y_pred))
  }
  
  res_ar <- extract_results(fit_ar, "ar")
  res_ma <- extract_results(fit_ma, "ma")
  
  # Combine predictions and prepare data for further analysis
  combined_predictions <- function(res_ar, res_ma, data) {
    e1 <- apply(res_ar$eta_pred, 2, mean)
    e2 <- apply(res_ma$eta_pred, 2, mean)
    f1 <- apply(res_ar$y_pred, 2, mean)
    f2 <- apply(res_ma$y_pred, 2, mean)
    
    pred_e <- array(c(e1, e2), dim = c(data$train_Nt + data$val_Nt + data$test_Nt, 2))
    train_f <- array(c(f1[1:data$train_Nt], f2[1:data$train_Nt]), dim = c(data$train_Nt, 2))
    val_f <- array(c(f1[(data$train_Nt + 1):(data$train_Nt + data$val_Nt)], 
                     f2[(data$train_Nt + 1):(data$train_Nt + data$val_Nt)]), dim = c(data$val_Nt, 2))
    test_f <- array(c(f1[(data$train_Nt + data$val_Nt + 1):(data$train_Nt + data$val_Nt + data$test_Nt)], 
                      f2[(data$train_Nt + data$val_Nt + 1):(data$train_Nt + data$val_Nt + data$test_Nt)]), dim = c(data$test_Nt, 2))
    return(list(pred_e = pred_e, train_f = train_f, val_f = val_f, test_f = test_f))
  }
  
  predictions <- combined_predictions(res_ar, res_ma, data)
  
  data_fit <- list(
    train_Nt = data$train_Nt,
    val_Nt = data$val_Nt,
    test_Nt = data$test_Nt,
    train_y = data$train_y,
    val_y = data$val_y,
    test_y = data$test_y,
    J = 2,
    pred_e = predictions$pred_e,
    train_f = predictions$train_f,
    val_f = predictions$val_f,
    test_f = predictions$test_f
  )
  
  # ===========================
  #    Plot Posterior Distributions
  # ===========================
  plot_posterior <- function(fit, parameters, title) {
    bayesplot::mcmc_areas(as.array(fit), pars = parameters) +
      labs(title = title)
  }
  
  post_plot <- list(
    post_coef = plot_posterior(fit_ar, c("ar_coef", "mu_ar", "sigma_ar"), "Posterior Distributions (AR)"),
    post_mu = plot_posterior(fit_ma, c("ma_coef", "mu_ma", "sigma_ma"), "Posterior Distributions (MA)")
  )
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    data_fit = data_fit,
    res_ar = res_ar,
    res_ma = res_ma,
    post_plot = post_plot
  ))
}
