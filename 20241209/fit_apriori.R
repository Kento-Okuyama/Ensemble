fit_apriori <- function(data, iter = 2000, chains = 4, refresh = 0) {
  # ===========================
  #    Helper Function
  # ===========================
  fit_model <- function(stan_code, data, iter, chains, refresh) {
    # Compile and fit the Stan model
    fit <- tryCatch({
      stan(model_code = stan_code, data = data, iter = iter, chains = chains, refresh = refresh)
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
    int<lower=1> N; // Number of individuals
    int<lower=1> train_Nt;
    int<lower=1> val_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
    real val_y[N, val_Nt];
    real test_y[N, test_Nt];
  }
  parameters {
    real<lower=-1, upper=1> ar_coef;    
    real mu_ar;                         
    real lambda;                        
    real<lower=0> sigma_ar;             
    real<lower=0> sigma_m;              
    real train_eta[N, train_Nt];           
  }
  model {
    ar_coef ~ normal(0.7, 0.2);
    mu_ar ~ normal(0, 5);
    sigma_ar ~ cauchy(0, 1);
    sigma_m ~ cauchy(0, 3);
    
    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ar, sigma_ar);
      for (t in 2:train_Nt) {
        train_eta[n, t] ~ normal((1 - ar_coef) * mu_ar + ar_coef * train_eta[n, t - 1], sigma_ar);
      }
      
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(lambda * train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real log_lik[N, train_Nt];
    real y_pred_ar[N, train_Nt + val_Nt + test_Nt];
    real eta_pred_ar[N, train_Nt + val_Nt + test_Nt];
    real test_y_pred[N, test_Nt];
    real SSE;
    real RMSE;
  
    SSE = 0;
    for (n in 1:N) {
      // log likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | lambda * train_eta[n, t], sigma_m);
      }
  
      // eta predictions for train, validation, and test data
      eta_pred_ar[n, 1] = normal_rng(mu_ar, sigma_ar);
      for (t in 2:(train_Nt + val_Nt + test_Nt)) {
        eta_pred_ar[n, t] = normal_rng((1 - ar_coef) * mu_ar + ar_coef * eta_pred_ar[n, t - 1], sigma_ar);
      }
  
      // y predictions for train, validation, and test data
      for (t in 1:(train_Nt + val_Nt + test_Nt)) {
        y_pred_ar[n, t] = normal_rng(lambda * eta_pred_ar[n, t], sigma_m);
      }
  
      // test set predictions
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_ar[n, train_Nt + val_Nt + t];
        SSE += (test_y[n, t] - test_y_pred[n, t])^2;
      }
    }
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # MA(1) model Stan code
  stan_code_ma <- "
  data {
  int<lower=1> N; // Number of individuals
  int<lower=1> train_Nt;
  int<lower=1> val_Nt;
  int<lower=1> test_Nt;
  real train_y[N, train_Nt];
  real val_y[N, val_Nt];
  real test_y[N, test_Nt];
  }
  parameters {
    real ma_coef;                        
    real mu_ma;                         
    real lambda;                        
    real<lower=0> sigma_ma;             
    real<lower=0> sigma_m;              
    real train_eta[N, train_Nt];           
  }
  model {
    ma_coef ~ normal(0.5, 0.2);
    mu_ma ~ normal(0, 5);
    sigma_ma ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 3);
  
    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ma, sigma_ma);
      for (t in 2:train_Nt) {
        train_eta[n, t] ~ normal(mu_ma + ma_coef * (train_eta[n, t-1] - mu_ma), sigma_ma);
      }
  
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(lambda * train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real log_lik[N, train_Nt];
    real y_pred_ma[N, train_Nt + val_Nt + test_Nt];
    real eta_pred_ma[N, train_Nt + val_Nt + test_Nt];
    real test_y_pred[N, test_Nt];
    real SSE;
    real RMSE;
  
    SSE = 0;
    for (n in 1:N) {
      // log likelihood
      for (t in 1:train_Nt) {
        log_lik[n, t] = normal_lpdf(train_y[n, t] | lambda * train_eta[n, t], sigma_m);
      }
  
      // eta predictions for train, validation, and test data
      eta_pred_ma[n, 1] = normal_rng(mu_ma, sigma_ma);
      for (t in 2:(train_Nt + val_Nt + test_Nt)) {
        eta_pred_ma[n, t] = normal_rng(mu_ma + ma_coef * (eta_pred_ma[n, t - 1] - mu_ma), sigma_ma);
      }
  
      // y predictions for train, validation, and test data
      for (t in 1:(train_Nt + val_Nt + test_Nt)) {
        y_pred_ma[n, t] = normal_rng(lambda * eta_pred_ma[n, t], sigma_m);
      }
  
      // test set predictions
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = y_pred_ma[n, train_Nt + val_Nt + t];
        SSE += (test_y[n, t] - test_y_pred[n, t])^2;
      }
    }
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  # ===========================
  #    Fit Models
  # ===========================
  fit_ar <- fit_model(stan_code_ar, data, iter, chains, refresh)
  fit_ma <- fit_model(stan_code_ma, data, iter, chains, refresh)
  
  # ===========================
  #    Extract and Process Results
  # ===========================
  extract_results <- function(fit, suffix) {
    log_lik <- extract_log_lik(fit, "log_lik")
    loo_result <- loo(log_lik, moment_match = TRUE)
    test_rmse <- mean(extract(fit, pars = "RMSE")$RMSE)
    eta_pred <- extract(fit, pars = paste0("eta_pred_", suffix))[[1]]
    y_pred <- extract(fit, pars = paste0("y_pred_", suffix))[[1]]
    return(list(fit = fit, log_lik = log_lik, loo_result = loo_result, test_rmse = test_rmse,
                eta_pred = eta_pred, y_pred = y_pred))
  }
  
  res_ar <- extract_results(fit_ar, "ar")
  res_ma <- extract_results(fit_ma, "ma")
  
  # Combine predictions
  combined_predictions <- function(res_ar, res_ma, data) {
    e1 <- apply(res_ar$eta_pred, c(2, 3), mean)
    e2 <- apply(res_ma$eta_pred, c(2, 3), mean)
    f1 <- apply(res_ar$y_pred, c(2, 3), mean)
    f2 <- apply(res_ma$y_pred, c(2, 3), mean)
    
    pred_e <- array(c(e1, e2), dim = c(data$N, data$train_Nt + data$val_Nt + data$test_Nt, 2))
    train_f <- array(c(f1[1:data$train_Nt], f2[1:data$train_Nt]), dim = c(data$N, data$train_Nt, 2))
    val_f <- array(c(f1[(data$train_Nt + 1):(data$train_Nt + data$val_Nt)], 
                     f2[(data$train_Nt + 1):(data$train_Nt + data$val_Nt)]), dim = c(data$N, data$val_Nt, 2))
    test_f <- array(c(f1[(data$train_Nt + data$val_Nt + 1):(data$train_Nt + data$val_Nt + data$test_Nt)], 
                      f2[(data$train_Nt + data$val_Nt + 1):(data$train_Nt + data$val_Nt + data$test_Nt)]), dim = c(data$N, data$test_Nt, 2))
    return(list(pred_e = pred_e, train_f = train_f, val_f = val_f, test_f = test_f))
  }
  
  predictions <- combined_predictions(res_ar, res_ma, data)
  
  data_fit <- list(
    N = data$N,
    J = 2,
    train_Nt = data$train_Nt,
    val_Nt = data$val_Nt,
    test_Nt = data$test_Nt,
    train_y = data$train_y,
    val_y = data$val_y,
    test_y = data$test_y,
    pred_e = predictions$pred_e,
    train_f = predictions$train_f,
    val_f = predictions$val_f,
    test_f = predictions$test_f
  )
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    data_fit = data_fit,
    res_ar = res_ar,
    res_ma = res_ma
  ))
}
