fit_apriori_obs <- function(data, iter = 2000, chains = 4) {
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
  }
  parameters {
    real<lower=0, upper=1> ar_coef;
    real mu_ar;
    real<lower=0> sigma_ar;
    real<lower=0> sigma_m;
    real train_eta[N, train_Nt];
  }
  model {
    ar_coef ~ beta(2, 2);
    mu_ar ~ normal(0, 5);
    sigma_ar ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    for (n in 1:N) {
      train_eta[n, 1] ~ normal(mu_ar, sigma_ar);
      for (t in 2:train_Nt) {
        train_eta[n, t] ~ normal((1 - ar_coef) * mu_ar + ar_coef * train_eta[n, t - 1], sigma_ar);
      }
    }

    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real y_pred_ar[N, train_Nt + test_Nt];
    real eta_pred[N, train_Nt + test_Nt];

    for (n in 1:N) {
      eta_pred[n, 1] = normal_rng(mu_ar, sigma_ar);
      y_pred_ar[n, 1] = normal_rng(eta_pred[n, 1], sigma_m);

      for (t in 2:(train_Nt + test_Nt)) {
        eta_pred[n, t] = normal_rng((1 - ar_coef) * mu_ar + ar_coef * eta_pred[n, t - 1], sigma_ar);
        y_pred_ar[n, t] = normal_rng(eta_pred[n, t], sigma_m);
      }
    }
  }
  "
  
  # MA(1) Model Stan Code
  stan_code_ma <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
  }
  parameters {
    real<lower=-1, upper=1> ma_coef;
    real mu_ma;
    real<lower=0> sigma_ma;
    real<lower=0> sigma_m;
  }
  model {
    ma_coef ~ normal(0, 0.5);
    mu_ma ~ normal(0, 5);
    sigma_ma ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    for (n in 1:N) {
      for (t in 2:train_Nt) {
        real prediction_ma = mu_ma + ma_coef * (train_y[n, t - 1] - mu_ma);
        train_y[n, t] ~ normal(prediction_ma, sigma_ma);
      }
    }
  }
  generated quantities {
    real y_pred_ma[N, train_Nt + test_Nt];

    for (n in 1:N) {
      y_pred_ma[n, 1] = mu_ma;

      for (t in 2:(train_Nt + test_Nt)) {
        y_pred_ma[n, t] = normal_rng(mu_ma + ma_coef * (y_pred_ma[n, t - 1] - mu_ma), sigma_ma);
      }
    }
  }
  "
  
  # White Noise Model Stan Code
  stan_code_wn <- "
  data {
    int<lower=1> N;
    int<lower=1> train_Nt;
    int<lower=1> test_Nt;
    real train_y[N, train_Nt];
  }
  parameters {
    real mu_wn;
    real<lower=0> sigma_wn;
    real<lower=0> sigma_m;
    real train_eta[N, train_Nt];
  }
  model {
    mu_wn ~ normal(0, 5);
    sigma_wn ~ cauchy(0, 2);
    sigma_m ~ cauchy(0, 2);

    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_eta[n, t] ~ normal(mu_wn, sigma_wn);
      }
    }

    for (n in 1:N) {
      for (t in 1:train_Nt) {
        train_y[n, t] ~ normal(train_eta[n, t], sigma_m);
      }
    }
  }
  generated quantities {
    real y_pred_wn[N, train_Nt + test_Nt];

    for (n in 1:N) {
      for (t in 1:(train_Nt + test_Nt)) {
        y_pred_wn[n, t] = normal_rng(mu_wn, sigma_wn);
      }
    }
  }
  "
  
  # Fit individual models
  fit_ar <- fit_model(stan_code_ar, data, iter, chains)
  fit_ma <- fit_model(stan_code_ma, data, iter, chains)
  fit_wn <- fit_model(stan_code_wn, data, iter, chains)
  
  # Extract predictions
  y_pred_ar <- extract(fit_ar, pars = "y_pred_ar")$y_pred_ar
  y_pred_ma <- extract(fit_ma, pars = "y_pred_ma")$y_pred_ma
  y_pred_wn <- extract(fit_wn, pars = "y_pred_wn")$y_pred_wn
  
  # Compute mean predictions
  f1 <- apply(y_pred_ar, c(2, 3), mean)
  f2 <- apply(y_pred_ma, c(2, 3), mean)
  f3 <- apply(y_pred_wn, c(2, 3), mean)
  
  # Combine predictions into a single array
  train_f <- array(c(f1[, 1:data$train_Nt], f2[, 1:data$train_Nt], f3[, 1:data$train_Nt]), dim = c(data$N, data$train_Nt, 3))
  test_f <- array(c(f1[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], f2[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)], f3[, (data$train_Nt + 1):(data$train_Nt + data$test_Nt)]), dim = c(data$N, data$test_Nt, 3))
  
  # Prepare output data
  data_fit <- list(
    N = data$N,
    train_Nt = data$train_Nt,
    test_Nt = data$test_Nt,
    train_y = data$train_y,
    test_y = data$test_y,
    J = 3,
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
  
  return(list(data_fit = data_fit, post_plot = post_plot))
}
