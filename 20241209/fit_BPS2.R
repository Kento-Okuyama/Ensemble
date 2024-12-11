fit_BPS2 <- function(data, iter = 2000, chains = 4, refresh = 0) {
  stan_code <- "
  data {
  int<lower=1> N;                    // Number of individuals
  int<lower=1> train_Nt;             // Training data length
  int<lower=1> val_Nt;               // Validation data length
  int<lower=1> test_Nt;              // Test data length
  int<lower=1> J;                    // Number of models
  vector[J] pred_e[N, train_Nt + val_Nt + test_Nt]; // Latent predictions
  real val_y[N, val_Nt];             // Validation observed data
  real val_f[N, val_Nt, J];          // Validation model predictions
  real test_y[N, test_Nt];           // Test observed data
  real test_f[N, test_Nt, J];        // Test model predictions
  }
  
  parameters {
    simplex[J] mu[N];                  // Initial model weights for each individual
    real alpha[N, val_Nt + test_Nt];   // Time-varying intercept for each individual
    vector[J] beta[N, val_Nt + test_Nt]; // Time-varying weights for each time step and individual
    vector[J] delta[N];                // Markov-switching coefficients for each individual
    real<lower=0> sigma[N];            // Observation noise standard deviation for each individual
    real<lower=0, upper=1> tau_a;      // Random walk noise for alpha
    real<lower=0, upper=1> tau_b;      // Random walk noise for beta
    real<lower=0, upper=1> tau_d;      // Random walk noise for delta
  }
  
  model {
    // Priors
    for (n in 1:N) {
      mu[n] ~ dirichlet(rep_vector(1.0, J));  
      sigma[n] ~ normal(0, 1);
      delta[n] ~ normal(0, tau_d);
    }
    tau_a ~ normal(0, 0.5);
    tau_b ~ normal(0, 0.3);
    tau_d ~ normal(0, 0.1);
  
    for (n in 1:N) {
      alpha[n, 1] ~ normal(0, tau_a);  
      beta[n, 1] ~ normal(mu[n], tau_b);  
  
      for (t in 2:(val_Nt + test_Nt)) {
        alpha[n, t] ~ normal(alpha[n, t - 1], tau_a);
        beta[n, t] ~ normal(beta[n, t - 1] + pred_e[n, train_Nt + t - 1] .* delta[n], tau_b);
      }
  
      for (t in 1:val_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = val_f[n, t, j];
        }
        val_y[n, t] ~ normal(alpha[n, t] + dot_product(beta[n, t], model_predictions), sigma[n]);
      }
    }
  }
  
  generated quantities {
    real log_lik[N, val_Nt];
    real test_y_pred[N, test_Nt];
    real SSE;
    real RMSE;
  
    SSE = 0;
  
    for (n in 1:N) {
      for (t in 1:val_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = val_f[n, t, j];
        }
        log_lik[n, t] = normal_lpdf(val_y[n, t] | alpha[n, t] + dot_product(beta[n, t], model_predictions), sigma[n]);
      }
  
      for (t in 1:test_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = test_f[n, t, j];
        }
        test_y_pred[n, t] = normal_rng(alpha[n, val_Nt + t] + dot_product(beta[n, val_Nt + t], model_predictions), sigma[n]);
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }
  
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  
  fit <- tryCatch({
    stan(model_code = stan_code, data = data, iter = iter, chains = chains, refresh = refresh)
  }, error = function(e) {
    stop("Stan model fitting failed: ", e)
  })
  
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  
  # print(loo_result)
  
  test_rmse <- mean(extract(fit, pars = "RMSE")$RMSE)
  # cat("\n", "Average RMSE across all samples:", test_rmse, "\n")

  return(list(
    fit = fit,
    log_lik = log_lik,
    loo_result = loo_result,
    test_rmse = test_rmse
  ))
}
