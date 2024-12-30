fit_BPS2 <- function(data, iter = 2000, chains = 4, refresh = 0) {
  # ===========================
  #    Stan Model Code
  # ===========================
  stan_code <- "
  data {
    int<lower=1> N;                  // Number of individuals
    int<lower=1> val_Nt;             // Validation data length
    int<lower=1> test_Nt;            // Test data length
    int<lower=1> J;                  // Number of models
    int<lower=1> S;                  // Number of samples per model
    vector[J] pred_e[S, N, train_Nt + val_Nt + test_Nt]; // Latent predictions (samples)
    real val_y[N, val_Nt];           // Validation observed data
    real val_f[S, N, val_Nt, J];     // Validation model predictions (samples)
    real test_y[N, test_Nt];         // Test observed data
    real test_f[S, N, test_Nt, J];   // Test model predictions (samples)
  }
  
  parameters {
    simplex[J] mu[N];                  // Initial model weights for each individual
    vector[val_Nt + test_Nt] alpha[N]; // Time-varying intercept for each individual
    vector[J] beta[N, val_Nt + test_Nt]; // Time-varying weights for each time step and individual
    vector[J] delta[N];                // Eta-dependent intercept change
    vector[J] theta[N];                // Eta-dependent model weight change
    real<lower=0> sigma[N];            // Observation noise standard deviation for each individual
    real<lower=0, upper=1> tau_a;      // Random walk noise for alpha
    real<lower=0, upper=1> tau_b;      // Random walk noise for beta
    real<lower=0, upper=1> tau_d;      // Random walk noise for delta
    real<lower=0, upper=1> tau_t;      // Random walk noise for theta
  }
  
  model {
    // Priors
    for (n in 1:N) {
      mu[n] ~ dirichlet(rep_vector(1.0, J));  
      sigma[n] ~ normal(0, 1e-1);
      delta[n] ~ normal(0, tau_d);
      theta[n] ~ normal(0, tau_t);
    }
    tau_a ~ normal(0, 1e-1);
    tau_b ~ normal(0, 1e-1);
    tau_d ~ normal(0, 1e-1);
    tau_t ~ normal(0, 1e-1);
  
    for (n in 1:N) {
      alpha[n, 1] ~ normal(0, tau_a);  
      beta[n, 1] ~ normal(mu[n], tau_b);  
  
      for (t in 2:(val_Nt + test_Nt)) {
        vector[J] mean_pred_e;
        for (j in 1:J) {
          mean_pred_e[j] = mean(pred_e[, n, train_Nt + t - 1, j]);
        }
        alpha[n, t] ~ normal(alpha[n, t - 1] + dot_product(mean_pred_e, delta[n]), tau_a);
        beta[n, t] ~ normal(beta[n, t - 1] + dot_product(mean_pred_e, theta[n]), tau_b);
      }
  
      for (t in 1:val_Nt) {
        vector[J] mean_model_predictions;
        for (j in 1:J) {
          mean_model_predictions[j] = mean(val_f[, n, t, j]);
        }
        val_y[n, t] ~ normal(alpha[n, t] + dot_product(beta[n, t], mean_model_predictions), sigma[n]);
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
        vector[J] mean_model_predictions;
        for (j in 1:J) {
          mean_model_predictions[j] = mean(val_f[, n, t, j]);
        }
        log_lik[n, t] = normal_lpdf(val_y[n, t] | alpha[n, t] + dot_product(beta[n, t], mean_model_predictions), sigma[n]);
      }
  
      for (t in 1:test_Nt) {
        vector[J] mean_model_predictions;
        for (j in 1:J) {
          mean_model_predictions[j] = mean(test_f[, n, t, j]);
        }
        test_y_pred[n, t] = normal_rng(alpha[n, val_Nt + t] + dot_product(beta[n, val_Nt + t], mean_model_predictions), sigma[n]);
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
  
  # Extract beta (time-varying weights)
  weights <- extract(fit, pars = "beta")$beta
  
  # Convert weights to a data frame for visualization
  weights_df <- weights
  dimnames(weights_df) <- list(
    iterations = 1:dim(weights)[1],
    individuals = 1:dim(weights)[2],
    time_steps = 1:dim(weights)[3],
    models = paste0("Model_", 1:dim(weights)[4])
  )
  
  # Compute average weights across all iterations
  average_weights <- apply(weights, c(2, 3, 4), mean)
  
  cat("\nAverage weights for each individual and model:\n")
  print(average_weights)
  
  return(list(
    fit = fit,
    log_lik = log_lik,
    loo_result = loo_result,
    test_rmse = mean(extract(fit, pars = "RMSE")$RMSE),
    weights = weights_df
  ))
}
