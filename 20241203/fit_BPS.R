fit_BPS <- function(data, iter = 2000, chains = 4) {
  
  # Stan model code for BPS
  stan_code <- "
  
  data {
    int<lower=1> val_Nt;            // Time series length for validation
    int<lower=1> test_Nt;             // Time series length for testing
    int<lower=1> J;                   // Number of models
    real val_y[val_Nt];        // Validation observed data
    real val_f[val_Nt, J];     // Validation model predictions
    real test_y[test_Nt];          // Test observed data
    real test_f[test_Nt, J];       // Test model predictions
  }
  
  parameters {
    vector[val_Nt + test_Nt] alpha;         // Time-varying intercept
    vector[J] beta[val_Nt + test_Nt];         // Time-varying weights for each time step
    real<lower=0> sigma;      // Observation noise standard deviation
    real<lower=0, upper=1> tau;      // Random walk noise standard deviation (alpha)
    real<lower=0, upper=1> eta;      // Random walk noise standard deviation (beta)
  }
  
  transformed parameters {
    vector[J] mu;
    mu = rep_vector(1.0/J, J);
  }
  model {
    // Priors
    sigma ~ normal(0, 1);
    tau ~ normal(0, 0.5);
    eta ~ normal(0, 0.3);
    
    alpha[1] ~ normal(0, tau);
    beta[1] ~ normal(mu, eta);
    
    for (t in 2:(val_Nt + test_Nt)) {
      alpha[t] ~ normal(alpha[t-1], tau);  // gaussian prior for intercepts
      beta[t] ~ normal(beta[t-1], eta);  // gaussian prior for weights
    }
  
    // Likelihood
    for (t in 1:val_Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = val_f[t, j];
      }
      val_y[t] ~ normal(alpha[t] + dot_product(beta[t], model_predictions), sigma);
    }
  }
  
  generated quantities {
    real log_lik[val_Nt];  // Log-likelihood for PSIS-LOO
    real test_y_pred[test_Nt]; // Predicted values for test data
    real SSE;
    real RMSE; 
    
    SSE = 0;
    for (t in 1:val_Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = val_f[t, j];
      }
      log_lik[t] = normal_lpdf(val_y[t] | alpha[t] + dot_product(beta[t], model_predictions), sigma);
    }
    
    for (t in 1:test_Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = test_f[t, j];
      }
      test_y_pred[t] = normal_rng(alpha[val_Nt + t] + dot_product(beta[val_Nt + t], model_predictions), sigma);
      SSE += square(test_y[t] - test_y_pred[t]);
    }
    RMSE = sqrt(SSE / test_Nt); 
  }
  "
  
  # Compile and fit BPS model
  fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
  
  # Extract log-likelihood for PSIS-LOO cross-validation
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  print(loo_result)
  
  # Extract RMSE
  test_rmse <- mean(extract(fit, "RMSE")$RMSE)
  cat("\n", "Average RMSE across all samples:", test_rmse, "\n")
  
  # Plot posterior distributions of stacking weights
  post_alpha_beta <- bayesplot::mcmc_areas(as.array(fit), pars = c("alpha[1]", "beta[1,1]", "beta[1,2]")) +
    labs(title = "Posterior Distributions of model intercept and weights")
  
  post_plot <- list(post_alpha_beta = post_alpha_beta)
  
  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, test_rmse = test_rmse, post_plot = post_plot))
}
