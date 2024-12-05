fit_BPS2 <- function(data, iter = 2000, chains = 4) {
  # ===========================
  #    Stan Model Code
  # ===========================
  # Bayesian Predictive Stacking with Markov-switching (BPS2)
  stan_code <- "
  data {
    int<lower=1> train_Nt;             // Training data length
    int<lower=1> val_Nt;              // Validation data length
    int<lower=1> test_Nt;             // Test data length
    int<lower=1> J;                   // Number of models
    vector[J] pred_e[train_Nt + val_Nt + test_Nt]; // Latent predictions
    real val_y[val_Nt];               // Validation observed data
    real val_f[val_Nt, J];            // Validation model predictions
    real test_y[test_Nt];             // Test observed data
    real test_f[test_Nt, J];          // Test model predictions
  }
  
  parameters {
    simplex[J] mu;
    real alpha[val_Nt + test_Nt];     // Time-varying intercept
    vector[J] beta[val_Nt + test_Nt]; // Time-varying weights
    vector[J] delta;                  // Markov-switching coefficients
    real<lower=0> sigma;              // Observation noise standard deviation
    real<lower=0, upper=1> tau_a;     // Random walk noise for alpha
    real<lower=0, upper=1> tau_b;     // Random walk noise for beta
    real<lower=0, upper=1> tau_d;     // Random walk noise for delta
  }

  model {
    // Priors
    mu ~ dirichlet(rep_vector(1.0, J));   // Dirichlet prior for initial model weights
    sigma ~ normal(0, 1);
    tau_a ~ normal(0, 0.5);
    tau_b ~ normal(0, 0.3);
    tau_d ~ normal(0, 0.1);
    
    alpha[1] ~ normal(0, tau_a);      // Initial intercept
    beta[1] ~ normal(mu, tau_b);      // Initial weights
    delta ~ normal(0, tau_d);         // Markov-switching coefficients
    
    // Random walk for alpha and beta
    for (t in 2:(val_Nt + test_Nt)) {
      alpha[t] ~ normal(alpha[t - 1], tau_a);
      beta[t] ~ normal(beta[t - 1] + pred_e[train_Nt + t - 1] .* delta, tau_b);
    }

    // Likelihood for validation data
    for (t in 1:val_Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = val_f[t, j];
      }
      val_y[t] ~ normal(alpha[t] + dot_product(beta[t], model_predictions), sigma);
    }
  }
  
  generated quantities {
    real log_lik[val_Nt];          // Log-likelihood for PSIS-LOO
    real test_y_pred[test_Nt];     // Predicted values for test data
    real SSE;                      // Sum of Squared Errors
    real RMSE;                     // Root Mean Square Error

    SSE = 0;

    // Log-likelihood for validation data
    for (t in 1:val_Nt) {
      vector[J] model_predictions;
      for (j in 1:J) {
        model_predictions[j] = val_f[t, j];
      }
      log_lik[t] = normal_lpdf(val_y[t] | alpha[t] + dot_product(beta[t], model_predictions), sigma);
    }

    // Predictions for test data and RMSE calculation
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
  
  # ===========================
  #    Fit the Stan Model
  # ===========================
  fit <- tryCatch({
    stan(model_code = stan_code, data = data, iter = iter, chains = chains)
  }, error = function(e) {
    stop("Stan model fitting failed: ", e)
  })
  
  # ===========================
  #    Extract and Process Results
  # ===========================
  # Extract log-likelihood for PSIS-LOO cross-validation
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  
  # Print LOO results
  print(loo_result)
  
  # Extract test RMSE
  test_rmse <- mean(extract(fit, "RMSE")$RMSE)
  cat("\n", "Average RMSE across all samples:", test_rmse, "\n")
  
  # ===========================
  #    Visualize Posterior Distributions
  # ===========================
  # Posterior distributions for intercept, weights, and delta
  post_alpha_beta <- bayesplot::mcmc_areas(
    as.array(fit),
    pars = c("alpha[1]", "beta[1,1]", "beta[1,2]", "delta[1]", "delta[2]")
  ) +
    labs(
      title = "Posterior Distributions of Intercept, Weights, and Delta",
      x = "Parameter Value",
      y = "Density"
    )
  
  # Prepare plot list
  post_plot <- list(post_alpha_beta = post_alpha_beta)
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    fit = fit,
    log_lik = log_lik,
    loo_result = loo_result,
    test_rmse = test_rmse,
    post_plot = post_plot
  ))
}
