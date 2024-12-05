for (i in 46:46) {
  # Update seed
  seed <- 123 + i  # Change seed for each iteration
  
  # Generate data
  df <- DGP(Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)
  
  # Fit models
  res_apriori <- fit_apriori(data = df, iter = n_iter, chains = n_chains)
  res_BMA <- fit_BMA(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
  res_BPS <- fit_BPS(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
  res_BPS2 <- fit_BPS2(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
  
  # Extract results
  results <- data.frame(
    run = i,
    elpd_loo_AR = res_apriori$res_ar$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_MA = res_apriori$res_ma$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_BMA = res_BMA$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_BPS = res_BPS$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_BPS2 = res_BPS2$loo_result$estimates["elpd_loo", "Estimate"],
    test_rmse_AR = res_apriori$res_ar$test_rmse,
    test_rmse_MA = res_apriori$res_ma$test_rmse,
    test_rmse_BMA = res_BMA$test_rmse,
    test_rmse_BPS = res_BPS$test_rmse,
    test_rmse_BPS2 = res_BPS2$test_rmse
  )
  
  # Append to list
  result_list[[i]] <- results
  
  # Update progress bar
  setTxtProgressBar(pb, i)
}
