repetition <- function(k=1, n=1){
  seed <- 123 + k # Set the random seed for reproducibility
  
  # Parameters for data simulation
  Nt <- 100  # Length of each time series
  
  # ===========================
  #    Simulate Data
  # ===========================
  # Generate data using the custom DGP function
  df <- DGP(N = N, Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)
  
  # Summarize the generated data's regimes
  table(c(df$train_regime[n,], df$val_regime[n,], df$test_regime[n,]))
  
  # Visualize the regimes across train, validation, and test sets
  plot(c(df$train_regime[n,], df$val_regime[n,], df$test_regime[n,]), type = "l", 
       main = "Regime Visualization", xlab = "Time", ylab = "Regime")
  
  # Visualize the eta across train, validation, and test sets
  plot(c(df$train_eta[n,], df$val_eta[n,], df$test_eta[n,]), type = "l", 
       main = "Eta Visualization", xlab = "Time", ylab = "Eta")
}

for (n in 1:N) {
  repetition(k=1, n=n)
}
