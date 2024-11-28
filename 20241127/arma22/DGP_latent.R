DGP_latent <- function(N = 5, Nt = 30, seed = 123, train_ratio = 0.8) {
  # Set the seed for reproducibility
  set.seed(seed)
  
  # Parameters for ARMA(2,2) model
  phi <- c(0.7, -0.3)  # AR coefficients
  theta <- c(0.5, -0.2)  # MA coefficients
  sigma <- 1            # Standard deviation of noise
  lambda <- 1           # Factor loading for eta -> y
  
  # Initialize matrices to store simulated data
  eta <- matrix(0, nrow = N, ncol = Nt)  # Latent states
  y <- matrix(0, nrow = N, ncol = Nt)    # Observed data
  epsilon <- matrix(0, nrow = N, ncol = Nt) # Noise (MA component)
  
  # Simulate ARMA(2,2) data
  for (n in 1:N) {
    for (t in 1:Nt) {
      # Generate white noise
      epsilon[n, t] <- rnorm(1, mean = 0, sd = sigma)
      
      if (t == 1) {
        # For the first time point, assume eta starts at noise
        eta[n, t] <- epsilon[n, t]
      } else if (t == 2) {
        # For the second time point, AR(1) + MA(1) terms are used
        eta[n, t] <- phi[1] * eta[n, t - 1] +
          theta[1] * epsilon[n, t - 1] +
          epsilon[n, t]
      } else {
        # For t >= 3, ARMA(2,2) full model is used
        eta[n, t] <- phi[1] * eta[n, t - 1] +
          phi[2] * eta[n, t - 2] +
          theta[1] * epsilon[n, t - 1] +
          theta[2] * epsilon[n, t - 2] +
          epsilon[n, t]
      }
      
      # Generate observed data y from eta
      y[n, t] <- lambda * eta[n, t] + rnorm(1, mean = 0, sd = sigma)
    }
  }
  
  # Split data into training and testing sets
  train_length <- floor(Nt * train_ratio) # Number of training time points
  test_length <- Nt - train_length        # Number of testing time points
  
  # Training and testing data
  train_eta <- eta[, 1:train_length]   # Training latent factor
  train_y <- y[, 1:train_length]       # Training observed data
  test_eta <- eta[, (train_length + 1):Nt] # Testing latent factor
  test_y <- y[, (train_length + 1):Nt]     # Testing observed data
  
  # Prepare data list for Stan or other models
  data <- list(
    N = N,                     # Number of time series
    train_Nt = train_length,   # Length of training time series
    test_Nt = test_length,     # Length of testing time series
    train_eta = train_eta,     # Training latent factor
    test_eta = test_eta,       # Testing latent factor
    train_y = train_y,         # Training observed data
    test_y = test_y            # Testing observed data
  )
  
  return(data)
}
