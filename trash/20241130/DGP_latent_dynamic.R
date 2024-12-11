DGP_latent <- function(Nt = 30, seed = 123, train_ratio = 0.8) {
  # Set the seed for reproducibility
  set.seed(seed)
  
  # Parameters for data simulation
  balance <- 0.5           # Weighting factor for target_mean calculation
  mu <- c(2, 1, 0)         # Means for each regime
  ar_pos <- 0.7            # Positive AR coefficient for regime 1
  ma_coef <- 0.5           # MA coefficient for regime 2
  lambda <- 1              # Factor loading for eta -> y
  sigma_m <- 1             # Measurement noise standard deviation
  sigma_s <- c(0.5, 0.6, 0.4) # Standard deviations for each regime
  small_prob <- 0.01       # Small constant probability for white noise transitions
  
  # Sigmoid function for monotonic transitions
  sigmoid <- function(x) {
    1 / (1 + exp(-x))
  }
  
  # Function to calculate dynamic transition matrix
  calculate_transition_matrix <- function(eta, alpha = 1, beta = 1) {
    # Dynamic transition probabilities (monotonic increasing function)
    p_11 <- sigmoid(alpha + beta * eta)  # AR to AR
    p_21 <- sigmoid(alpha + beta * eta)  # MA to AR
    
    # Transition matrix
    transition_matrix <- matrix(
      c(
        p_11 - small_prob, p_11, small_prob,  # Row 1: From AR(1)
        p_21 - small_prob, 1 - p_21, small_prob,  # Row 2: From MA(1)
        0.3, 0.3, 0.4                             # Row 3: From White Noise
      ),
      nrow = 3, byrow = TRUE
    )
    
    # Normalize each row to ensure valid probabilities
    for (i in 1:nrow(transition_matrix)) {
      transition_matrix[i, ] <- transition_matrix[i, ] / sum(transition_matrix[i, ])
    }
    
    return(transition_matrix)
  }
  
  # Initialize matrices to store simulated data
  eta <- rep(NA, length = Nt)      # Latent states
  y <- rep(NA, length = Nt)        # Observed data
  regime <- rep(NA, length = Nt)   # Regime labels
  
  # Simulate data with regime transitions
  # Initialize the first time step
  current_regime <- 1  # Start with regime 1
  regime[1] <- current_regime  # Record the regime
  eta[1] <- rnorm(1, mu[current_regime], sigma_s[current_regime])  # Initial latent state
  y[1] <- lambda * eta[1] + rnorm(1, 0, sigma_m)  # Initial observed data
  
  # Simulate the remaining time steps
  for (t in 2:Nt) {
    # Dynamically calculate the transition matrix
    transition_matrix <- calculate_transition_matrix(eta[t - 1], alpha = 1, beta = 1)
    
    # Get the current transition probabilities from the matrix
    current_transition_probs <- transition_matrix[current_regime, ]
    
    # Sample the next regime based on the transition probabilities
    current_regime <- sample(1:3, 1, prob = current_transition_probs)
    regime[t] <- current_regime  # Record the current regime
    
    # Calculate target mean for eta based on the balance parameter
    target_mean <- balance * mu[current_regime] + (1 - balance) * eta[t - 1]
    
    # Generate eta based on the current regime
    if (current_regime == 1) {
      # Regime 1: AR(1) with positive coefficient
      eta[t] <- (1 - ar_pos) * target_mean + ar_pos * eta[t - 1] +
        rnorm(1, 0, sigma_s[current_regime])
    } else if (current_regime == 2) {
      # Regime 2: MA(1) process
      eta[t] <- target_mean + ma_coef * (eta[t - 1] - target_mean) +
        rnorm(1, 0, sigma_s[current_regime])
    } else {
      # Regime 3: Pure white noise
      eta[t] <- rnorm(1, target_mean, sigma_s[current_regime])
    }
    
    # Generate observed data y from eta
    y[t] <- lambda * eta[t] + rnorm(1, 0, sigma_m)
  }
  
  # Split data into training and testing sets
  train_length <- floor(Nt * train_ratio)  # Number of training time points
  test_length <- Nt - train_length         # Number of testing time points
  
  # Training and testing data
  train_eta <- eta[1:train_length]       # Training latent factor
  train_y <- y[1:train_length]           # Training observed data
  train_regime <- regime[1:train_length] # Training regime data
  test_eta <- eta[(train_length + 1):Nt] # Testing latent factor
  test_y <- y[(train_length + 1):Nt]     # Testing observed data
  test_regime <- regime[(train_length + 1):Nt] # Testing regime data
  
  # Prepare data list for Stan
  stan_data <- list(
    train_Nt = train_length,   # Length of training time series
    test_Nt = test_length,     # Length of testing time series
    train_eta = train_eta,     # Training latent factor
    test_eta = test_eta,       # Testing latent factor
    train_y = train_y,         # Training observed data
    test_y = test_y,           # Testing observed data
    train_regime = train_regime, # Training regimes
    test_regime = test_regime   # Testing regimes
  )
  
  return(stan_data)
}
