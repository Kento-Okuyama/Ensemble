set.seed(123)  # Seed for reproducibility

# Step 1: Initialize the sample size N and the number of measurement occasions Nt
N <- 30  # Sample size
Nt <- 50   # Number of time points
burnin <- 50  # Burn-in period

# Step 2: Initialize y_1i0, y_2i, and S_i0
y_1i0 <- rnorm(N, mean = 0, sd = 1)  # Initial values for y_1i0
y_2i <- rnorm(N, mean = 0, sd = 1)   # Individual-specific covariates y_2i
S_i0 <- rbinom(N, 1, 0.5)            # Initial state S_i0 is either 0 or 1

# Adjust the parameters to discourage switching back to the previous regime
gamma <- list(
  "0" = -c(2.5, 0.3, 0.2, 0.1),  # γ_1^0 = -2.5 makes switching from state 0 to state 1 less likely
  "1" = c(2.5, 0.1, 0.05, 0.05)  # γ_1^1 = 2.5 makes staying in state 1 more likely
)

# Structural model parameter settings (example initial values)
c_s <- c(0.5, 0.1)       # c^0, c^1
phi_s <- c(0.3, 0.7)     # φ^0, φ^1
sigma_SM <- c(1.0, 1.5)  # σ_SM^0, σ_SM^1

# Definition of the sigmoid function
sigmoid <- function(x) {
  ifelse(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))
}

# Matrices to store the results
S_it <- matrix(NA, nrow = N, ncol = (burnin + Nt))  # States S_it
y_1it <- matrix(NA, nrow = N, ncol = (burnin + Nt)) # Observed values y_1it

# Step 3: Loop over time steps t = 1,...,Nt
for (t in 1:(burnin + Nt)) {
  for (i in 1:N) {
    # Calculate the transition probability p_it^1,S_{i0} based on the previous state
    if (t == 1) {
      prev_S <- S_i0[i]
      prev_y1 <- y_1i0[i]
    } else {
      prev_S <- S_it[i, t-1]
      prev_y1 <- y_1it[i, t-1]
    }
    
    # Calculate the transition probability p_it^1,s' depending on the state S_it
    p_it_1_0 <- sigmoid(gamma[[as.character(prev_S)]][1] + 
                          gamma[[as.character(prev_S)]][2] * prev_y1 + 
                          gamma[[as.character(prev_S)]][3] * y_2i[i] + 
                          gamma[[as.character(prev_S)]][4] * prev_y1 * y_2i[i])
    
    # Step 3b: Sample state S_{it} from a Bernoulli distribution
    S_it[i, t] <- rbinom(1, 1, p_it_1_0)
    
    # Step 3c: Sample y_{1it}
    S_curr <- S_it[i, t]
    y_1it[i, t] <- rnorm(1, mean = c_s[S_curr + 1] + phi_s[S_curr + 1] * prev_y1, sd = sigma_SM[S_curr + 1])
  }
}

# Exclude burn-in period
S_it <- S_it[, burnin + 1:Nt]
y_1it <- y_1it[, burnin + 1:Nt]

# Check output (optional)
print(S_it)
print(y_1it)
