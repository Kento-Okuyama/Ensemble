kalman_filter <- function(Y1, y2, B1, B2, B3, Lambda1, Q, R, eta1_i0_0, P_i0_0, precomputed_eta2 = NULL) {
  # Y1: N x Nt x O1 array (N subjects, Nt time points, O1 observed variables)
  # y2: N x O2 matrix (N subjects, O2 observed variables)
  # B1: L1 x 1 vector
  # B2: L1 x L1 matrix
  # B3: L1 x L2 matrix
  # Lambda1: O1 x L1 matrix
  # Q: L1 x L1 matrix
  # R: O1 x O1 matrix
  # eta1_i0_0: L1 x 1 vector (initial mean for each subject)
  # P_i0_0: L1 x L1 matrix (initial covariance for each subject)
  # precomputed_eta2: N x L2 matrix (eta2i values for each subject, precomputed from Eq 1)
  
  # Example dimensions
  N <- dim(Y1) # Number of subjects
  Nt <- dim(Y1) # Number of time points
  O1 <- dim(Y1) # Dimension of observed y1
  L1 <- nrow(B2) # Dimension of latent state eta1
  L2 <- ncol(B3) # Dimension of latent state eta2
  
  # Initialize storage for results
  # eta_1it|t: filtered latent state mean
  eta1_filtered <- array(NA, dim=c(k, Nt, N)) 
  # P_it|t: filtered latent state covariance
  P_filtered <- array(NA, dim=c(k, k, Nt, N)) 
  # v_it: one-step-ahead prediction error (innovation)
  v_errors <- array(NA, dim=c(O1, Nt, N)) 
  # F_it: covariance of the innovation
  F_covs <- array(NA, dim=c(O1, O1, Nt, N)) 
  # f(y_1it|...): log-likelihood for each observation
  log_likelihoods <- array(NA, dim=c(Nt, N)) 
  
  # Ensure B1, B3, eta1_i0_0 are column vectors if they are 1D arrays
  if (is.null(dim(B1)) && length(B1) == L1) B1 <- as.matrix(B1)
  if (is.null(dim(B3)) && length(B3) == L1 && L2 == 1) B3 <- as.matrix(B3)
  if (is.null(dim(eta1_i0_0)) && length(eta1_i0_0) == L1) eta1_i0_0 <- as.matrix(eta1_i0_0)
  
  # Identity matrix for Joseph form
  I_mat <- diag(L1) 
  
  for (i in 1:N) {
    # Initialize for subject i
    eta_prev <- eta1_i0_0 # eta_1i,t-1|t-1
    P_prev <- P_i0_0 # P_i,t-1|t-1
    
    # Handle diffuse initialization for P_i0_0 if applicable
    # If P_i0_0 is provided as a zero matrix or indicates diffuse prior
    if (all(diag(P_i0_0) == 0) || is.null(P_i0_0)) {
      P_prev <- diag(L1) * 1e8 # Arbitrarily large constants for diffuse prior
    }
    
    # Extract subject-specific data
    y1_i <- Y1[i,,] # Assuming Y1 is N x Nt x O1
    # Ensure y1_i is a matrix even if Nt=1 or O1=1
    if (Nt == 1) y1_i <- matrix(y1_i, nrow = 1)
    if (O1 == 1 && Nt > 1) y1_i <- matrix(y1_i, ncol = 1)
    
    # Compute eta2i from Equation CFA or use precomputed values
    # For this report, we assume eta2_i is provided as an input (precomputed_eta2)
    # If Lambda2 and y2i were inputs, then:
    # eta2_i <- solve(t(Lambda2) %*% Lambda2) %*% t(Lambda2) %*% y2_i # Example: OLS estimate for eta2i
    eta2_i <- as.matrix(precomputed_eta2[i,]) # Ensure it's a column vector
    
    for (t in 1:Nt) {
      # Extract current observation
      y1_it <- as.matrix(y1_i[t,]) # Ensure it's a column vector
      
      # --- Prediction Step ---
      # Equation (2): Predict latent state mean
      eta_pred <- B1 + B2 %*% eta_prev + B3 %*% eta2_i
      
      # Equation (3): Predict latent state covariance
      P_pred <- B2 %*% P_prev %*% t(B2) + Q
      
      # --- One-Step-Ahead Prediction Error ---
      # Equation (4): Compute innovation
      v_it <- y1_it - Lambda1 %*% eta_pred
      
      # Equation (5): Compute innovation covariance
      F_it <- Lambda1 %*% P_pred %*% t(Lambda1) + R
      
      # --- Kalman Gain Calculation ---
      # Invert F_it (handle potential singularity with pseudo-inverse if needed)
      # For robustness, consider using `ginv` from `MASS` package for pseudo-inverse
      F_it_inv <- tryCatch(solve(F_it), error = function(e) {
        warning("F_it is singular, using pseudo-inverse. Error: ", e$message)
        require(MASS) # Ensure MASS is loaded for ginv
        ginv(F_it)
      })
      K_it <- P_pred %*% t(Lambda1) %*% F_it_inv
      
      # --- Update Step ---
      # Equation (6): Update latent state mean
      eta_updated <- eta_pred + K_it %*% v_it
      
      # Equation (9): Update latent state covariance (Joseph form for numerical stability)
      P_updated <- (I_mat - K_it %*% Lambda1) %*% P_pred %*% t(I_mat - K_it %*% Lambda1) + K_it %*% R %*% t(K_it)
      
      # --- Compute Multivariate Normal Likelihood ---
      # Equation (8):
      # Ensure F_it is positive definite for log(det) and solve
      log_det_F_it <- log(det(F_it))
      
      # Check for numerical stability issues with F_it_inv
      if (any(is.infinite(F_it_inv)) || any(is.nan(F_it_inv))) {
        log_likelihood_it <- -Inf
      } else {
        exponent_term <- -0.5 * t(v_it) %*% F_it_inv %*% v_it
        log_likelihood_it <- -0.5 * O1 * log(2 * pi) - 0.5 * log_det_F_it + exponent_term
      }
      
      # Store results
      eta1_filtered[,t,i] <- eta_updated
      P_filtered[,,t,i] <- P_updated
      v_errors[,t,i] <- v_it
      F_covs[,,t,i] <- F_it
      log_likelihoods[t,i] <- log_likelihood_it
      
      # Prepare for next iteration
      eta_prev <- eta_updated
      P_prev <- P_updated
    }
  }
  
  return(list(
    eta1_filtered = eta1_filtered,
    P_filtered = P_filtered,
    v_errors = v_errors,
    F_covs = F_covs,
    log_likelihoods = log_likelihoods
  ))
}