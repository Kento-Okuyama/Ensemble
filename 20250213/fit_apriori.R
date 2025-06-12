# This complete code adds observation noise to 'F' so that F != E in both AR(1) and MA(1).
# All comments are in English.

library(KFAS)

# =======================
#     AR(1) KFAS
# =======================
fit_ar_kfas <- function(data) {
  # data$train_y : N x train_Nt
  # 1) Reshape data for KFAS: time=rows, series=cols
  Y <- t(data$train_y)  # (train_Nt) x N
  
  N        <- data$N
  train_Nt <- data$train_Nt
  val_Nt   <- data$val_Nt
  test_Nt  <- data$test_Nt
  horizon  <- val_Nt + test_Nt
  
  # 2) Build AR(1) model with a common phi
  model_ar <- SSModel(
    Y ~ -1 + SSMcustom(
      Z = diag(1, nrow = N),
      T = diag(NA, nrow = N),
      R = diag(1, nrow = N),
      Q = diag(NA, nrow = N),
      state_names = paste0("state_", 1:N)
    ),
    H = diag(NA, nrow = N)
  )
  
  # 3) Update function
  updatefn_ar <- function(pars, model) {
    # pars[1] => phi in (-1,1) via tanh
    # pars[2] => sigma_q = exp(pars[2])
    # pars[3] => sigma_h = exp(pars[3])
    phi  <- tanh(pars[1])
    sigQ <- exp(pars[2])
    sigH <- exp(pars[3])
    
    model$T[,,1] <- diag(phi, nrow = N)
    model$Q[,,1] <- diag(sigQ^2, nrow = N)
    model$H[,,1] <- diag(sigH^2, nrow = N)
    model
  }
  
  # 4) Fit SSM
  fit_ar <- fitSSM(
    model   = model_ar,
    inits   = c(0.0, log(0.1), log(0.1)),
    updatefn = updatefn_ar,
    method  = "BFGS"
  )
  
  # 5) KFS
  kfs_ar <- KFS(
    fit_ar$model,
    filtering = c("state","mean"),
    smoothing = c("state","mean")
  )
  
  # 6) Forecast val + test steps
  pred_ar_list <- predict(fit_ar$model, n.ahead = horizon)
  pred_ar_mat  <- do.call(cbind, lapply(pred_ar_list, as.numeric))
  colnames(pred_ar_mat) <- paste0("y", 1:N)
  
  list(
    fit_ar   = fit_ar,
    kfs_ar   = kfs_ar,      # kfs_ar$alphahat is the smoothed states
    pred_ar  = pred_ar_mat  # (horizon) x N
  )
}

# =======================
#     MA(1) KFAS
# =======================
fit_ma_kfas <- function(data) {
  # data$train_y : N x train_Nt
  # 1) Reshape data
  Y <- t(data$train_y)
  
  N        <- data$N
  train_Nt <- data$train_Nt
  val_Nt   <- data$val_Nt
  test_Nt  <- data$test_Nt
  horizon  <- val_Nt + test_Nt
  
  # 2) Build MA(1) model
  model_ma <- SSModel(
    Y ~ -1 + SSMcustom(
      Z = matrix(NA, nrow = N, ncol = 2*N),
      T = matrix(NA, nrow = 2*N, ncol = 2*N),
      R = diag(1, nrow = 2*N),
      Q = diag(NA, nrow = 2*N),
      state_names = c(
        paste0("delta_now_", 1:N),
        paste0("delta_prev_", 1:N)
      )
    ),
    H = diag(NA, nrow = N)
  )
  
  # 3) Update function
  updatefn_ma <- function(pars, model) {
    # pars[1] = theta
    # pars[2] = mu
    # pars[3] = log(sigma_ma)
    # pars[4] = log(sigma_obs)
    theta  <- pars[1]
    mu     <- pars[2]
    sigSys <- exp(pars[3])
    sigObs <- exp(pars[4])
    
    # Shift observations by mu
    model$y <- model$y - mu
    
    # Z = [I_N  0]
    Z_new <- cbind(diag(1, N), matrix(0, N, N))
    model$Z[,,1] <- Z_new
    
    # T block
    zeroN <- matrix(0, N, N)
    eyeN  <- diag(1, N)
    T_mat <- rbind(
      cbind(zeroN,    theta * eyeN),
      cbind(eyeN,     zeroN)
    )
    model$T[,,1] <- T_mat
    
    # Q => noise on "delta_now" states
    Q_mat <- rbind(
      cbind(sigSys^2 * eyeN, zeroN),
      cbind(zeroN,           zeroN)
    )
    model$Q[,,1] <- Q_mat
    
    # H => observation noise
    model$H[,,1] <- diag(sigObs^2, N)
    model
  }
  
  # 4) Fit SSM
  fit_ma <- fitSSM(
    model = model_ma,
    inits = c(0.3, 0.0, log(0.1), log(0.1)),
    updatefn = updatefn_ma,
    method = "BFGS"
  )
  
  # 5) KFS
  kfs_ma <- KFS(
    fit_ma$model,
    filtering = c("state","mean"),
    smoothing = c("state","mean")
  )
  
  # 6) Forecast
  pred_ma_list <- predict(fit_ma$model, n.ahead = horizon)
  
  # Get parameter estimates
  par_est   <- fit_ma$optim.out$par
  theta_hat <- par_est[1]
  mu_hat    <- par_est[2]
  
  # Add mu back to the predicted values
  pred_ma_mat <- do.call(cbind, lapply(pred_ma_list, function(ts_obj) {
    as.numeric(ts_obj) + mu_hat
  }))
  colnames(pred_ma_mat) <- paste0("y", 1:N)
  
  list(
    fit_ma     = fit_ma,
    kfs_ma     = kfs_ma,
    theta_hat  = theta_hat,
    mu_hat     = mu_hat,
    pred_ma    = pred_ma_mat  # (horizon) x N
  )
}

# ========================
#   Fit AR(1) & MA(1)
#   Then combine results
# ========================
fit_apriori <- function(data) {
  # 1) Fit AR(1) & MA(1) via KFAS
  res_ar_kfas <- fit_ar_kfas(data)
  res_ma_kfas <- fit_ma_kfas(data)
  
  # 2) Get state estimates
  kfs_ar <- res_ar_kfas$kfs_ar
  kfs_ma <- res_ma_kfas$kfs_ma
  alph_ar <- kfs_ar$alphahat
  alph_ma <- kfs_ma$alphahat
  
  N        <- data$N
  train_Nt <- data$train_Nt
  val_Nt   <- data$val_Nt
  test_Nt  <- data$test_Nt
  horizon  <- val_Nt + test_Nt
  
  # MA(1) => "delta_now" part
  alph_ma_now <- alph_ma[, 1:N, drop = FALSE]
  
  # E: smoothed states for AR(1) and MA(1)
  e_ar_train <- alph_ar
  e_ar_test  <- res_ar_kfas$pred_ar
  e_ar_full  <- rbind(e_ar_train, e_ar_test)
  
  e_ma_train <- alph_ma_now
  e_ma_test  <- res_ma_kfas$pred_ma
  e_ma_full  <- rbind(e_ma_train, e_ma_test)
  
  e_ar_t <- t(e_ar_full)
  e_ma_t <- t(e_ma_full)
  
  pred_e <- array(NA, c(N, train_Nt + val_Nt + test_Nt, 2))
  pred_e[,,1] <- e_ar_t
  pred_e[,,2] <- e_ma_t
  
  # F: predicted observations for AR(1) and MA(1)
  f_ar_train <- alph_ar
  f_ar_test  <- res_ar_kfas$pred_ar
  f_ar_full  <- rbind(f_ar_train, f_ar_test)
  
  f_ma_train <- alph_ma_now
  f_ma_test  <- res_ma_kfas$pred_ma
  f_ma_full  <- rbind(f_ma_train, f_ma_test)
  
  # Add observation noise to AR(1)
  sigH_ar <- exp(res_ar_kfas$fit_ar$optim.out$par[3])  # observation noise from AR
  f_ar_train <- f_ar_train + matrix(
    rnorm(length(f_ar_train), 0, sigH_ar),
    nrow = nrow(f_ar_train), ncol = ncol(f_ar_train)
  )
  f_ar_test <- f_ar_test + matrix(
    rnorm(length(f_ar_test), 0, sigH_ar),
    nrow = nrow(f_ar_test), ncol = ncol(f_ar_test)
  )
  f_ar_full <- rbind(f_ar_train, f_ar_test)
  f_ar_t <- t(f_ar_full)
  
  # Add observation noise to MA(1)
  sigH_ma <- exp(res_ma_kfas$fit_ma$optim.out$par[4])  # observation noise from MA
  f_ma_train <- f_ma_train + matrix(
    rnorm(length(f_ma_train), 0, sigH_ma),
    nrow = nrow(f_ma_train), ncol = ncol(f_ma_train)
  )
  f_ma_test <- f_ma_test + matrix(
    rnorm(length(f_ma_test), 0, sigH_ma),
    nrow = nrow(f_ma_test), ncol = ncol(f_ma_test)
  )
  f_ma_full <- rbind(f_ma_train, f_ma_test)
  f_ma_t <- t(f_ma_full)
  
  # Combine everything into pred_f
  pred_f <- array(NA, c(N, train_Nt + val_Nt + test_Nt, 2))
  pred_f[,,1] <- f_ar_t
  pred_f[,,2] <- f_ma_t
  
  # Subset train/val/test
  train_e <- pred_e[, 1:train_Nt, ]
  val_e   <- pred_e[, (train_Nt+1):(train_Nt+val_Nt), ]
  test_e  <- pred_e[, (train_Nt+val_Nt+1):(train_Nt+val_Nt+test_Nt), ]
  
  train_f <- pred_f[, 1:train_Nt, ]
  val_f   <- pred_f[, (train_Nt+1):(train_Nt+val_Nt), ]
  test_f  <- pred_f[, (train_Nt+val_Nt+1):(train_Nt+val_Nt+test_Nt), ]
  
  data_fit <- list(
    N        = data$N,
    J        = 2,
    train_Nt = data$train_Nt,
    val_Nt   = data$val_Nt,
    test_Nt  = data$test_Nt,
    train_y  = data$train_y,
    val_y    = data$val_y,
    test_y   = data$test_y,
    pred_e   = pred_e,
    pred_f   = pred_f,
    train_e  = train_e,
    val_e    = val_e,
    test_e   = test_e,
    train_f  = train_f,
    val_f    = val_f,
    test_f   = test_f
  )
  
  # 3) Results
  res_ar <- list(
    fit_ar = res_ar_kfas$fit_ar,
    kfs_ar = res_ar_kfas$kfs_ar
  )
  res_ma <- list(
    fit_ma = res_ma_kfas$fit_ma,
    kfs_ma = res_ma_kfas$kfs_ma
  )
  
  # 4) Compute test RMSE for AR(1) and MA(1) using pred_f as the predicted observations
  SSE_ar <- 0
  SSE_ma <- 0
  for (n in 1:N) {
    for (t in 1:test_Nt) {
      SSE_ar <- SSE_ar + (data$test_y[n, t] - test_f[n, t, 1])^2
      SSE_ma <- SSE_ma + (data$test_y[n, t] - test_f[n, t, 2])^2
    }
  }
  test_rmse_ar <- sqrt(SSE_ar / (N * test_Nt))
  test_rmse_ma <- sqrt(SSE_ma / (N * test_Nt))
  
  list(
    data_fit      = data_fit,
    res_ar        = res_ar,
    res_ma        = res_ma,
    test_rmse_ar  = test_rmse_ar,
    test_rmse_ma  = test_rmse_ma
  )
}
