# ============  MA(1) KFAS  ============ #
fit_ma_kfas <- function(data) {
  # data$train_y : N x train_Nt
  
  # 1) Reshape
  Y <- t(data$train_y)
  N <- data$N
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
  
  # 3) updatefn
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
    
    # Z = [ I_N  0 ]
    Z_new <- cbind(diag(1, N), matrix(0, N, N))
    model$Z[,,1] <- Z_new
    
    # T = block( zeroN, theta*I; I, zeroN )
    zeroN <- matrix(0, N, N)
    eyeN  <- diag(1, N)
    T_mat <- rbind(
      cbind(zeroN,    theta*eyeN),
      cbind(eyeN,     zeroN)
    )
    model$T[,,1] <- T_mat
    
    # Q => noise on "delta_now" part
    Q_mat <- rbind(
      cbind(sigSys^2 * eyeN, zeroN),
      cbind(zeroN,           zeroN)
    )
    model$Q[,,1] <- Q_mat
    
    # H => observation noise
    model$H[,,1] <- diag(sigObs^2, N)
    
    return(model)
  }
  
  # 4) fitSSM
  fit_ma <- fitSSM(
    model = model_ma,
    inits = c(0.3, 0.0, log(0.1), log(0.1)),
    updatefn = updatefn_ma,
    method = "BFGS"
  )
  
  # フィルタ & 平滑化 (stateを指定)
  kfs_ma <- KFS(
    fit_ma$model,
    filtering = c("state","mean"),
    smoothing = c("state","mean")
  )
  
  # 5) Forecast
  pred_ma_list <- predict(fit_ma$model, n.ahead = horizon)
  
  # Parameter estimates => add mu back
  par_est   <- fit_ma$optim.out$par
  theta_hat <- par_est[1]
  mu_hat    <- par_est[2]
  
  pred_ma_mat <- do.call(cbind, lapply(pred_ma_list, function(ts_obj) {
    as.numeric(ts_obj) + mu_hat
  }))
  colnames(pred_ma_mat) <- paste0("y", 1:N)
  
  return(list(
    fit_ma     = fit_ma,
    kfs_ma     = kfs_ma,         # kfs_ma$alphahat がちゃんと入る
    theta_hat  = theta_hat,
    mu_hat     = mu_hat,
    pred_ma    = pred_ma_mat     # (horizon) x N
  ))
}

