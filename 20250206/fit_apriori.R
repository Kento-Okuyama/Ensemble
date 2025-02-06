# ============  統合関数 (Stan版との互換)  ============ #
fit_apriori <- function(data) {
  # 1) AR(1) & MA(1) fit via KFAS
  res_ar_kfas <- fit_ar_kfas(data) 
  res_ma_kfas <- fit_ma_kfas(data) 
  
  # 2) Get state estimates
  kfs_ar   <- res_ar_kfas$kfs_ar
  kfs_ma   <- res_ma_kfas$kfs_ma
  alph_ar  <- kfs_ar$alphahat      # now not NULL
  alph_ma  <- kfs_ma$alphahat      # now not NULL
  
  # (以下、元のコードとほぼ同じ処理)
  # ...
  #   - Combine train states + future predictions
  #   - Build pred_e, pred_f as 3D arrays
  #   - Slice them into train_e, val_e, test_e etc.
  
  # 以下はオリジナルコードのまま（省略）。 
  # ただし alph_ar, alph_ma が実際に非NULLになったはずなので、 
  # AR(1)モデルのtrain_Nt x N, MA(1)モデルの train_Nt x (2N) などが取得できます。
  
  #---------------------------------------------
  # (同じコードを再掲) 
  #---------------------------------------------
  N        <- data$N
  train_Nt <- data$train_Nt
  val_Nt   <- data$val_Nt
  test_Nt  <- data$test_Nt
  horizon  <- val_Nt + test_Nt
  
  # MA(1) => "delta_now" part
  alph_ma_now <- alph_ma[, 1:N, drop=FALSE]
  
  e_ar_train <- alph_ar
  e_ar_test  <- res_ar_kfas$pred_ar  # horizon x N
  e_ar_full  <- rbind(e_ar_train, e_ar_test)
  
  e_ma_train <- alph_ma_now
  e_ma_test  <- res_ma_kfas$pred_ma
  e_ma_full  <- rbind(e_ma_train, e_ma_test)
  
  e_ar_t <- t(e_ar_full)
  e_ma_t <- t(e_ma_full)
  
  pred_e <- array(NA, c(N, train_Nt + val_Nt + test_Nt, 2))
  pred_e[,,1] <- e_ar_t
  pred_e[,,2] <- e_ma_t
  
  f_ar_train <- alph_ar
  f_ar_full  <- rbind(f_ar_train, res_ar_kfas$pred_ar)
  f_ar_t     <- t(f_ar_full)
  
  f_ma_train <- alph_ma_now
  f_ma_full  <- rbind(f_ma_train, res_ma_kfas$pred_ma)
  f_ma_t     <- t(f_ma_full)
  
  pred_f <- array(NA, c(N, train_Nt + val_Nt + test_Nt, 2))
  pred_f[,,1] <- f_ar_t
  pred_f[,,2] <- f_ma_t
  
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
  
  # res_ar, res_ma
  res_ar <- list(
    fit_ar   = res_ar_kfas$fit_ar,
    kfs_ar   = res_ar_kfas$kfs_ar
  )
  res_ma <- list(
    fit_ma   = res_ma_kfas$fit_ma,
    kfs_ma   = res_ma_kfas$kfs_ma
  )
  
  return(list(
    data_fit = data_fit,
    res_ar   = res_ar,
    res_ma   = res_ma
  ))
}