# 必要なライブラリをインストールしてロード
library(torch)
library(dplyr)

# ----------------------------------------------------------------
# Part 1: カルマンフィルター (変更なし)
# ----------------------------------------------------------------
kalman_filter_torch <- function(Y1, precomputed_eta2, b0, B1, b2, Lambda1, Q, R, eta1_i0_0, P_i0_0) {
  dims <- Y1$shape
  N <- dims[1]; Nt <- dims[2]; O1 <- dims[3]; L1 <- B1$shape[1]; L2 <- b2$shape[2]
  
  total_log_likelihood <- torch_tensor(0, dtype = torch_float(), device = Y1$device)
  I_mat <- torch_eye(L1, device = Y1$device)
  
  for (i in 1:N) {
    eta_prev <- eta1_i0_0$clone()
    P_prev <- P_i0_0$clone()
    y1_i <- Y1[i, , ]
    eta2_i <- precomputed_eta2[i, ]$reshape(c(L2, 1))
    
    for (t in 1:Nt) {
      y1_it <- y1_i[t, ]$reshape(c(O1, 1))
      
      eta_pred <- b0 + torch_matmul(B1, eta_prev) + torch_matmul(b2, eta2_i)
      P_pred <- torch_matmul(torch_matmul(B1, P_prev), B1$transpose(1, 2)) + Q
      
      v_it <- y1_it - torch_matmul(Lambda1, eta_pred)
      F_it <- torch_matmul(torch_matmul(Lambda1, P_pred), Lambda1$transpose(1, 2)) + R
      F_it_inv <- torch_pinverse(F_it)
      K_it <- torch_matmul(torch_matmul(P_pred, Lambda1$transpose(1, 2)), F_it_inv)
      
      eta_updated <- eta_pred + torch_matmul(K_it, v_it)
      term1 <- I_mat - torch_matmul(K_it, Lambda1)
      P_updated <- torch_matmul(torch_matmul(term1, P_pred), term1$transpose(1, 2)) + 
        torch_matmul(torch_matmul(K_it, R), K_it$transpose(1, 2))
      
      log_det_F_it <- torch_logdet(F_it)
      exponent_term <- -0.5 * torch_matmul(v_it$transpose(1, 2), torch_matmul(F_it_inv, v_it))
      log_likelihood_it <- -0.5 * O1 * log(2 * pi) - 0.5 * log_det_F_it + exponent_term
      
      total_log_likelihood <- total_log_likelihood + log_likelihood_it
      
      eta_prev <- eta_updated
      P_prev <- P_updated
    }
  }
  return(total_log_likelihood)
}

# ----------------------------------------------------------------
# Part 2: データ生成 (変更なし)
# ----------------------------------------------------------------
cat("--- 1. Generating Simulation Data (User Specified Function Version) ---\n")

N <- 50; Nt <- 25; O1 <- 6; L1 <- 2; L2 <- 2

b0_true <- torch_tensor(c(0.1, -0.1))$reshape(c(L1, 1))
B1_true <- torch_tensor(matrix(c(0.7, 0.1, -0.1, 0.6), L1, L1), dtype = torch_float())
b2_true <- torch_tensor(matrix(c(0.5, 0.0, 0.2, 0.4), L1, L2), dtype = torch_float())
Lambda1_true <- torch_tensor(matrix(c(1, 0.8, 0.6, 0, 0, 0, 0, 0, 0, 1, 1.2, 0.8), O1, L1, byrow = FALSE), dtype = torch_float())
Q_true <- torch_eye(L1) * 0.5
R_true <- torch_eye(O1) * 1.0

q_dist <- distr_multivariate_normal(torch_zeros(L1), Q_true)
r_dist <- distr_multivariate_normal(torch_zeros(O1), R_true)

eta2_true <- torch_randn(N, L2)
Y1_generated <- torch_zeros(N, Nt, O1)

for (i in 1:N) {
  eta1_prev <- torch_zeros(L1, 1)
  eta2_i <- eta2_true[i, ]$reshape(c(L2, 1))
  for (t in 1:Nt) {
    eta1_mean_t <- b0_true + torch_matmul(B1_true, eta1_prev) + torch_matmul(b2_true, eta2_i)
    noise_q <- q_dist$sample()$reshape(c(L1, 1))
    eta1_t <- eta1_mean_t + noise_q
    y1_mean_t <- torch_matmul(Lambda1_true, eta1_t)
    noise_r <- r_dist$sample()$reshape(c(O1, 1))
    y1_t <- y1_mean_t + noise_r
    Y1_generated[i, t, ] <- y1_t$squeeze()
    eta1_prev <- eta1_t
  }
}
cat("Simulation data generated.\n\n")

## ----------------------------------------------------------------
## Part 3 & 4: グリッドサーチによるハイパーパラメータ探索
## ----------------------------------------------------------------

# --- 1. ハイパーパラメータのグリッドを定義 ---
hyper_grid <- expand.grid(
  learning_rate = c(0.01, 0.005, 0.001),
  l2_decay = c(1e-2, 1e-3, 1e-4),
  stringsAsFactors = FALSE
)

num_epochs <- 1000 
num_runs_per_setting <- 1

all_results <- list()
cat(sprintf("--- Starting Grid Search for %d Combinations ---\n", nrow(hyper_grid)))

# [修正点 1/2] ループの前に、最小損失と最良パラメータを保存する変数を初期化
best_loss <- Inf
best_b0 <- NULL
best_B1 <- NULL
best_b2 <- NULL
best_lambda1_free_params <- NULL

for (i in 1:nrow(hyper_grid)) {
  current_lr <- hyper_grid$learning_rate[i]
  current_decay <- hyper_grid$l2_decay[i]
  
  for (run in 1:num_runs_per_setting) {
    cat(sprintf("\n--- Combination %d/%d | Run %d/%d: LR=%.4f, Decay=%.4f ---\n", 
                i, nrow(hyper_grid), run, num_runs_per_setting, current_lr, current_decay))
    
    # --- パラメータ初期化 ---
    b0 <- torch_randn(L1, 1, requires_grad = TRUE)
    
    # -------------------- B1の初期化（ここを変更） --------------------
    
    # 新しい初期化方法：対角・非対角でサンプリング範囲を変える
    # 1. まずはRの標準的な行列を作成
    B1_init_matrix <- matrix(0, nrow = L1, ncol = L1)
    
    # 2. 対角成分を U(0.5, 1.0) からサンプリング
    diag_vals <- runif(L1, min = 0.5, max = 1.0)
    diag(B1_init_matrix) <- diag_vals
    
    # 3. 非対角成分を U(-0.3, 0.3) からサンプリング
    #   (L1*L1 - L1) 個の乱数を生成
    off_diag_vals <- runif(L1*L1 - L1, min = -0.3, max = 0.3)
    #   非対角要素に値を設定
    B1_init_matrix[row(B1_init_matrix) != col(B1_init_matrix)] <- off_diag_vals
    
    # 4. 作成した行列を、勾配計算が可能なtorchテンソルに変換
    B1 <- torch_tensor(B1_init_matrix, dtype = torch_float(), requires_grad = TRUE)
    
    # cat("B1 has been initialized with informed priors:\n")
    # print(B1_init_matrix)
    
    # -------------------- 変更ここまで --------------------    b2 <- torch_randn(L1, L2, requires_grad = TRUE)
    lambda1_free_params <- torch_randn(4, requires_grad = TRUE)
    
    Q <- Q_true$clone(); R <- R_true$clone()
    eta1_i0_0 <- torch_zeros(L1, 1); P_i0_0 <- torch_eye(L1) * 1e3
    
    optimizer <- optim_adamw(
      list(b0, B1, b2, lambda1_free_params), 
      lr = current_lr, 
      weight_decay = current_decay
    )
    
    for (epoch in 1:num_epochs) {
      Lambda1 <- torch_zeros(O1, L1)
      Lambda1[1, 1] <- 1
      Lambda1[4, 2] <- 1
      Lambda1[2, 1] <- lambda1_free_params[1]
      Lambda1[3, 1] <- lambda1_free_params[2]
      Lambda1[5, 2] <- lambda1_free_params[3]
      Lambda1[6, 2] <- lambda1_free_params[4]
      
      optimizer$zero_grad()
      
      log_likelihood <- kalman_filter_torch(
        Y1 = Y1_generated, precomputed_eta2 = eta2_true,
        b0 = b0, B1 = B1, b2 = b2, Lambda1 = Lambda1,
        Q = Q, R = R, eta1_i0_0 = eta1_i0_0, P_i0_0 = P_i0_0
      )
      
      loss <- -log_likelihood
      loss$backward()
      optimizer$step()
    }
    
    final_loss <- loss$item()
    cat(sprintf("Combination %d/%d Finished. Final Loss: %.4f\n", i, nrow(hyper_grid), final_loss))
    
    result_row <- data.frame(
      learning_rate = current_lr,
      l2_decay = current_decay,
      run = run,
      final_loss = final_loss
    )
    all_results[[length(all_results) + 1]] <- result_row
    
    # [修正点 2/2] 現在の損失がこれまでの最小損失より小さい場合、パラメータを保存する
    if (final_loss < best_loss) {
      cat(sprintf(">>> New best model found! Loss: %.4f\n", final_loss))
      best_loss <- final_loss
      best_b0 <- b0$clone()$detach()
      best_B1 <- B1$clone()$detach()
      best_b2 <- b2$clone()$detach()
      best_lambda1_free_params <- lambda1_free_params$clone()$detach()
    }
  }
}

results_df <- dplyr::bind_rows(all_results)
cat("\n\n--- Grid Search Results ---\n")
print(results_df %>% arrange(final_loss))

best_params_config <- results_df %>% top_n(-1, final_loss)
cat("\n--- Best Hyperparameters ---\n")
print(best_params_config)


# --- 最終結果の確認 ---
cat("\n\n--- Final Estimated Parameters (from best run) ---\n")

# 最良の実行から得られた自由パラメータを使ってLambda1を再構築
Lambda1_estimated <- torch_zeros(O1, L1)
Lambda1_estimated[1, 1] <- 1
Lambda1_estimated[4, 2] <- 1
Lambda1_estimated[2, 1] <- best_lambda1_free_params[1]
Lambda1_estimated[3, 1] <- best_lambda1_free_params[2]
Lambda1_estimated[5, 2] <- best_lambda1_free_params[3]
Lambda1_estimated[6, 2] <- best_lambda1_free_params[4]

cat("True Lambda1:\n")
print(Lambda1_true)
cat("\nEstimated Lambda1:\n")
print(Lambda1_estimated)

cat("\nTrue b0:\n")
print(b0_true)
cat("\nEstimated b0:\n")
print(best_b0)

cat("\nTrue B1:\n")
print(B1_true)
cat("\nEstimated B1:\n")
print(best_B1)

cat("\nTrue b2:\n")
print(b2_true)
cat("\nEstimated b2:\n")
print(best_b2)