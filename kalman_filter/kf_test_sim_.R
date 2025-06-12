# 必要なライブラリをインストールしてロード
# install.packages("torch")
library(torch)

# ----------------------------------------------------------------
# お客様ご提供のコード Part 1: torch を用いたカルマンフィルターの実装
# ----------------------------------------------------------------
kalman_filter_torch <- function(Y1, precomputed_eta2, b0, B1, b2, Lambda1, Q, R, eta1_i0_0, P_i0_0) {
  
  # --- 入力テンソルの次元を取得 ---
  dims <- Y1$shape
  N <- dims[1]  # 被験者数
  Nt <- dims[2] # 時間点数
  O1 <- dims[3] # 観測変数 y1 の次元
  L1 <- B1$shape[1] # 潜在変数 eta1 の次元
  
  # --- 結果を格納するためのテンソルを初期化 ---
  total_log_likelihood <- torch_tensor(0, dtype = torch_float())
  
  I_mat <- torch_eye(L1)
  
  for (i in 1:N) {
    # --- 被験者 i の初期値を設定 ---
    eta_prev <- eta1_i0_0$clone() # eta_1i,t-1|t-1
    P_prev <- P_i0_0$clone()   # P_i,t-1|t-1
    
    # --- 被験者 i のデータを抽出 ---
    y1_i <- Y1[i, , ] # Nt x O1
    eta2_i <- precomputed_eta2[i, ]$unsqueeze(1)$transpose(1, 2) # L2 x 1
    
    for (t in 1:Nt) {
      y1_it <- y1_i[t, ]$unsqueeze(1)$transpose(1, 2) # O1 x 1
      
      # --- 予測ステップ ---
      # Eq (2): 潜在状態の予測
      eta_pred <- b0 + torch_matmul(B1, eta_prev) + torch_matmul(b2, eta2_i)
      
      # Eq (3): 潜在状態の共分散の予測
      P_pred <- torch_matmul(torch_matmul(B1, P_prev), B1$transpose(1, 2)) + Q
      
      # --- 1期先予測誤差 ---
      # Eq (4): イノベーション (予測誤差)
      v_it <- y1_it - torch_matmul(Lambda1, eta_pred)
      
      # Eq (5): イノベーションの共分散
      F_it <- torch_matmul(torch_matmul(Lambda1, P_pred), Lambda1$transpose(1, 2)) + R
      
      # --- カルマンゲインの計算 ---
      F_it_inv <- torch_pinverse(F_it)
      K_it <- torch_matmul(torch_matmul(P_pred, Lambda1$transpose(1, 2)), F_it_inv)
      
      # --- 更新ステップ ---
      eta_updated <- eta_pred + torch_matmul(K_it, v_it)
      term1 <- I_mat - torch_matmul(K_it, Lambda1)
      P_updated <- torch_matmul(torch_matmul(term1, P_pred), term1$transpose(1, 2)) + 
        torch_matmul(torch_matmul(K_it, R), K_it$transpose(1, 2))
      
      # --- 対数尤度の計算 ---
      log_det_F_it <- torch_log(torch_det(F_it))
      exponent_term <- -0.5 * torch_matmul(torch_matmul(v_it$transpose(1, 2), F_it_inv), v_it)
      log_likelihood_it <- -0.5 * O1 * log(2 * pi) - 0.5 * log_det_F_it + exponent_term
      
      total_log_likelihood <- total_log_likelihood + log_likelihood_it
      
      eta_prev <- eta_updated
      P_prev <- P_updated
    }
  }
  
  return(total_log_likelihood)
}


# ----------------------------------------------------------------
# お客様ご提供のコード Part 2: 検証用のシミュレーションデータ生成
# ----------------------------------------------------------------
cat("--- 1. Generating Simulation Data ---\n")

# --- シミュレーションの次元を設定 ---
N <- 50
Nt <- 25
O1 <- 6
O2 <- 2
L1 <- 2
L2 <- 2

# --- 真のパラメータを torch テンソルとして定義 ---
b0_true <- torch_tensor(c(0.1, -0.1), dtype = torch_float())$unsqueeze(1)$transpose(1, 2)
B1_true <- torch_tensor(matrix(c(0.7, 0.1, -0.1, 0.6), L1, L1), dtype = torch_float())
b2_true <- torch_tensor(matrix(c(0.5, 0.0, 0.2, 0.4), L1, L2), dtype = torch_float())
Lambda1_true <- torch_tensor(matrix(c(1, 0.8, 0.6, 0, 0, 0, 0, 0, 0, 1, 1.2, 0.8), O1, L1, byrow = FALSE), dtype = torch_float())

Q_true <- torch_eye(L1) * 0.5
R_true <- torch_eye(O1) * 1.0

# --- データ生成 ---
eta2_true <- torch_randn(N, L2)
eta1_true <- torch_zeros(N, Nt, L1)
Y1_generated <- torch_zeros(N, Nt, O1)
eta1_prev <- torch_zeros(N, L1)

# お客様の環境で動作する関数名を尊重
q_dist <- distr_multivariate_normal(torch_zeros(L1), Q_true)
r_dist <- distr_multivariate_normal(torch_zeros(O1), R_true)

for (t in 1:Nt) {
  eta1_mean_t <- b0_true + torch_matmul(eta1_prev, B1_true$transpose(1, 2)) + torch_matmul(eta2_true, b2_true$transpose(1, 2))
  eta1_t <- eta1_mean_t + q_dist$sample(c(N))
  eta1_true[, t, ] <- eta1_t
  
  y1_mean_t <- torch_matmul(eta1_t, Lambda1_true$transpose(1, 2))
  Y1_generated[, t, ] <- y1_mean_t + r_dist$sample(c(N))
  
  eta1_prev <- eta1_t
}

cat("Simulation data generated.\n\n")

## ----------------------------------------------------------------
## Part 3 & 4: Adamオプティマイザーによるパラメータ推定と検証（複数回実行）
## ----------------------------------------------------------------
num_runs <- 3 # 何回独立した学習を行うか
learning_rate <- 0.01
num_epochs <- 200 # 十分な学習のためにエポック数を200に設定

# --- この外側のループで、毎回パラメータが初期化される ---
for (run in 1:num_runs) {
  
  cat(sprintf("\n--- Starting Run: %d / %d ---\n", run, num_runs))
  
  # --- Part 3: パラメータ初期化 ---
  cat("Initializing parameters for optimization...\n")
  
  b0 <- torch_randn(L1, 1, requires_grad = TRUE)
  B1 <- torch_randn(L1, L1, requires_grad = TRUE)
  b2 <- torch_randn(L1, L2, requires_grad = TRUE)
  Lambda1 <- torch_randn(O1, L1, requires_grad = TRUE)
  
  Q <- Q_true$clone()
  R <- R_true$clone()
  
  eta1_i0_0 <- torch_zeros(L1, 1)
  P_i0_0 <- torch_eye(L1) * 1e3
  
  # --- オプティマイザーを定義 ---
  optimizer <- optim_adam(list(b0, B1, b2, Lambda1), lr = learning_rate)
  
  cat(paste("Starting optimization with Adam. Epochs:", num_epochs, "LR:", learning_rate, "\n"))
  
  # --- トレーニングループ ---
  for (epoch in 1:num_epochs) {
    optimizer$zero_grad()
    
    log_likelihood <- kalman_filter_torch(
      Y1 = Y1_generated, 
      precomputed_eta2 = eta2_true,
      b0 = b0, B1 = B1, b2 = b2, Lambda1 = Lambda1,
      Q = Q, R = R,
      eta1_i0_0 = eta1_i0_0,
      P_i0_0 = P_i0_0
    )
    
    loss <- -log_likelihood
    loss$backward()
    optimizer$step()
    
    if (epoch %% 50 == 0 || epoch == 1) {
      cat(sprintf("Epoch: %d, Loss (Negative Log-Likelihood): %.4f\n", epoch, loss$item()))
    }
  }
  
  cat("Optimization finished for run", run, ".\n\n")
  
  # --- Part 4: 結果の検証 ---
  cat("--- Verifying Results for run", run, "---\n")
  
  cat("True b0:\n"); print(as.matrix(b0_true))
  cat("Estimated b0:\n"); print(as.matrix(b0$detach()))
  cat("---\n")
  
  cat("True B1:\n"); print(as.matrix(B1_true))
  cat("Estimated B1:\n"); print(as.matrix(B1$detach()))
  cat("---\n")
  
  cat("True b2:\n"); print(as.matrix(b2_true))
  cat("Estimated b2:\n"); print(as.matrix(b2$detach()))
  cat("---\n")
  
  cat("True Lambda1:\n"); print(as.matrix(Lambda1_true))
  cat("Estimated Lambda1:\n"); print(as.matrix(Lambda1$detach()))
  cat("---\n")
}