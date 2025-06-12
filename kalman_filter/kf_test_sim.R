# 必要なライブラリをインストールしてロード
# install.packages("torch")
library(torch)

## ----------------------------------------------------------------
## Part 1: torch を用いたカルマンフィルターの実装
## ----------------------------------------------------------------
kalman_filter_torch <- function(Y1, precomputed_eta2, B1, B2, B3, Lambda1, Q, R, eta1_i0_0, P_i0_0) {
  
  # --- 入力テンソルの次元を取得 ---
  dims <- Y1$shape
  N <- dims[1]  # 被験者数
  Nt <- dims[2] # 時間点数
  O1 <- dims[3] # 観測変数 y1 の次元
  L1 <- B2$shape[1] # 潜在変数 eta1 の次元
  
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
      eta_pred <- B1 + torch_matmul(B2, eta_prev) + torch_matmul(B3, eta2_i)
      
      # Eq (3): 潜在状態の共分散の予測
      P_pred <- torch_matmul(torch_matmul(B2, P_prev), B2$transpose(1, 2)) + Q
      
      # --- 1期先予測誤差 ---
      # Eq (4): イノベーション (予測誤差)
      v_it <- y1_it - torch_matmul(Lambda1, eta_pred)
      
      # Eq (5): イノベーションの共分散
      F_it <- torch_matmul(torch_matmul(Lambda1, P_pred), Lambda1$transpose(1, 2)) + R
      
      # --- カルマンゲインの計算 ---
      # torch_inverse は不安定な場合があるため、torch_linalg_pinv (疑似逆行列) を使用
      F_it_inv <- torch_pinverse(F_it)
      K_it <- torch_matmul(torch_matmul(P_pred, Lambda1$transpose(1, 2)), F_it_inv)
      
      # --- 更新ステップ ---
      # Eq (6): 潜在状態の更新
      eta_updated <- eta_pred + torch_matmul(K_it, v_it)
      
      # Eq (9): 潜在状態の共分散の更新 (Joseph form)
      # P_updated <- (I_mat - K_it %*% Lambda1) %*% P_pred %*% t(I_mat - K_it %*% Lambda1) + K_it %*% R %*% t(K_it)
      term1 <- I_mat - torch_matmul(K_it, Lambda1)
      P_updated <- torch_matmul(torch_matmul(term1, P_pred), term1$transpose(1, 2)) + 
        torch_matmul(torch_matmul(K_it, R), K_it$transpose(1, 2))
      
      # --- 対数尤度の計算 ---
      # Eq (8):
      # F_it の行列式の対数を計算
      log_det_F_it <- torch_log(torch_det(F_it))
      
      # 尤度計算
      exponent_term <- -0.5 * torch_matmul(torch_matmul(v_it$transpose(1, 2), F_it_inv), v_it)
      log_likelihood_it <- -0.5 * O1 * log(2 * pi) - 0.5 * log_det_F_it + exponent_term
      
      # 合計対数尤度に加算
      total_log_likelihood <- total_log_likelihood + log_likelihood_it
      
      # --- 次のイテレーションの準備 ---
      eta_prev <- eta_updated
      P_prev <- P_updated
    }
  }
  
  return(total_log_likelihood)
}


## ----------------------------------------------------------------
## Part 2: 検証用のシミュレーションデータ生成
## ----------------------------------------------------------------
cat("--- 1. Generating Simulation Data ---\n")

# --- シミュレーションの次元を設定 ---
N <- 50      # 被験者数
Nt <- 25     # 時間点数
O1 <- 3      # 観測変数 y1 の次元
O2 <- 2      # 観測変数 y2 の次元 (eta2 の推定に使用)
L1 <- 2      # 時間可変の潜在変数 eta1 の次元
L2 <- 2      # 時間不変の潜在変数 eta2 の次元

# --- 真のパラメータを torch テンソルとして定義 ---
B1_true <- torch_tensor(c(0.1, -0.1), dtype = torch_float())$unsqueeze(1)$transpose(1, 2)
B2_true <- torch_tensor(matrix(c(0.7, 0.1, -0.1, 0.6), L1, L1), dtype = torch_float())
B3_true <- torch_tensor(matrix(c(0.5, 0.0, 0.2, 0.4), L1, L2), dtype = torch_float())
Lambda1_true <- torch_tensor(matrix(c(1, 0.8, 0, 0, 1, 0.6), O1, L1, byrow = TRUE), dtype = torch_float())

# ノイズの共分散行列 (簡単のため、既知として扱う)
Q_true <- torch_eye(L1) * 0.5
R_true <- torch_eye(O1) * 1.0

# --- データ生成 ---
# 時間不変の潜在変数 eta2 を生成 (標準正規分布に従うとする)
eta2_true <- torch_randn(N, L2)

# 時間可変の潜在変数 eta1 と観測変数 Y1 を生成
eta1_true <- torch_zeros(N, Nt, L1)
Y1_generated <- torch_zeros(N, Nt, O1)
eta1_prev <- torch_zeros(N, L1) # t=0 の eta1 はゼロとする

# 正規分布ノイズを生成するための分布オブジェクト
q_dist <- distr_multivariate_normal(torch_zeros(L1), Q_true)
r_dist <- distr_multivariate_normal(torch_zeros(O1), R_true)

for (t in 1:Nt) {
  # eta1 を更新
  eta1_mean_t <- B1_true + torch_matmul(eta1_prev, B2_true$transpose(1, 2)) + torch_matmul(eta2_true, B3_true$transpose(1, 2))
  eta1_t <- eta1_mean_t + q_dist$sample(c(N))
  eta1_true[, t, ] <- eta1_t
  
  # Y1 を生成
  y1_mean_t <- torch_matmul(eta1_t, Lambda1_true$transpose(1, 2))
  Y1_generated[, t, ] <- y1_mean_t + r_dist$sample(c(N))
  
  eta1_prev <- eta1_t
}

cat("Simulation data generated.\n\n")

## ----------------------------------------------------------------
## Part 3: Adamオプティマイザーによるパラメータ推定
## ----------------------------------------------------------------
cat("--- 2. Initializing Parameters for Optimization ---\n")

# --- 推定するパラメータをランダムに初期化 ---
# requires_grad=TRUE にして勾配計算の対象とする
B1 <- torch_randn(L1, 1, requires_grad = TRUE)
B2 <- torch_randn(L1, L1, requires_grad = TRUE)
B3 <- torch_randn(L1, L2, requires_grad = TRUE)
Lambda1 <- torch_randn(O1, L1, requires_grad = TRUE)

# Q と R は簡単のため真の値に固定（未知の場合はこれらも推定対象にする）
Q <- Q_true$clone()
R <- R_true$clone()

# --- 初期状態の平均と共分散 ---
# (これも固定値として扱う)
eta1_i0_0 <- torch_zeros(L1, 1)
P_i0_0 <- torch_eye(L1) * 1e3 # 無情報事前分布の代わり

# --- オプティマイザーを設定 ---
learning_rate <- 0.01
optimizer <- optim_adam(list(B1, B2, B3, Lambda1), lr = learning_rate)
num_epochs <- 1

cat(paste("Starting optimization with Adam. Epochs:", num_epochs, "LR:", learning_rate, "\n"))

# --- トレーニングループ ---
for (epoch in 1:num_epochs) {
  # 勾配をゼロにリセット
  optimizer$zero_grad()
  
  # モデルを呼び出し、対数尤度を計算
  log_likelihood <- kalman_filter_torch(
    Y1 = Y1_generated, 
    precomputed_eta2 = eta2_true,
    B1 = B1, B2 = B2, B3 = B3, Lambda1 = Lambda1,
    Q = Q, R = R,
    eta1_i0_0 = eta1_i0_0,
    P_i0_0 = P_i0_0
  )
  
  # 損失を計算（対数尤度を最大化するため、負の値を最小化する）
  loss <- -log_likelihood
  
  # 勾配を計算
  loss$backward()
  
  # パラメータを更新
  optimizer$step()
  
  if (epoch %% 20 == 0) {
    cat(sprintf("Epoch: %d, Loss (Negative Log-Likelihood): %.4f\n", epoch, loss$item()))
  }
}

cat("Optimization finished.\n\n")


## ----------------------------------------------------------------
## Part 4: 結果の検証
## ----------------------------------------------------------------
cat("--- 3. Verifying Results ---\n")

# as.matrix() でRの行列に変換
cat("True B1:\n")
print(as.matrix(B1_true))
cat("Estimated B1:\n")
print(as.matrix(B1))
cat("---\n")

cat("True B2:\n")
print(as.matrix(B2_true))
cat("Estimated B2:\n")
print(as.matrix(B2))
cat("---\n")

cat("True B3:\n")
print(as.matrix(B3_true))
cat("Estimated B3:\n")
print(as.matrix(B3))
cat("---\n")

cat("True Lambda1:\n")
print(as.matrix(Lambda1_true))
cat("Estimated Lambda1:\n")
print(as.matrix(Lambda1))
cat("---\n")