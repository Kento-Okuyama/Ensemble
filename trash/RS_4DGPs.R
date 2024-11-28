# 必要なパッケージのロード
library(rstan)
library(bayesplot)
library(loo)

# Stanコードの定義
stan_code <- "
data {
  int<lower=1> N;                  // 時系列の数
  int<lower=1> Nt;                 // 各時系列の長さ
  int<lower=1, upper=4> regime[N, Nt];  // 各時点でのレジームインデックス
  real y[N, Nt];                    // 観測データ行列 (各時系列が1行)
}
parameters {
  vector[4] mu;                    // 各レジームの平均
  real<lower=0, upper=1> lambda;   // 平滑化パラメータ（0に近いほど前の観測値を重視）
  real<lower=-1, upper=1> ar_pos;  // レジーム1のAR係数
  real<lower=-1, upper=1> ar_neg;  // レジーム2のAR係数
  real<lower=-1, upper=1> ma_coef; // レジーム3のMA係数
  vector<lower=0>[4] sigma;        // 各レジームの標準偏差
}
model {
  // Setting prior distributions
  mu ~ normal(0, 5);             // Prior for regime means
  lambda ~ beta(2, 2);           // Prior for smoothing parameter
  ar_pos ~ normal(0, 0.5);       // Prior for AR+ coefficient (centered near 0)
  ar_neg ~ normal(0, 0.5);       // Prior for AR- coefficient (centered near 0)
  ma_coef ~ normal(0, 0.5);      // Prior for MA coefficient
  sigma ~ cauchy(0, 2);          // Weakly informative prior for standard deviations

  for (n in 1:N) {
    for (t in 2:Nt) {
      int k = regime[n, t];  // 現在のレジーム
      
      // 遷移先レジームの平均に徐々に近づくための平滑化
      real target_mean = lambda * mu[k] + (1 - lambda) * y[n, t-1];

      if (k == 1) {
        y[n, t] ~ normal(target_mean + ar_pos * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 2) {
        y[n, t] ~ normal(target_mean + ar_neg * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 3) {
        y[n, t] ~ normal(target_mean + ma_coef * (y[n, t-1] - target_mean), sigma[k]);
      } else {
        y[n, t] ~ normal(target_mean, sigma[k]);
      }
    }
  }
}
generated quantities {
  real y_rep[N, Nt-1];           // 予測生成
  real log_lik[N, Nt-1];         // 各観測値の対数尤度
  
  for (n in 1:N) {
    for (t in 2:Nt) {
      int k = regime[n, t];
      real target_mean = lambda * mu[k] + (1 - lambda) * y[n, t-1];
      
      if (k == 1) {
        y_rep[n, t-1] = normal_rng(target_mean + ar_pos * (y[n, t-1] - target_mean), sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean + ar_pos * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 2) {
        y_rep[n, t-1] = normal_rng(target_mean + ar_neg * (y[n, t-1] - target_mean), sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean + ar_neg * (y[n, t-1] - target_mean), sigma[k]);
      } else if (k == 3) {
        y_rep[n, t-1] = normal_rng(target_mean + ma_coef * (y[n, t-1] - target_mean), sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean + ma_coef * (y[n, t-1] - target_mean), sigma[k]);
      } else {
        y_rep[n, t-1] = normal_rng(target_mean, sigma[k]);
        log_lik[n, t-1] = normal_lpdf(y[n, t] | target_mean, sigma[k]);
      }
    }
  }
}
"

# データの整形とStanモデルの実行
set.seed(123)
N <- 5    # 時系列の数
Nt <- 30   # 各時系列の長さ
lambda <- 0.5  # 平滑化パラメータ（Stanモデルのlambdaに対応）

# 各レジームごとのパラメータ設定
mu <- c(2, -2, 1, 0)         # 各レジームの平均値
ar_pos <- 0.7                # AR+の自己回帰係数
ar_neg <- -0.7               # AR-の自己回帰係数
ma_coef <- 0.5               # MA(1)の移動平均係数
sigma <- c(0.5, 0.7, 0.6, 0.4)  # 各レジームの標準偏差

# レジームの遷移行列の定義（例として一様に設定）
transition_matrix <- matrix(c(
  0.8, 0.1, 0.05, 0.05,  
  0.1, 0.8, 0.05, 0.05,  
  0.05, 0.05, 0.8, 0.1,  
  0.05, 0.05, 0.1, 0.8   
), nrow = 4, byrow = TRUE)

# 各時系列のデータとレジームインデックスを設定
y <- matrix(NA, nrow = N, ncol = Nt)
regime <- matrix(NA, nrow = N, ncol = Nt)

# 各時系列について遷移行列を使ってマルコフ過程に基づいたデータ生成
for (i in 1:N) {
  current_regime <- sample(1:4, 1)  
  y[i, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])  # 初期値を生成
  regime[i, 1] <- current_regime
  
  for (t in 2:Nt) {
    # レジームの遷移
    current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
    regime[i, t] <- current_regime
    
    # 平滑化された目標平均値を計算
    target_mean <- lambda * mu[current_regime] + (1 - lambda) * y[i, t - 1]
    
    # レジームに応じて観測値を生成
    if (current_regime == 1) {  # AR(1) with positive coefficient
      y[i, t] <- target_mean + ar_pos * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
      
    } else if (current_regime == 2) {  # AR(1) with negative coefficient
      y[i, t] <- target_mean + ar_neg * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
      
    } else if (current_regime == 3) {  # MA(1) process
      y[i, t] <- target_mean + ma_coef * (y[i, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
      
    } else {  # White noise
      y[i, t] <- rnorm(1, mean = target_mean, sd = sigma[current_regime])
    }
  }
}

# Stanに渡すデータリスト
stan_data <- list(
  N = N,
  Nt = Nt,
  regime = regime,
  y = y
)

# モデルのコンパイルとサンプリング
fit <- stan(model_code = stan_code, data = stan_data, iter = 2000, chains = 4)

# PSISを使った逐次除外交差検証
log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
loo_result <- loo(log_lik, moment_match = TRUE)
print(loo_result)

# 結果のプロット
bayesplot::mcmc_areas(as.array(fit), pars = c("ar_pos", "ar_neg", "ma_coef"))
bayesplot::mcmc_areas(as.array(fit), pars = c("sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]"))
bayesplot::mcmc_areas(as.array(fit), pars = c("mu[1]", "mu[2]", "mu[3]", "mu[4]", "lambda"))
