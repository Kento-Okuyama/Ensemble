# 必要なライブラリをロード
library(forecast)
library(rstan) # Bayesian Hierarchical StackingにはStanが必要

# オプション設定（計算の並列化）
rstan_options(auto_write=TRUE)
options(mc.cores=parallel::detectCores())

# AR(1) モデルのシミュレーションデータ生成
set.seed(123)
sim_data_ar1 <- arima.sim(n = 100, model = list(ar = 0.9))

# Stanモデルのコード
stan_code <- '
data {
  int<lower=0> N;  // データ点の数
  vector[N] y;     // 観測値
}
parameters {
  real alpha;  // AR(1)パラメータ
  real<lower=0> sigma;  // 観測ノイズの標準偏差
}
model {
  // 事前分布
  alpha ~ normal(0, 1);  // AR(1)パラメータの事前分布
  sigma ~ cauchy(0, 2.5);  // 観測ノイズの標準偏差の事前分布
  
  // 尤度
  for (n in 2:N)
    y[n] ~ normal(alpha * y[n-1], sigma);  // AR(1)モデル
}
'

# Stanモデルのコンパイル
stan_model <- stan_model(model_code = stan_code)

# データの準備
N <- length(sim_data_ar1)
stan_data <- list(N = N, y = sim_data_ar1)

# Stanモデルの実行
fit <- sampling(stan_model, data = stan_data, iter = 2000, chains = 4)

# 結果のサマリー
print(fit)

# 結果表示
stan_hist(fit)
