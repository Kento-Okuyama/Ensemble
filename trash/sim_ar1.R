set.seed(123) # 再現性のため

# ARモデル用データ生成
y_ar <- arima.sim(model = list(ar = 0.9), n = 100)

# ARモデルのフィットと評価
ar_orders <- 0:5 # AR(0)からAR(5)まで
ar_results <- data.frame(order = integer(), loglik = numeric(), aic = numeric())

for(order in ar_orders) {
  fit <- arima(y_ar, order = c(order, 0, 0))
  ar_results <- rbind(ar_results, data.frame(order = order, loglik = fit$loglik, aic = fit$aic))
}

print(ar_results)


set.seed(123) # 再現性のため

# モデルオーダーごとに結果を格納するリスト
ar_results <- list()

# ARモデルのオーダー
ar_orders <- 0:5

# 各オーダーに対して200回のシミュレーションを実行
for(order in ar_orders) {
  loglik_list <- numeric(200)
  aic_list <- numeric(200)
  
  for(i in 1:200) {
    y_ar <- arima.sim(model = list(ar = 0.9), n = 100)
    fit <- arima(y_ar, order = c(order, 0, 0))
    loglik_list[i] <- fit$loglik
    aic_list[i] <- fit$aic
  }
  
  # 各オーダーの結果をリストに格納
  ar_results[[paste0("AR(", order, ")")]] <- list(
    mean_loglik = mean(loglik_list),
    mean_aic = mean(aic_list)
  )
}

# 結果の表示
for(order in names(ar_results)) {
  cat(order, "\n")
  cat("平均ログ尤度:", ar_results[[order]]$mean_loglik, "\n")
  cat("平均AIC:", ar_results[[order]]$mean_aic, "\n\n")
}

