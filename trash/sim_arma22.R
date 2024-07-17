set.seed(123) # 再現性のため

# ARMAモデル用データ生成
y_arma <- arima.sim(model = list(ar = c(0.5, -0.25), ma = c(0.5, -0.3)), n = 100)

# ARMAモデルのフィットと評価
arma_orders <- expand.grid(ar = 1:3, ma = 0:3) # ARMA(1,0)からARMA(3,3)まで
arma_results <- data.frame(ar_order = integer(), ma_order = integer(), loglik = numeric(), aic = numeric())

for(i in 1:nrow(arma_orders)) {
  fit <- arima(y_arma, order = c(arma_orders$ar[i], 0, arma_orders$ma[i]))
  arma_results <- rbind(arma_results, data.frame(ar_order = arma_orders$ar[i], ma_order = arma_orders$ma[i], loglik = fit$loglik, aic = fit$aic))
}

print(arma_results)

set.seed(123) # 再現性のため

# ARMAモデルの設定
arma_orders <- expand.grid(ar = 1:3, ma = 0:3) # ARMA(1,0)からARMA(3,3)まで

# 結果を格納するリスト
arma_results_list <- vector("list", nrow(arma_orders))

for(i in 1:nrow(arma_orders)) {
  loglik_list <- numeric(200)
  aic_list <- numeric(200)
  
  for(j in 1:200) {
    # tryCatchを使用してエラーをキャッチ
    result <- tryCatch({
      y_arma <- arima.sim(model = list(ar = c(0.5, -0.25), ma = c(0.5, -0.3)), n = 100)
      fit <- arima(y_arma, order = c(arma_orders$ar[i], 0, arma_orders$ma[i]))
      list(loglik = fit$loglik, aic = fit$aic)
    }, error = function(e) {
      # エラーが発生した場合はNAを返す
      list(loglik = NA, aic = NA)
    })
    
    # ログ尤度とAICを保存
    loglik_list[j] <- result$loglik
    aic_list[j] <- result$aic
  }
  
  # NAを除外して平均を計算
  valid_loglik <- loglik_list[!is.na(loglik_list)]
  valid_aic <- aic_list[!is.na(aic_list)]
  
  # 結果をリストに格納
  arma_results_list[[i]] <- list(
    ar_order = arma_orders$ar[i], 
    ma_order = arma_orders$ma[i], 
    mean_loglik = mean(valid_loglik), 
    mean_aic = mean(valid_aic)
  )
}

# 結果の表示
for(i in 1:length(arma_results_list)) {
  cat(paste("ARMA(", arma_results_list[[i]]$ar_order, ",", arma_results_list[[i]]$ma_order, ")\n", sep=""))
  cat("平均ログ尤度:", arma_results_list[[i]]$mean_loglik, "\n")
  cat("平均AIC:", arma_results_list[[i]]$mean_aic, "\n\n")
}

