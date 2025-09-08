import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal

# === 変更点 (1/4): シミュレーションモードを切り替える変数を追加 ===
# 'informative' または 'uninformative' を選択
SIMULATION_MODE = 'informative' 
print(f"--- Running Simulation in '{SIMULATION_MODE}' mode ---")
# === 変更ここまで ===

# Teeクラス (コンソールとファイルへの同時出力)
class Tee:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout; self.stderr = sys.stderr
        sys.stdout = self; sys.stderr = self
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback): self.close()
    def write(self, data): self.stdout.write(data); self.file.write(data)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self):
        sys.stdout = self.stdout; sys.stderr = self.stderr
        self.file.close()

# Teeクラスを使って全体の処理を囲む
file_path = r"model_comparison.txt"
with Tee(file_path):
    # ----------------------------------------------------------------
    # Part 0: モデルパラメータの定義
    # ----------------------------------------------------------------
    print("--- 0. Defining Parameters for the 3-to-1 Factor Regime-Switching Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N = 100; Nt = 25; O = 9; J = 2
    L1_state1 = 3; L1_state2 = 1
    
    B1_true_state1 = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.5]], dtype=torch.float32, device=device)
    Lambda1_state1_matrix = np.zeros((O, L1_state1), dtype=np.float32)
    Lambda1_state1_matrix[0:3, 0]=[1.0,0.8,0.7]; Lambda1_state1_matrix[3:6, 1]=[1.0,1.2,0.8]; Lambda1_state1_matrix[6:9, 2]=[1.0,0.9,0.7]
    Lambda1_true_state1 = torch.tensor(Lambda1_state1_matrix, device=device)
    B1_true_state2 = torch.tensor([[0.6]], dtype=torch.float32, device=device)
    Lambda1_state2_matrix = np.zeros((O, L1_state2), dtype=np.float32)
    Lambda1_state2_matrix[0:9, 0] = [1.0, 0.8, 0.7, 1.0, 1.1, 0.9, 0.9, 0.8, 0.7]
    Lambda1_true_state2 = torch.tensor(Lambda1_state2_matrix, device=device)
    Q_state1=torch.eye(L1_state1, device=device); Q_state2=torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept=-2.5; gamma_task=0.1; gamma_goal=0.1; gamma_bond=0.1

    # ▼▼▼【ここから追加】▼▼▼
    # Train-Test Splitのパラメータ
    Nt_train = 20 # 全25時点のうち、最初の20時点を訓練用とする
    Nt_test = Nt - Nt_train
    print(f"Time series split: {Nt_train} for training, {Nt_test} for testing.")
    # ▲▲▲【ここまで追加】▲▲▲

# ----------------------------------------------------------------
# Part 1: データ生成（ファイルが存在しない場合のみ実行）
# ----------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼【ここからPart 1全体を置き換え】▼▼▼▼▼▼▼▼▼
print("\n--- 1. Generating Simulation Data ---")

if os.path.exists(DATA_FILE):
    print(f"Loading pre-computed simulation data from '{DATA_FILE}'...")
    saved_data = torch.load(DATA_FILE, weights_only=False)
    Y_generated = saved_data['Y_generated'].to(device)
    actual_states = saved_data['actual_states']
    eta_true_history = saved_data['eta_true_history'].to(device)
    print("Data loading complete.")
else:
    print("No pre-computed data file found. Generating new simulation data...")
    # DGP parameters
    b0_true_state1 = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(1)
    B1_true_state1 = torch.diag(torch.tensor([0.7, 0.7, 0.7], device=device))
    lambda1_true_values_state1 = [1.2, 0.8, 1.1, 0.9, 1.3, 0.7]
    Lambda1_true_state1 = torch.zeros(O, L1_state1, device=device)
    Lambda1_true_state1[0,0]=1; Lambda1_true_state1[1,0]=lambda1_true_values_state1[0]; Lambda1_true_state1[2,0]=lambda1_true_values_state1[1]
    Lambda1_true_state1[3,1]=1; Lambda1_true_state1[4,1]=lambda1_true_values_state1[2]; Lambda1_true_state1[5,1]=lambda1_true_values_state1[3]
    Lambda1_true_state1[6,2]=1; Lambda1_true_state1[7,2]=lambda1_true_values_state1[4]; Lambda1_true_state1[8,2]=lambda1_true_values_state1[5]
    
    b0_true_state2 = torch.tensor([0.0], device=device).unsqueeze(1)
    B1_true_state2 = torch.tensor([[0.9]], device=device)
    lambda1_true_values_state2 = [1.0] * 8
    Lambda1_true_state2 = torch.zeros(O, L1_state2, device=device)
    Lambda1_true_state2[0,0]=1.0
    Lambda1_true_state2[1:9,0] = torch.tensor(lambda1_true_values_state2, device=device)

    Q_state1=torch.eye(L1_state1, device=device); Q_state2=torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept=-2.5; gamma_task=0.1; gamma_goal=0.1; gamma_bond=0.1

    Y_generated = torch.zeros(N, Nt, O, device=device)
    actual_states = np.zeros((N, Nt))
    eta_true_history = torch.full((N, Nt, L1_state1), float('nan'), device=device) 
    
    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)
    
    for i in trange(N, desc="Generating data for each person"):
        eta_history = torch.randn(L1_state1, 1, device=device)
        current_state = 1; has_switched = False
        for t in range(Nt):
            actual_states[i, t] = current_state
            if current_state == 1 and t > 0:
                z = gamma_intercept + (eta_history[0]*gamma_task + eta_history[1]*gamma_goal + eta_history[2]*gamma_bond)
                if random.random() < (1 / (1 + math.exp(-z))): current_state = 2
            
            if current_state == 1:
                eta_t = (B1_true_state1 @ eta_history) + q_dist_s1.sample().reshape(L1_state1, 1)
                y_mean_t = Lambda1_true_state1 @ eta_t
            else: 
                if not has_switched:
                    eta_history = torch.tensor([eta_history.mean()], device=device).reshape(L1_state2, 1)
                    has_switched = True
                eta_t = (B1_true_state2 @ eta_history) + q_dist_s2.sample().reshape(L1_state2, 1)
                y_mean_t = Lambda1_true_state2 @ eta_t
            
            Y_generated[i, t, :] = (y_mean_t + r_dist.sample().reshape(O, 1)).squeeze()
            
            if current_state == 1:
                eta_true_history[i, t, :] = eta_t.squeeze()
            else:
                eta_true_history[i, t, 0] = eta_t.squeeze()
            eta_history = eta_t
    
    print("Simulation data generated.")
    print(f"Saving simulation data to '{DATA_FILE}'...")
    data_to_save = {
        'Y_generated': Y_generated.cpu(),
        'actual_states': actual_states,
        'eta_true_history': eta_true_history.cpu()
    }
    torch.save(data_to_save, DATA_FILE)
    print("Data saving complete.")

    # ----------------------------------------------------------------
    # Part 2: 共通の関数定義
    # ----------------------------------------------------------------
    print("\n--- 2. Defining Common Functions ---")

    # 論文に基づき、尤度を時間で重み付けする関数
    def calculate_time_weights(T, gamma, device):
        """Calculates non-decreasing weights for time series likelihood."""
        t_values = torch.arange(T, device=device, dtype=torch.float32)
        T_max = T - 1 if T > 1 else 1
        pi = 1 + gamma - (1 - t_values / T_max)**2
        # 論文に従い正規化係数を適用
        normalization = T / pi.sum()
        return normalization * pi

    # グローバルパラメータとしてGAMMA_WEIGHTを定義
    GAMMA_WEIGHT = 0.1

    # ▼▼▼【ここを修正】▼▼▼
    time_weights_full = calculate_time_weights(Nt, GAMMA_WEIGHT, device)
    time_weights_train = calculate_time_weights(Nt_train, GAMMA_WEIGHT, device)
    print(f"Calculated time weights for training (normalized for {Nt_train} steps): {np.round(time_weights_train.cpu().numpy(), 3)}")
    # ▲▲▲【ここまで修正】▲▲▲
    
    def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0, time_weights=None): # <- 引数に追加
        N, Nt, O1 = Y.shape; L1 = B1.shape[0]
        
        eta_prev = eta1_i0_0.expand(N, -1, -1)
        P_prev = P_i0_0.expand(N, -1, -1)
        
        total_log_likelihood = 0.0
        
        for t in range(Nt):
            y_t = Y[:, t, :].unsqueeze(-1)
            
            eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
            P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
            
            v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
            F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
            
            F_inv_t = torch.linalg.pinv(F_t)
            K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), F_inv_t)
            
            eta_updated = eta_pred + torch.bmm(K_t, v_t)
            
            I_KL = torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
            P_updated = torch.bmm(torch.bmm(I_KL, P_pred), I_KL.transpose(-1, -2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(-1, -2))

            dist = torch.distributions.MultivariateNormal(loc=torch.zeros(O1, device=device), covariance_matrix=F_t)
            log_likelihood_t = dist.log_prob(v_t.squeeze(-1))
            
            if time_weights is not None:
                total_log_likelihood += (time_weights[t] * log_likelihood_t).sum()
            else:
                total_log_likelihood += log_likelihood_t.sum()
            
            eta_prev = eta_updated
            P_prev = P_updated
            
        return total_log_likelihood

    def get_kalman_predictions_and_latents(Y, b0, B1, Lambda1, Q, R):
        N, Nt, O1 = Y.shape
        L1 = B1.shape[0]
        
        y_pred_series = torch.zeros_like(Y)
        eta_series = torch.zeros(N, Nt, L1, device=device)
        # ▼▼▼【ここから追加】▼▼▼
        P_series = torch.zeros(N, Nt, L1, L1, device=device) # Pの履歴を保存する変数を追加
        # ▲▲▲【ここまで追加】▲▲▲
        
        eta_prev = torch.zeros(N, L1, 1, device=device)
        P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
        
        for t in range(Nt):
            y_t = Y[:, t, :].unsqueeze(-1)
            
            eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
            y_pred_series[:, t, :] = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred).squeeze(-1)
            
            v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
            P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
            F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
            
            K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
            
            eta_updated = eta_pred + torch.bmm(K_t, v_t)
            P_updated = torch.bmm((torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))), P_pred)
            
            eta_series[:, t, :] = eta_updated.squeeze(-1)
            # ▼▼▼【ここから追加】▼▼▼
            P_series[:, t, :, :] = P_updated # 更新されたPを保存
            # ▲▲▲【ここまで追加】▲▲▲
            
            eta_prev = eta_updated
            P_prev = P_updated
            
        # ▼▼▼【ここを修正】▼▼▼
        return y_pred_series, eta_series, P_series # P_seriesも返すように変更
        # ▲▲▲【ここまで修正】▲▲▲
        
    def calculate_rmse(y_true, y_pred): return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
    
    def calculate_sens_spec(actual_states, predicted_states_binary):
        ground_truth_binary = (actual_states == 2).astype(int)
        TP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 1)); FN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 1))
        TN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 0)); FP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 0))
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0; specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        return sensitivity, specificity

    def get_kalman_log_likelihoods_per_step(Y, b0, B1, Lambda1, Q, R):
        N, Nt, O1 = Y.shape
        L1 = B1.shape[0]
        
        eta_prev = torch.zeros(N, L1, 1, device=device)
        P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
        
        log_likelihood_series = torch.zeros(N, Nt, device=device)

        for t in range(Nt):
            y_t = Y[:, t, :].unsqueeze(-1)
            
            eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
            P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
            
            v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
            F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
            
            F_t += torch.eye(O1, device=device) * 1e-6 # 数値的安定性のためのジッター

            dist = torch.distributions.MultivariateNormal(loc=torch.zeros(O1, device=device), covariance_matrix=F_t)
            log_likelihood_series[:, t] = dist.log_prob(v_t.squeeze(-1))
            
            # フィルターの更新ステップ
            K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
            eta_updated = eta_pred + torch.bmm(K_t, v_t)
            P_updated = torch.bmm((torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))), P_pred)
            
            eta_prev = eta_updated
            P_prev = P_updated
            
        return log_likelihood_series

    def calculate_ssr_eta(eta_true, eta_est, actual_states, model_type):
        """
        Calculates the Sum of Squared Residuals for the latent variable eta,
        handling different model structures and dimensionality.
        """
        total_ssr = 0.0
        
        # Create boolean masks for states
        mask_state1 = (actual_states == 1)
        mask_state2 = (actual_states == 2)
        
        # Ensure tensors are on the same device for masking
        eta_true_device = eta_true.device
        mask_state1_torch = torch.from_numpy(mask_state1).to(eta_true_device)
        mask_state2_torch = torch.from_numpy(mask_state2).to(eta_true_device)

        if model_type == '3fac':
            # --- 3-Factor Baseline Model Evaluation ---
            true_s1 = eta_true[mask_state1_torch]
            est_s1 = eta_est[mask_state1_torch]
            valid_mask_s1 = ~torch.isnan(true_s1)
            ssr_s1 = torch.sum((true_s1[valid_mask_s1] - est_s1[valid_mask_s1])**2)
            
            true_s2 = eta_true[:, :, 0][mask_state2_torch]
            est_s2 = eta_est[mask_state2_torch].mean(dim=-1)
            valid_mask_s2 = ~torch.isnan(true_s2)
            ssr_s2 = torch.sum((true_s2[valid_mask_s2] - est_s2[valid_mask_s2])**2)
            
            total_ssr = (ssr_s1 + ssr_s2).item()

        elif model_type == '1fac':
            # --- 1-Factor Baseline Model Evaluation ---
            true_s1 = eta_true[mask_state1_torch].mean(dim=-1)
            est_s1 = eta_est[:, :, 0][mask_state1_torch]
            valid_mask_s1 = ~torch.isnan(true_s1)
            ssr_s1 = torch.sum((true_s1[valid_mask_s1] - est_s1[valid_mask_s1])**2)

            true_s2 = eta_true[:, :, 0][mask_state2_torch]
            est_s2 = eta_est[:, :, 0][mask_state2_torch]
            valid_mask_s2 = ~torch.isnan(true_s2)
            ssr_s2 = torch.sum((true_s2[valid_mask_s2] - est_s2[valid_mask_s2])**2)
            
            total_ssr = (ssr_s1 + ssr_s2).item()

        elif model_type == 'FRS':
            # --- FRS Model Evaluation ---
            eta_r1, eta_r2 = eta_est
            # State 1 part
            true_s1 = eta_true[mask_state1_torch]
            # ▼▼▼【ここを修正】▼▼▼
            est_s1 = eta_r1[mask_state1_torch][:, :3]
            valid_mask_s1 = ~torch.isnan(true_s1)
            ssr_s1 = torch.sum((true_s1[valid_mask_s1] - est_s1[valid_mask_s1])**2)

            # State 2 part
            true_s2 = eta_true[:, :, 0][mask_state2_torch]
            # ▼▼▼【ここを修正】▼▼▼
            est_s2 = eta_r2[mask_state2_torch][:, 3]
            valid_mask_s2 = ~torch.isnan(true_s2)
            ssr_s2 = torch.sum((true_s2[valid_mask_s2] - est_s2[valid_mask_s2])**2)
            
            total_ssr = (ssr_s1 + ssr_s2).item()

        elif model_type == 'BPS':
            # --- BPS Model Evaluation ---
            latents_3fac, latents_1fac, weights = eta_est
            
            latents_1fac_proj = latents_1fac.repeat(1, 1, 3)
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            eta_bps_est = w1 * latents_3fac + w2 * latents_1fac_proj
            
            # State 1 part
            true_s1 = eta_true[mask_state1_torch]
            est_s1 = eta_bps_est[mask_state1_torch]
            valid_mask_s1 = ~torch.isnan(true_s1)
            ssr_s1 = torch.sum((true_s1[valid_mask_s1] - est_s1[valid_mask_s1])**2)

            # State 2 part
            true_s2 = eta_true[:, :, 0][mask_state2_torch]
            est_s2 = eta_bps_est[mask_state2_torch].mean(dim=-1)
            valid_mask_s2 = ~torch.isnan(true_s2)
            ssr_s2 = torch.sum((true_s2[valid_mask_s2] - est_s2[valid_mask_s2])**2)
            
            total_ssr = (ssr_s1 + ssr_s2).item()

        return total_ssr

    def generate_forecasts(model_params, Y_full_data, Nt_train):
        """学習済みカルマンフィルタモデルを使い、1期先予測を生成する"""
        N, Nt, O1 = Y_full_data.shape
        L1 = model_params['B1'].shape[0]
        Nt_test = Nt - Nt_train
        
        # 予測結果を保存するテンソルを初期化
        y_forecasts = torch.zeros(N, Nt_test, O1, device=device)
        eta_forecasts = torch.zeros(N, Nt_test, L1, device=device)
        P_forecasts = torch.zeros(N, Nt_test, L1, L1, device=device)

        # 1. 訓練データ全体でフィルタリングを行い、最終状態を取得
        _, eta_series_train, P_series_train = get_kalman_predictions_and_latents(
            Y_full_data[:, :Nt_train, :], model_params['b0'], model_params['B1'], 
            model_params['Lambda1'], model_params['Q'], model_params['R']
        )
        # 訓練期間の最後の潜在変数と共分散を、予測の初期値とする
        eta_prev = eta_series_train[:, -1, :].unsqueeze(-1)
        P_prev = P_series_train[:, -1, :, :]

        # 2. テスト期間を1ステップずつ予測
        with torch.no_grad():
            for t_idx, t in enumerate(range(Nt_train, Nt)):
                # --- Prediction Step ---
                eta_pred = model_params['b0'] + torch.bmm(model_params['B1'].expand(N, -1, -1), eta_prev)
                P_pred = torch.bmm(torch.bmm(model_params['B1'].expand(N, -1, -1), P_prev), model_params['B1'].transpose(0, 1).expand(N, -1, -1)) + model_params['Q']

                # 観測値yを予測
                y_pred_t = torch.bmm(model_params['Lambda1'].expand(N, -1, -1), eta_pred).squeeze(-1)
                
                # 予測結果を保存
                y_forecasts[:, t_idx, :] = y_pred_t
                eta_forecasts[:, t_idx, :] = eta_pred.squeeze(-1)
                P_forecasts[:, t_idx, :, :] = P_pred

                # --- Update Step (次の予測のために、実際の観測値で状態を更新) ---
                y_t_actual = Y_full_data[:, t, :].unsqueeze(-1)
                v_t = y_t_actual - torch.bmm(model_params['Lambda1'].expand(N, -1, -1), eta_pred)
                F_t = torch.bmm(torch.bmm(model_params['Lambda1'].expand(N, -1, -1), P_pred), model_params['Lambda1'].transpose(0, 1).expand(N, -1, -1)) + model_params['R']
                K_t = torch.bmm(torch.bmm(P_pred, model_params['Lambda1'].transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
                
                eta_updated = eta_pred + torch.bmm(K_t, v_t)
                P_updated = torch.bmm((torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, model_params['Lambda1'].expand(N, -1, -1))), P_pred)

                # 更新された状態を次のステップに引き継ぐ
                eta_prev = eta_updated
                P_prev = P_updated

        return y_forecasts, eta_forecasts, P_forecasts

    def forecast_bps(bps_params, Y_full_data, Nt_train, best_params_3fac, best_params_1fac):
        """MCMCで推定されたBPSパラメータを使い、1期先予測を生成する"""
        N, Nt, O = Y_full_data.shape
        Nt_test = Nt - Nt_train
        Y_train_internal = Y_full_data[:, :Nt_train, :]

        # 1. 訓練期間のデータでBPSフィルターを回し、最終状態を取得
        with torch.no_grad():
            preds_3fac_train, _, _ = get_kalman_predictions_and_latents(Y_train_internal, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
            preds_1fac_train, _, _ = get_kalman_predictions_and_latents(Y_train_internal, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
            log_lik_3fac_train = get_kalman_log_likelihoods_per_step(Y_train_internal, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
            log_lik_1fac_train = get_kalman_log_likelihoods_per_step(Y_train_internal, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
            
            # MCMCから得られたパラメータの平均値を使って、訓練期間のフィルタリングを実行
            phi = torch.sigmoid(bps_params['phi_logit']); H = torch.diag(bps_params['h_diag'])
            Q = torch.diag(torch.exp(bps_params['log_q_diag'])); R = torch.diag(torch.exp(bps_params['log_r_diag']))
            intercept = bps_params['intercept']
            
            beta_tm1 = torch.zeros(N, J, device=device)
            P_tm1 = torch.eye(J, device=device).expand(N, -1, -1) * 1e-3
            for t in range(Nt_train):
                data_drift = torch.stack([log_lik_3fac_train[:, t], log_lik_1fac_train[:, t]], dim=-1)
                beta_pred = intercept + phi * beta_tm1
                P_pred = torch.diag_embed(phi**2) @ P_tm1 @ torch.diag_embed(phi**2).T + Q
                v = data_drift - (H @ beta_pred.unsqueeze(-1)).squeeze(-1)
                F = H @ P_pred @ H.T + R; K = P_pred @ H.T @ torch.linalg.pinv(F)
                beta_t = beta_pred + (K @ v.unsqueeze(-1)).squeeze(-1)
                I_KH = torch.eye(J, device=device) - K @ H
                P_t = I_KH @ P_pred @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
                beta_tm1 = beta_t; P_tm1 = P_t
                
            beta_prev = beta_tm1; P_prev = P_tm1

        # 2. テスト期間で1期先予測を繰り返す
        y_forecasts_bps = torch.zeros(N, Nt_test, O, device=device)
        weights_forecast_bps = torch.zeros(N, Nt_test, J, device=device)
        beta_forecast_bps = torch.zeros(N, Nt_test, J, device=device)
        P_forecast_bps = torch.zeros(N, Nt_test, J, J, device=device)

        with torch.no_grad():
            for t_idx, t in enumerate(range(Nt_train, Nt)):
                log_lik_3fac_t = get_kalman_log_likelihoods_per_step(Y_full_data[:, t:t+1, :], best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R']).squeeze()
                log_lik_1fac_t = get_kalman_log_likelihoods_per_step(Y_full_data[:, t:t+1, :], best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R']).squeeze()
                data_drift = torch.stack([log_lik_3fac_t, log_lik_1fac_t], dim=-1)

                beta_pred = intercept + phi * beta_prev
                P_pred = torch.diag_embed(phi**2) @ P_prev @ torch.diag_embed(phi**2).T + Q
                
                beta_forecast_bps[:, t_idx, :] = beta_pred
                P_forecast_bps[:, t_idx, :, :] = P_pred
                
                predicted_weights = torch.softmax(beta_pred, dim=-1)
                weights_forecast_bps[:, t_idx, :] = predicted_weights
                
                y_tm1 = Y_full_data[:, t-1:t, :]
                preds_3fac_t, _, _ = get_kalman_predictions_and_latents(y_tm1, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
                preds_1fac_t, _, _ = get_kalman_predictions_and_latents(y_tm1, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
                
                y_pred_mixed = predicted_weights[:, 0].unsqueeze(-1) * preds_3fac_t.squeeze(1) + \
                            predicted_weights[:, 1].unsqueeze(-1) * preds_1fac_t.squeeze(1)
                y_forecasts_bps[:, t_idx, :] = y_pred_mixed
                
                v = data_drift - (H @ beta_pred.unsqueeze(-1)).squeeze(-1)
                F = H @ P_pred @ H.T + R; K = P_pred @ H.T @ torch.linalg.pinv(F)
                beta_updated = beta_pred + (K @ v.unsqueeze(-1)).squeeze(-1)
                I_KH = torch.eye(J, device=device) - K @ H
                P_updated = I_KH @ P_pred @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
                
                beta_prev = beta_updated
                P_prev = P_updated
                
        return y_forecasts_bps, weights_forecast_bps, beta_forecast_bps, P_forecast_bps

    def forecast_rs(model_rs, Y_full_data, Nt_train):
        """学習済みRS-KFモデルを使い、1期先予測を生成する"""
        N, Nt, O = Y_full_data.shape
        Nt_test = Nt - Nt_train
        
        # 1. 訓練期間のデータでフィルターを回し、最終状態を取得
        with torch.no_grad():
            _ = model_rs(Y_full_data[:, :Nt_train, :])
            # 訓練期間最終時点の状態を取得
            prob_prev = model_rs.filtered_probs[:, -1, :]
            eta_r1_prev = model_rs.eta_filtered_r1_history[:, -1, :, None]
            eta_r2_prev = model_rs.eta_filtered_r2_history[:, -1, :, None]
            P_r1_prev = model_rs.P_filtered_r1_history[:, -1, :, :]
            P_r2_prev = model_rs.P_filtered_r2_history[:, -1, :, :]

        # 2. テスト期間で1期先予測を繰り返すための器を準備
        y_forecasts_rs = torch.zeros(N, Nt_test, O, device=device)
        probs_forecast_rs = torch.zeros(N, Nt_test, 2, device=device)
        eta_r1_forecasts_rs = torch.zeros(N, Nt_test, model_rs.L, device=device)
        eta_r2_forecasts_rs = torch.zeros(N, Nt_test, model_rs.L, device=device)
        P_r1_forecasts_rs = torch.zeros(N, Nt_test, model_rs.L, model_rs.L, device=device)
        P_r2_forecasts_rs = torch.zeros(N, Nt_test, model_rs.L, model_rs.L, device=device)

        with torch.no_grad():
            # モデルから必要な行列を取得
            (B11, B21, B12, B22), (L1_lambda, L2_lambda), (Q, R) = model_rs._build_matrices()
            
            # テスト期間を1時点ずつループ
            for t_idx, t in enumerate(range(Nt_train, Nt)):
                y_t = Y_full_data[:, t, :].unsqueeze(-1)
                
                # --- 1ステップ分のフィルター処理 (model_rs.forwardから抜粋) ---
                
                # スイッチ確率の計算
                eta_state1_components = eta_r1_prev[:, :model_rs.L-1, :].squeeze(-1)
                logit_p11 = model_rs.gamma1 + (eta_state1_components * model_rs.gamma2_learnable).sum(-1)
                p11 = torch.sigmoid(logit_p11)
                p22 = torch.full((N,), 0.9999, device=device) 
                transition_probs = torch.stack([torch.stack([p11, 1-p11], dim=1), torch.stack([1-p22, p22], dim=1)], dim=1)
                
                # --- Prediction Step ---
                eta_pred_11 = B11 @ eta_r1_prev; P_pred_11 = B11 @ P_r1_prev @ B11.mT + Q
                eta_pred_12 = B21 @ eta_r2_prev; P_pred_12 = B21 @ P_r2_prev @ B21.mT + Q
                eta_pred_21 = B12 @ eta_r1_prev; P_pred_21 = B12 @ P_r1_prev @ B12.mT + Q
                eta_pred_22 = B22 @ eta_r2_prev; P_pred_22 = B22 @ P_r2_prev @ B22.mT + Q

                # 観測値の1期先予測を計算
                y_pred_for_t = prob_prev[:,0].view(N,1,1)* (L1_lambda @ eta_pred_11) + \
                            prob_prev[:,1].view(N,1,1)* (L2_lambda @ eta_pred_22)

                y_forecasts_rs[:, t_idx, :] = y_pred_for_t.squeeze(-1)

                # --- Update Step ---
                v_11 = y_t - L1_lambda @ eta_pred_11; F_11 = L1_lambda @ P_pred_11 @ L1_lambda.mT + R
                v_12 = y_t - L1_lambda @ eta_pred_12; F_12 = L1_lambda @ P_pred_12 @ L1_lambda.mT + R
                v_21 = y_t - L2_lambda @ eta_pred_21; F_21 = L2_lambda @ P_pred_21 @ L2_lambda.mT + R
                v_22 = y_t - L2_lambda @ eta_pred_22; F_22 = L2_lambda @ P_pred_22 @ L2_lambda.mT + R
                
                f_jitter = torch.eye(O, device=device) * 1e-5
                F_11 += f_jitter; F_12 += f_jitter; F_21 += f_jitter; F_22 += f_jitter

                log_lik_11 = MultivariateNormal(loc=torch.zeros_like(v_11.squeeze(-1)), covariance_matrix=F_11).log_prob(v_11.squeeze(-1))
                log_lik_12 = MultivariateNormal(loc=torch.zeros_like(v_12.squeeze(-1)), covariance_matrix=F_12).log_prob(v_12.squeeze(-1))
                log_lik_21 = MultivariateNormal(loc=torch.zeros_like(v_21.squeeze(-1)), covariance_matrix=F_21).log_prob(v_21.squeeze(-1))
                log_lik_22 = MultivariateNormal(loc=torch.zeros_like(v_22.squeeze(-1)), covariance_matrix=F_22).log_prob(v_22.squeeze(-1))
                
                log_prob_prev = torch.log(prob_prev + 1e-9)

                log_prob_t_11 = log_prob_prev[:, 0] + torch.log(transition_probs[:, 0, 0] + 1e-9) + log_lik_11
                log_prob_t_12 = log_prob_prev[:, 1] + torch.log(transition_probs[:, 1, 0] + 1e-9) + log_lik_12
                log_prob_t_21 = log_prob_prev[:, 0] + torch.log(transition_probs[:, 0, 1] + 1e-9) + log_lik_21
                log_prob_t_22 = log_prob_prev[:, 1] + torch.log(transition_probs[:, 1, 1] + 1e-9) + log_lik_22

                log_prob_t = torch.stack([log_prob_t_11, log_prob_t_12, log_prob_t_21, log_prob_t_22], dim=1)
                
                log_likelihood_t = torch.logsumexp(log_prob_t, dim=1)
                
                prob_t = torch.exp(log_prob_t - log_likelihood_t.unsqueeze(1))
                
                prob_t_r1 = prob_t[:, 0] + prob_t[:, 1]
                prob_t_r2 = prob_t[:, 2] + prob_t[:, 3]

                W_21 = prob_t[:, 2] / (prob_t_r2 + 1e-9)
                W_22 = prob_t[:, 3] / (prob_t_r2 + 1e-9)
                
                # --- Collapsing Step ---
                K_11 = P_pred_11 @ L1_lambda.mT @ torch.linalg.pinv(F_11); eta_upd_11 = eta_pred_11 + K_11 @ v_11
                K_21 = P_pred_21 @ L2_lambda.mT @ torch.linalg.pinv(F_21); eta_upd_21 = eta_pred_21 + K_21 @ v_21
                K_22 = P_pred_22 @ L2_lambda.mT @ torch.linalg.pinv(F_22); eta_upd_22 = eta_pred_22 + K_22 @ v_22

                I_L = torch.eye(model_rs.L, device=device)
                I_KL_11 = I_L - K_11 @ L1_lambda; P_upd_11 = I_KL_11 @ P_pred_11 @ I_KL_11.mT + K_11 @ R @ K_11.mT
                I_KL_21 = I_L - K_21 @ L2_lambda; P_upd_21 = I_KL_21 @ P_pred_21 @ I_KL_21.mT + K_21 @ R @ K_21.mT
                I_KL_22 = I_L - K_22 @ L2_lambda; P_upd_22 = I_KL_22 @ P_pred_22 @ I_KL_22.mT + K_22 @ R @ K_22.mT

                eta_marg_r1_t = eta_upd_11
                P_marg_r1_t = P_upd_11
                eta_marg_r2_t = W_21.view(N, 1, 1) * eta_upd_21 + W_22.view(N, 1, 1) * eta_upd_22
                P_marg_r2_t = W_21.view(N,1,1) * (P_upd_21 + (eta_marg_r2_t-eta_upd_21) @ (eta_marg_r2_t-eta_upd_21).transpose(-1,-2)) + \
                            W_22.view(N,1,1) * (P_upd_22 + (eta_marg_r2_t-eta_upd_22) @ (eta_marg_r2_t-eta_upd_22).transpose(-1,-2))
                
                # 予測された潜在変数と共分散を保存
                eta_r1_forecasts_rs[:, t_idx, :] = eta_marg_r1_t.squeeze(-1)
                eta_r2_forecasts_rs[:, t_idx, :] = eta_marg_r2_t.squeeze(-1)
                P_r1_forecasts_rs[:, t_idx, :, :] = P_marg_r1_t
                P_r2_forecasts_rs[:, t_idx, :, :] = P_marg_r2_t
                
                # 次のループのために状態を更新
                eta_r1_prev, P_r1_prev = eta_marg_r1_t, P_marg_r1_t
                eta_r2_prev, P_r2_prev = eta_marg_r2_t, P_marg_r2_t
                prob_prev = torch.stack([prob_t_r1, prob_t_r2], dim=1)
                
                # 予測された確率を保存
                probs_forecast_rs[:, t_idx, :] = prob_prev
        
        return y_forecasts_rs, probs_forecast_rs, eta_r1_forecasts_rs, eta_r2_forecasts_rs, P_r1_forecasts_rs, P_r2_forecasts_rs

    # ▼▼▼【ここから追加】▼▼▼
    def get_bps_filtered_history(bps_params, y_obs, y_pred_expert1, y_pred_expert2, log_lik_3fac, log_lik_1fac):
        """MCMCで推定されたBPSパラメータを使い、全期間のフィルタリング履歴を取得する"""
        N, Nt, _ = y_obs.shape
        
        # MCMCから得られたパラメータの平均値を使用
        phi = torch.sigmoid(bps_params['phi_logit']); H = torch.diag(bps_params['h_diag'])
        Q = torch.diag(torch.exp(bps_params['log_q_diag'])); R = torch.diag(torch.exp(bps_params['log_r_diag']))
        intercept = bps_params['intercept']

        beta_history = torch.zeros(N, Nt, J, device=device)
        P_history = torch.zeros(N, Nt, J, J, device=device)
        
        beta_tm1 = torch.zeros(N, J, device=device)
        P_tm1 = torch.eye(J, device=device).expand(N, -1, -1) * 1e-3

        with torch.no_grad():
            for t in range(Nt):
                data_drift = torch.stack([log_lik_3fac[:, t], log_lik_1fac[:, t]], dim=-1)
                beta_pred = intercept + phi * beta_tm1
                P_pred = torch.diag_embed(phi**2) @ P_tm1 @ torch.diag_embed(phi**2).T + Q
                v = data_drift - (H @ beta_pred.unsqueeze(-1)).squeeze(-1)
                F = H @ P_pred @ H.T + R; K = P_pred @ H.T @ torch.linalg.pinv(F)
                beta_t = beta_pred + (K @ v.unsqueeze(-1)).squeeze(-1)
                I_KH = torch.eye(J, device=device) - K @ H
                P_t = I_KH @ P_pred @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
                
                beta_history[:, t, :] = beta_t
                P_history[:, t, :, :] = P_t
                beta_tm1 = beta_t
                P_tm1 = P_t
                
        return beta_history, P_history
    # ▲▲▲【ここまで追加】▲▲▲

    # ----------------------------------------------------------------
    # Part 3: ベースラインモデルのパラメータ推定と評価
    # ----------------------------------------------------------------
    print("\n--- 3. Estimation and Evaluation of Baseline Models ---")
    num_runs = 1
    learning_rate = 0.001
    num_epochs = 10000  # 最大エポック数を十分に大きく設定
    l2_decay = 0.01

    # === ▼▼▼ Early Stoppingのパラメータを修正 ▼▼▼ ===
    patience = 100
    # === ▲▲▲ ここまで ▲▲▲ ===

    results_file_3fac = 'baseline_3fac_results.pt'
    results_file_1fac = 'baseline_1fac_results.pt'

    # --- 3a. 3-Factor Model ---
    print("\n--- 3a. 3-Factor Model ---")

    if os.path.exists(results_file_3fac):
        print(f"Loading pre-computed 3-factor model results from '{results_file_3fac}'...")
        saved_data = torch.load(results_file_3fac, weights_only=False)
        best_loss_3fac = saved_data['best_loss']; best_params_3fac = saved_data['best_params']
        rmse_3fac = saved_data['rmse']; sens_3fac = saved_data['sens']; spec_3fac = saved_data['spec']
        duration_3fac = saved_data.get('duration', 0.0); failed_runs_3fac = 0
        ssr_eta_3fac = saved_data.get('ssr_eta', float('nan'))
        history_3fac = {}
        print("Loading complete.")
    else:
        print(f"No pre-computed file found. Running 3-factor model estimation...")
        start_time_3fac = time.time()
        best_loss_3fac = float('inf'); best_params_3fac = {}; failed_runs_3fac = 0
        history_3fac = {}
        
        for run in range(num_runs):
            print(f"Starting Run {run+1}/{num_runs}...")
            b0 = torch.randn(L1_state1, 1, requires_grad=True, device=device)
            b1_free_params = torch.randn(L1_state1, requires_grad=True, device=device)
            lambda1_free_params = torch.randn(6, requires_grad=True, device=device)
            log_q_diag_3fac = torch.zeros(L1_state1, requires_grad=True, device=device)
            log_r_diag_3fac = torch.zeros(O, requires_grad=True, device=device)
            
            params_to_learn_3fac = [b0, b1_free_params, lambda1_free_params, log_q_diag_3fac, log_r_diag_3fac]
            optimizer = torch.optim.AdamW(params_to_learn_3fac, lr=learning_rate, weight_decay=l2_decay)
            
            patience_counter = 0; best_loss_in_run = float('inf'); best_params_in_run = {}
            param_names_to_track_3fac = ['b0', 'b1_free_params', 'lambda1_free_params', 'log_q_diag_3fac', 'log_r_diag_3fac']
            history_3fac_in_run = {name: [] for name in param_names_to_track_3fac}

            # ▼▼▼【ここを修正】▼▼▼
            absolute_threshold = 0.1 # 閾値の定義をループの外に

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                Q_est_3fac = torch.diag(torch.exp(log_q_diag_3fac)); R_est_3fac = torch.diag(torch.exp(log_r_diag_3fac))
                B1 = torch.diag(b1_free_params); Lambda1 = torch.zeros(O, L1_state1, device=device)
                Lambda1[0,0]=1; Lambda1[1,0]=lambda1_free_params[0]; Lambda1[2,0]=lambda1_free_params[1]
                Lambda1[3,1]=1; Lambda1[4,1]=lambda1_free_params[2]; Lambda1[5,1]=lambda1_free_params[3]
                Lambda1[6,2]=1; Lambda1[7,2]=lambda1_free_params[4]; Lambda1[8,2]=lambda1_free_params[5]
                loss = -kalman_filter_torch_loss(Y_train, b0, B1, Lambda1, Q_est_3fac, R_est_3fac, torch.zeros(L1_state1, 1, device=device), torch.eye(L1_state1, device=device) * 1e3, time_weights=time_weights_train)                
                if torch.isnan(loss): failed_runs_3fac += 1; print(f"  Run {run+1} failed due to NaN loss."); break
                loss.backward()
                torch.nn.utils.clip_grad_norm_([b0, b1_free_params, lambda1_free_params], 1.0); optimizer.step()
                
                if (epoch + 1) % 100 == 0:
                    print(f"  [Run {run + 1}, Epoch {epoch + 1:04d}] loss: {loss.item():.4f} (Best: {best_loss_in_run:.4f})")
                    with torch.no_grad():
                        for name, param in zip(param_names_to_track_3fac, params_to_learn_3fac):
                            history_3fac_in_run[name].append(param.clone().cpu().numpy())

                if best_loss_in_run - loss.item() > absolute_threshold:                    
                    best_loss_in_run = loss.item()
                    patience_counter = 0
                    best_params_in_run = {'b0': b0.detach(), 'B1': B1.detach(), 'Lambda1': Lambda1.detach(), 'Q': Q_est_3fac.detach(), 'R': R_est_3fac.detach()}
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"    -> Early stopping triggered at epoch {epoch + 1}.")
                    break
            # ▲▲▲【ここまで修正】▲▲▲

            if not torch.isnan(loss) and best_loss_in_run < best_loss_3fac:
                best_loss_3fac = best_loss_in_run
                best_params_3fac = best_params_in_run
                history_3fac = history_3fac_in_run
                print(f"-> New best loss for 3-Factor Model: {best_loss_3fac:.4f}")
                    
        duration_3fac = time.time() - start_time_3fac if 'start_time_3fac' in locals() else duration_3fac
        
        print("\n--- 3-Factor Model Parameter Update History (Best Run) ---")
        if history_3fac:
            for name, history in history_3fac.items():
                print(f"\nParameter: {name}")
                for i, val in enumerate(history):
                    epoch_num = (i + 1) * 100
                    display_val = np.exp(val) if 'log_' in name else val
                    print(f"  Epoch {epoch_num:>4d}: {np.round(display_val.flatten(), 4)}")

        print("\n--- Evaluating 3-Factor Model on Test Set ---")
        preds_3fac_forecast, latents_3fac_forecast, _ = generate_forecasts(best_params_3fac, Y_generated, Nt_train)

        # RMSEはテストデータで評価
        rmse_3fac = calculate_rmse(Y_test, preds_3fac_forecast)
        print(f"Test RMSE: {rmse_3fac:.4f}")

        # ▼▼▼【ここから修正】▼▼▼
        # SSR(eta)をテストデータ（予測値）で評価する
        # これは「予測誤差」であり、訓練時の「推定誤差」とは性質が異なることに注意
        ssr_eta_3fac = calculate_ssr_eta(eta_true_history_test.cpu(), latents_3fac_forecast.cpu(), actual_states_test, model_type='3fac')
        print(f"Test Forecast SSR(eta): {ssr_eta_3fac:.4f}")

        # Sens/Specは引き続き一旦、仮の値を設定
        sens_3fac, spec_3fac = (0.0, 1.0) 
        # ▲▲▲【ここまで修正】▲▲▲

        if not os.path.exists(results_file_3fac):
            print(f"Saving 3-factor model results to '{results_file_3fac}'...")
            results_to_save = {'best_loss': best_loss_3fac, 'best_params': best_params_3fac, 'rmse': rmse_3fac, 'sens': sens_3fac, 'spec': spec_3fac, 'duration': duration_3fac, 'ssr_eta': ssr_eta_3fac}
            torch.save(results_to_save, results_file_3fac)
            print("Saving complete.")

    # --- 3b. 1-Factor Model ---
    print("\n--- 3b. 1-Factor Model ---")
    if os.path.exists(results_file_1fac):
        print(f"Loading pre-computed 1-factor model results from '{results_file_1fac}'...")
        saved_data = torch.load(results_file_1fac, weights_only=False)
        best_loss_1fac = saved_data['best_loss']; best_params_1fac = saved_data['best_params']
        rmse_1fac = saved_data['rmse']; sens_1fac = saved_data['sens']; spec_1fac = saved_data['spec']
        duration_1fac = saved_data.get('duration', 0.0); failed_runs_1fac = 0
        ssr_eta_1fac = saved_data.get('ssr_eta', float('nan'))
        history_1fac = {}
        print("Loading complete.")
    else:
        print(f"No pre-computed file found. Running 1-factor model estimation...")
        start_time_1fac = time.time()
        best_loss_1fac = float('inf'); best_params_1fac = {}; failed_runs_1fac = 0
        history_1fac = {}
        
        for run in range(num_runs):
            print(f"Starting Run {run+1}/{num_runs}...")
            b0 = torch.randn(L1_state2, 1, requires_grad=True, device=device); B1 = torch.randn(L1_state2, L1_state2, requires_grad=True, device=device)
            lambda1_free_params = torch.randn(8, requires_grad=True, device=device)
            log_q_diag_1fac = torch.zeros(L1_state2, requires_grad=True, device=device)
            log_r_diag_1fac = torch.zeros(O, requires_grad=True, device=device)
            params_to_learn_1fac = [b0, B1, lambda1_free_params, log_q_diag_1fac, log_r_diag_1fac]
            optimizer = torch.optim.AdamW(params_to_learn_1fac, lr=learning_rate, weight_decay=l2_decay)
            patience_counter = 0; best_loss_in_run = float('inf'); best_params_in_run = {}
            param_names_to_track_1fac = ['b0', 'B1', 'lambda1_free_params', 'log_q_diag_1fac', 'log_r_diag_1fac']
            history_1fac_in_run = {name: [] for name in param_names_to_track_1fac}
            
            # ▼▼▼【ここから修正】▼▼▼
            absolute_threshold = 0.1

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                Q_est_1fac = torch.diag(torch.exp(log_q_diag_1fac)); R_est_1fac = torch.diag(torch.exp(log_r_diag_1fac))
                Lambda1 = torch.zeros(O, L1_state2, device=device)
                Lambda1[0,0]=1.0; Lambda1[1:9,0] = lambda1_free_params[0:8]
                loss = -kalman_filter_torch_loss(Y_train, b0, B1, Lambda1, Q_est_1fac, R_est_1fac, torch.zeros(L1_state2, 1, device=device), torch.eye(L1_state2, device=device) * 1e3, time_weights=time_weights_train)
                if torch.isnan(loss): failed_runs_1fac += 1; print(f"  Run {run+1} failed due to NaN loss."); break
                loss.backward()
                torch.nn.utils.clip_grad_norm_([b0, B1, lambda1_free_params], 1.0); optimizer.step()
                
                if (epoch + 1) % 100 == 0:
                    print(f"  [Run {run + 1}, Epoch {epoch + 1:04d}] loss: {loss.item():.4f} (Best: {best_loss_in_run:.4f})")
                    with torch.no_grad():
                        for name, param in zip(param_names_to_track_1fac, params_to_learn_1fac):
                            history_1fac_in_run[name].append(param.clone().cpu().numpy())

                if best_loss_in_run - loss.item() > absolute_threshold:
                    best_loss_in_run = loss.item()
                    patience_counter = 0
                    best_params_in_run = {'b0': b0.detach(), 'B1': B1.detach(), 'Lambda1': Lambda1.detach(), 'Q': Q_est_1fac.detach(), 'R': R_est_1fac.detach()}
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"    -> Early stopping triggered at epoch {epoch + 1}.")
                    break
            # ▲▲▲【ここまで修正】▲▲▲

            if not torch.isnan(loss) and best_loss_in_run < best_loss_1fac:
                best_loss_1fac = best_loss_in_run
                best_params_1fac = best_params_in_run
                history_1fac = history_1fac_in_run
                print(f"-> New best loss for 1-Factor Model: {best_loss_1fac:.4f}")

        duration_1fac = time.time() - start_time_1fac if 'start_time_1fac' in locals() else duration_1fac

        print("\n--- 1-Factor Model Parameter Update History (Best Run) ---")
        if history_1fac:
            for name, history in history_1fac.items():
                print(f"\nParameter: {name}")
                for i, val in enumerate(history):
                    epoch_num = (i + 1) * 100
                    display_val = np.exp(val) if 'log_' in name else val
                    print(f"  Epoch {epoch_num:>4d}: {np.round(display_val.flatten(), 4)}")

        print("\n--- Evaluating 1-Factor Model on Test Set ---")
        preds_1fac_forecast, latents_1fac_forecast, _ = generate_forecasts(best_params_1fac, Y_generated, Nt_train)
        rmse_1fac = calculate_rmse(Y_test, preds_1fac_forecast)
        ssr_eta_1fac = calculate_ssr_eta(eta_true_history_test.cpu(), latents_1fac_forecast.cpu(), actual_states_test, model_type='1fac')
        sens_1fac, spec_1fac = (1.0, 0.0) # 仮の値 (常にState 2と予測する場合)
        print(f"Test RMSE: {rmse_1fac:.4f}")
        print(f"Test Forecast SSR(eta): {ssr_eta_1fac:.4f}")

        if not os.path.exists(results_file_1fac):
            print(f"Saving 1-factor model results to '{results_file_1fac}'...")
            results_to_save = {'best_loss': best_loss_1fac, 'best_params': best_params_1fac, 'rmse': rmse_1fac, 'sens': sens_1fac, 'spec': spec_1fac, 'duration': duration_1fac, 'ssr_eta': ssr_eta_1fac}
            torch.save(results_to_save, results_file_1fac)
            print("Saving complete.")

    # ==============================================================================
    # ▼▼▼【Part 4 と Part 5 を以下に置き換え】▼▼▼
    # ----------------------------------------------------------------
    # Part 4: BPS固定混合モデルの定義と学習
    # ----------------------------------------------------------------
    from pyro.infer import Predictive, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    import torch.nn.functional as F

    def bps_fixed_mixture_model(y_obs, eta_3fac_series, log_lik_3fac, log_lik_1fac, sim_mode):
        """
        【BPS固定混合モデル版】
        遷移パラメータ(gamma)のみを学習し、観測尤度は事前学習済みの
        ベースラインモデルから計算する。
        """
        N, Nt, O = y_obs.shape
        
        # --- 1. 学習対象のパラメータ(gamma)の事前分布を定義 ---
        if sim_mode == 'informative':
            true_gamma0 = torch.tensor(gamma_intercept, device=device)
            true_gamma_eta = torch.tensor([gamma_task, gamma_goal, gamma_bond], device=device)
            gamma0 = pyro.sample("gamma0", dist.Normal(true_gamma0, 0.1))
            gamma_eta = pyro.sample("gamma_eta", dist.Normal(true_gamma_eta, 0.1).to_event(1))
        else:
            gamma0 = pyro.sample("gamma0", dist.HalfNormal(1.0))
            gamma_eta = pyro.sample("gamma_eta", dist.Normal(0.0, 1.0).to_event(1))

        # --- 2. フィルタリングのロジックを Pyro 内で実行 ---
        with pyro.plate("individuals", N):
            # フィルターの初期化
            if sim_mode == 'informative':
                log_beliefs = torch.log(torch.tensor([0.999, 0.001], device=device))
            else:
                log_beliefs = torch.log(torch.tensor([0.5, 0.5], device=device))

            for t in pyro.markov(range(Nt)):
                # --- 予測ステップ ---
                if t > 0:
                    eta_t_minus_1 = eta_3fac_series[:, t-1, :]
                    logit_p11_t = gamma0 + (eta_t_minus_1 * gamma_eta).sum(-1)
                    
                    log_p11_t = F.logsigmoid(logit_p11_t)
                    log_p12_t = F.logsigmoid(-logit_p11_t)
                    
                    p22 = torch.tensor(0.9999, device=device)
                    log_p22 = torch.log(p22)
                    log_p21 = torch.log(1.0 - p22)

                    row1 = torch.stack([log_p11_t, log_p12_t], dim=-1)
                    # ▼▼▼【ここを修正】▼▼▼
                    row2 = torch.stack([log_p21, log_p22], dim=-1).expand(N, -1)
                    # ▲▲▲【修正ここまで】▲▲▲
                    log_transition_matrix = torch.stack([row1, row2], dim=1)
                    
                    # 予測信念 P(S_t | y_{1:t-1})
                    log_beliefs = torch.logsumexp(log_beliefs.unsqueeze(2) + log_transition_matrix, dim=1)

                # --- 更新ステップ ---
                # 観測尤度 P(y_t | S_t) を事前計算済みの値から取得
                log_liks_t = torch.stack([log_lik_3fac[:, t], log_lik_1fac[:, t]], dim=-1)
                
                # 混合対数尤度 log P(y_t | y_{1:t-1})
                marginal_log_lik = torch.logsumexp(log_beliefs + log_liks_t, dim=-1)
                
                # 目的関数に混合対数尤度を追加
                pyro.factor(f"obs_log_prob_{t}", marginal_log_lik)

                # 更新後信念 log P(S_t | y_{1:t})
                log_beliefs = (log_beliefs + log_liks_t) - marginal_log_lik.unsqueeze(-1)

    # --- BPSモデルへの入力データを準備 ---
    print("\n--- Pre-calculating inputs for BPS Fixed-Mixture model (using TRAINING data) ---")
    with torch.no_grad():
        log_lik_3fac_train = get_kalman_log_likelihoods_per_step(Y_train, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
        log_lik_1fac_train = get_kalman_log_likelihoods_per_step(Y_train, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
        _, latents_3fac_train, _ = get_kalman_predictions_and_latents(Y_train, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
    print("Pre-calculation complete.")

    # --- VIの実行または結果の読み込み ---
    print(f"\n\n--- 4. Defining and Training BPS Fixed-Mixture Model (mode='{SIMULATION_MODE}') ---")
    bps_results_file = f'bps_fixed_mixture_results_{SIMULATION_MODE}.pt'

    if os.path.exists(bps_results_file):
        print(f"Loading pre-computed BPS Fixed-Mixture results from '{bps_results_file}'...")
        saved_data_bps = torch.load(bps_results_file, weights_only=False)
        best_params_bps = saved_data_bps['best_params']; final_bps_loss = saved_data_bps.get('loss', float('nan'))
        duration_bps = saved_data_bps.get('duration', float('nan')); guide = None # ガイドは学習時のみ存在
        print("Loading complete.")
    else:
        print("Starting BPS Fixed-Mixture model training with VI...")
        start_time_bps = time.time()
        
        guide = AutoDiagonalNormal(bps_fixed_mixture_model)
        pyro.clear_param_store()
        optimizer = Adam({"lr": 0.005}) # 少し学習率を上げても良いかもしれない
        svi = SVI(bps_fixed_mixture_model, guide, optimizer, loss=Trace_ELBO())
        
        n_steps = 4000
        patience, patience_counter, best_loss_bps = 200, 0, float('inf')
        for step in range(n_steps):
            # 渡すデータを変更
            loss = -svi.step(Y_train, latents_3fac_train, log_lik_3fac_train, log_lik_1fac_train, SIMULATION_MODE) / N

            if torch.isnan(torch.tensor(loss)): print("Loss became NaN. Stopping training."); break
            if loss < best_loss_bps:
                best_loss_bps = loss; patience_counter = 0
            else:
                patience_counter += 1
            if (step + 1) % 100 == 0: print(f" NegLogLik loss: {loss:.4f} (Best: {best_loss_bps:.4f})")
            if patience_counter >= patience: print(f"     -> Early stopping triggered at step {step + 1}."); break
                
        duration_bps = time.time() - start_time_bps
        final_bps_loss = best_loss_bps
        print(f"VI training finished. Duration: {duration_bps:.2f}s")
        best_params_bps = guide.median()
        
        print("\n--- Best BPS Fixed-Mixture Parameters (Median from Guide) ---")
        for name, value in best_params_bps.items(): print(f"  Best {name}: {np.round(value.cpu().numpy(), 4)}")
        print(f"\nSaving BPS Fixed-Mixture results to '{bps_results_file}'...")
        results_to_save_bps = {'loss': final_bps_loss, 'best_params': best_params_bps, 'duration': duration_bps}
        torch.save(results_to_save_bps, bps_results_file)
        print("Saving complete.")

    # ----------------------------------------------------------------
    # Part 5: BPS固定混合モデルの評価
    # ----------------------------------------------------------------
    print("\n--- 5. Evaluating BPS Fixed-Mixture Model on Test Set ---")

    # Part 5 内の evaluate_bps_fixed_mixture 関数を置き換え

    def evaluate_bps_fixed_mixture(params, y_full, eta_3fac_full, log_lik_3fac_full, log_lik_1fac_full, sim_mode):
        """学習したgammaパラメータを使い、全期間の状態確率（重み）を計算する"""
        N, Nt, O = y_full.shape
        gamma0 = params['gamma0']; gamma_eta = params['gamma_eta']
        
        weights_history = torch.zeros(N, Nt, 2, device=device)

        with torch.no_grad():
            if sim_mode == 'informative':
                log_beliefs = torch.log(torch.tensor([0.999, 0.001], device=device)).expand(N, -1)
            else:
                log_beliefs = torch.log(torch.tensor([0.5, 0.5], device=device)).expand(N, -1)

            for t in range(Nt):
                if t > 0:
                    eta_t_minus_1 = eta_3fac_full[:, t-1, :]
                    logit_p11_t = gamma0 + (eta_t_minus_1 * gamma_eta).sum(-1)
                    log_p11_t = F.logsigmoid(logit_p11_t); log_p12_t = F.logsigmoid(-logit_p11_t)
                    p22 = torch.tensor(0.9999, device=device); log_p22 = torch.log(p22); log_p21 = torch.log(1.0 - p22)
                    
                    row1 = torch.stack([log_p11_t, log_p12_t], dim=-1)
                    
                    # ▼▼▼【ここを修正】▼▼▼
                    # 学習用関数と同様に、row2をN人分に拡張する
                    row2 = torch.stack([log_p21, log_p22], dim=-1).expand(N, -1)
                    # ▲▲▲【修正ここまで】▲▲▲

                    log_transition_matrix = torch.stack([row1, row2], dim=1)
                    log_beliefs = torch.logsumexp(log_beliefs.unsqueeze(2) + log_transition_matrix, dim=1)
                
                log_liks_t = torch.stack([log_lik_3fac_full[:, t], log_lik_1fac_full[:, t]], dim=-1)
                marginal_log_lik = torch.logsumexp(log_beliefs + log_liks_t, dim=-1)
                log_beliefs = (log_beliefs + log_liks_t) - marginal_log_lik.unsqueeze(-1)
                weights_history[:, t, :] = torch.exp(log_beliefs)
                
        return weights_history

    # Part 5 の "評価に必要な全期間のデータを準備" セクション

    # --- 評価に必要な全期間のデータを準備 ---
    print("Pre-calculating inputs for full-period BPS evaluation...")
    with torch.no_grad():
        log_lik_3fac_full = get_kalman_log_likelihoods_per_step(Y_generated, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
        log_lik_1fac_full = get_kalman_log_likelihoods_per_step(Y_generated, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
        _, latents_3fac_full, _ = get_kalman_predictions_and_latents(Y_generated, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
        
        # ▼▼▼【ここに追加】▼▼▼
        # 1因子モデルの全期間の潜在変数(eta)を計算する行が抜けていたため追加
        _, latents_1fac_full, _ = get_kalman_predictions_and_latents(Y_generated, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
        # ▲▲▲【追加ここまで】▲▲▲

        preds_3fac_full, _, _ = get_kalman_predictions_and_latents(Y_generated, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
        preds_1fac_full, _, _ = get_kalman_predictions_and_latents(Y_generated, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
    print("Pre-calculation complete.")

    # --- 全期間の重みを計算 ---
    weights_full = evaluate_bps_fixed_mixture(best_params_bps, Y_generated, latents_3fac_full, log_lik_3fac_full, log_lik_1fac_full, SIMULATION_MODE)
    weights_test = weights_full[:, Nt_train:, :]

    # --- テストデータで評価指標を計算 ---
    # (この部分は以前のコードと同じ)
    preds_3fac_test = preds_3fac_full[:, Nt_train:, :]
    preds_1fac_test = preds_1fac_full[:, Nt_train:, :]
    y_pred_mixed_test = weights_test[..., 0].unsqueeze(-1) * preds_3fac_test + weights_test[..., 1].unsqueeze(-1) * preds_1fac_test
    rmse_bps = calculate_rmse(Y_test, y_pred_mixed_test)
    ssr_eta_bps = calculate_ssr_eta(eta_true_history_test.cpu(), (latents_3fac_full[:, Nt_train:, :].cpu(), latents_1fac_full[:, Nt_train:, :].cpu(), weights_test.cpu()), actual_states_test, model_type='BPS')
    predicted_states_bps = (weights_test[:, :, 1] > 0.5).cpu().numpy().astype(int)
    sens_bps, spec_bps = calculate_sens_spec(actual_states_test, predicted_states_bps)
    print(f"Test RMSE: {rmse_bps:.4f}")
    print(f"Test Sensitivity: {sens_bps:.4f}, Specificity: {spec_bps:.4f}")
    print(f"Test Forecast SSR(eta): {ssr_eta_bps:.4f}")

    # ▲▲▲【置き換えここまで】▲▲▲
    # ==============================================================================

    # ----------------------------------------------------------------
    # Part 6: RS-KFモデルの定義と学習（メモに基づき修正）
    # ----------------------------------------------------------------
    print("\n\n--- 6. Defining and Training the Regime-Switching Kalman Filter Model ---")
    rs_kf_results_file = 'rs_kf_results.pt'

    class RegimeSwitchingKF(torch.nn.Module):
        def __init__(self, O, time_weights, initial_belief='informative', init_params=None):
            super().__init__()
            self.initial_belief = initial_belief
            self.time_weights = time_weights
            self.O = O
            self.dim_g, self.dim_t, self.dim_b, self.dim_w = 1, 1, 1, 1
            self.L = self.dim_g + self.dim_t + self.dim_b + self.dim_w
            if init_params:
                print("     -> Initializing FRS model with pre-trained baseline parameters.")
                self.B_G = torch.nn.Parameter(init_params['B1_3fac'][0, 0].clone().reshape(self.dim_g, self.dim_g))
                self.B_T = torch.nn.Parameter(init_params['B1_3fac'][1, 1].clone().reshape(self.dim_t, self.dim_t))
                self.B_B = torch.nn.Parameter(init_params['B1_3fac'][2, 2].clone().reshape(self.dim_b, self.dim_b))
                self.B_W = torch.nn.Parameter(init_params['B1_1fac'][0, 0].clone().reshape(self.dim_w, self.dim_w))
                self.lambda_r1_free = torch.nn.Parameter(init_params['lambda1_free_3fac'].clone())
                self.lambda_r2_free = torch.nn.Parameter(init_params['lambda1_free_1fac'].clone())
                q_diag_3fac = torch.diag(init_params['Q_3fac']); q_diag_1fac = torch.diag(init_params['Q_1fac'])
                initial_q_diag = torch.cat([q_diag_3fac, q_diag_1fac])
                self.q_diag = torch.nn.Parameter(initial_q_diag.clone())
                r_diag_3fac = torch.diag(init_params['R_3fac']); r_diag_1fac = torch.diag(init_params['R_1fac'])
                initial_r_diag = (r_diag_3fac + r_diag_1fac) / 2.0
                self.r_diag = torch.nn.Parameter(initial_r_diag.clone())
                # ▼▼▼【ここから修正】▼▼▼
                # HMMパラメータ(gamma)は、DGPの真値周辺からサンプリングして初期化
                print("     -> Initializing FRS HMM parameters by SAMPLING AROUND TRUE DGP values.")
                std_dev = 0.1 # 真値に加えるノイズの大きさ
                self.gamma1 = torch.nn.Parameter(dist.Normal(torch.tensor(gamma_intercept, device=device), std_dev).sample())
                true_gamma2 = torch.tensor([gamma_task, gamma_goal, gamma_bond], device=device)
                self.gamma2_learnable = torch.nn.Parameter(dist.Normal(true_gamma2, std_dev).sample())
                # ▲▲▲【修正ここまで】▲▲▲
            else:
                print("     -> Initializing FRS model with random parameters.")
                beta_dist = torch.distributions.beta.Beta(5.0, 1.5)
                self.B_G = torch.nn.Parameter(beta_dist.sample().reshape(self.dim_g, self.dim_g))
                self.B_T = torch.nn.Parameter(beta_dist.sample().reshape(self.dim_t, self.dim_t))
                self.B_B = torch.nn.Parameter(beta_dist.sample().reshape(self.dim_b, self.dim_b))
                self.B_W = torch.nn.Parameter(beta_dist.sample().reshape(self.dim_w, self.dim_w))
                self.lambda_r1_free = torch.nn.Parameter(torch.randn(6))
                self.lambda_r2_free = torch.nn.Parameter(torch.randn(O - 1))
                self.q_diag = torch.nn.Parameter(torch.rand(self.L))
                self.r_diag = torch.nn.Parameter(torch.rand(O))
                self.gamma1 = torch.nn.Parameter(torch.abs(torch.randn(()))) 
                self.gamma2_learnable = torch.nn.Parameter(torch.randn(self.L - 1))

            self.eta_filtered_r1_history = None; self.eta_filtered_r2_history = None
            self.P_filtered_r1_history = None; self.P_filtered_r2_history = None
            self.filtered_probs = None; self.predicted_y = None
        def _build_matrices(self):
            L, dg, dt, db, dw = self.L, self.dim_g, self.dim_t, self.dim_b, self.dim_w
            B_1_to_1 = torch.zeros(L, L, device=device); B_1_to_1[0:dg, 0:dg] = self.B_G
            B_1_to_1[dg:dg+dt, dg:dg+dt] = self.B_T; B_1_to_1[dg+dt:dg+dt+db, dg+dt:dg+dt+db] = self.B_B
            B_2_to_2 = torch.zeros(L, L, device=device); B_2_to_2[dg+dt+db:, dg+dt+db:] = self.B_W
            B_1_to_2 = torch.zeros(L, L, device=device)
            B_1_to_2[dg+dt+db:, 0:dg] = 1/3; B_1_to_2[dg+dt+db:, dg:dg+dt] = 1/3; B_1_to_2[dg+dt+db:, dg+dt:dg+dt+db] = 1/3
            B_2_to_1 = torch.zeros(L, L, device=device)
            Lambda_r1 = torch.zeros(self.O, L, device=device)
            Lambda_r1[0, 0] = 1.0; Lambda_r1[1, 0] = self.lambda_r1_free[0]; Lambda_r1[2, 0] = self.lambda_r1_free[1]
            Lambda_r1[3, 1] = 1.0; Lambda_r1[4, 1] = self.lambda_r1_free[2]; Lambda_r1[5, 1] = self.lambda_r1_free[3]
            Lambda_r1[6, 2] = 1.0; Lambda_r1[7, 2] = self.lambda_r1_free[4]; Lambda_r1[8, 2] = self.lambda_r1_free[5]
            Lambda_r2 = torch.zeros(self.O, L, device=device)
            Lambda_r2[0, dg+dt+db] = 1.0; Lambda_r2[1:, dg+dt+db] = self.lambda_r2_free
            Q = torch.diag(self.q_diag.abs() + 1e-4); R = torch.diag(self.r_diag.abs() + 1e-4)
            return (B_1_to_1, B_2_to_1, B_1_to_2, B_2_to_2), (Lambda_r1, Lambda_r2), (Q, R)
        def forward(self, y):
            N, Nt, O = y.shape
            (B11, B21, B12, B22), (L1_lambda, L2_lambda), (Q, R) = self._build_matrices()
            self.eta_filtered_r1_history = torch.zeros(N, Nt, self.L, device=device); self.eta_filtered_r2_history = torch.zeros(N, Nt, self.L, device=device)
            self.P_filtered_r1_history = torch.zeros(N, Nt, self.L, self.L, device=device); self.P_filtered_r2_history = torch.zeros(N, Nt, self.L, self.L, device=device)
            self.filtered_probs = torch.zeros(N, Nt, 2, device=device); self.predicted_y = torch.zeros(N, Nt, O, device=device)
            prob_tm1 = torch.zeros(N, 2, device=device)
            eta_marg_r1_tm1 = torch.zeros(N, self.L, 1, device=device); eta_marg_r2_tm1 = torch.zeros(N, self.L, 1, device=device)
            initial_P_diag = torch.tensor([1e3, 1e3, 1e3, 1e-9], device=device); P_0 = torch.diag(initial_P_diag).expand(N, -1, -1)
            P_marg_r1_tm1 = P_0; P_marg_r2_tm1 = P_0.clone()
            if self.initial_belief == 'informative': prob_tm1[:, 0] = 0.99; prob_tm1[:, 1] = 0.01
            else: prob_tm1[:, 0] = 0.5; prob_tm1[:, 1] = 0.5
            total_log_likelihood = 0.0
            for t in range(Nt):
                y_t = y[:, t, :].unsqueeze(-1)
                eta_state1_components = eta_marg_r1_tm1[:, :self.L-1, :].squeeze(-1)
                logit_p11 = self.gamma1 + (eta_state1_components * self.gamma2_learnable).sum(-1)
                p11 = torch.sigmoid(logit_p11); p22 = torch.full((N,), 0.9999, device=device)
                transition_probs = torch.stack([torch.stack([p11, 1-p11], dim=1), torch.stack([1-p22, p22], dim=1)], dim=1)
                eta_pred_11 = B11 @ eta_marg_r1_tm1; P_pred_11 = B11 @ P_marg_r1_tm1 @ B11.mT + Q
                eta_pred_12 = B21 @ eta_marg_r2_tm1; P_pred_12 = B21 @ P_marg_r2_tm1 @ B21.mT + Q
                eta_pred_21 = B12 @ eta_marg_r1_tm1; P_pred_21 = B12 @ P_marg_r1_tm1 @ B12.mT + Q
                eta_pred_22 = B22 @ eta_marg_r2_tm1; P_pred_22 = B22 @ P_marg_r2_tm1 @ B22.mT + Q
                v_11 = y_t - L1_lambda @ eta_pred_11; F_11 = L1_lambda @ P_pred_11 @ L1_lambda.mT + R
                v_12 = y_t - L1_lambda @ eta_pred_12; F_12 = L1_lambda @ P_pred_12 @ L1_lambda.mT + R
                v_21 = y_t - L2_lambda @ eta_pred_21; F_21 = L2_lambda @ P_pred_21 @ L2_lambda.mT + R
                v_22 = y_t - L2_lambda @ eta_pred_22; F_22 = L2_lambda @ P_pred_22 @ L2_lambda.mT + R
                f_jitter = torch.eye(O, device=device) * 1e-5
                F_11 += f_jitter; F_12 += f_jitter; F_21 += f_jitter; F_22 += f_jitter
                log_lik_11 = MultivariateNormal(loc=torch.zeros_like(v_11.squeeze(-1)), covariance_matrix=F_11).log_prob(v_11.squeeze(-1))
                log_lik_12 = MultivariateNormal(loc=torch.zeros_like(v_12.squeeze(-1)), covariance_matrix=F_12).log_prob(v_12.squeeze(-1))
                log_lik_21 = MultivariateNormal(loc=torch.zeros_like(v_21.squeeze(-1)), covariance_matrix=F_21).log_prob(v_21.squeeze(-1))
                log_lik_22 = MultivariateNormal(loc=torch.zeros_like(v_22.squeeze(-1)), covariance_matrix=F_22).log_prob(v_22.squeeze(-1))
                log_prob_tm1 = torch.log(prob_tm1 + 1e-9)
                log_prob_t_11 = log_prob_tm1[:, 0] + torch.log(transition_probs[:, 0, 0] + 1e-9) + log_lik_11
                log_prob_t_12 = log_prob_tm1[:, 1] + torch.log(transition_probs[:, 1, 0] + 1e-9) + log_lik_12
                log_prob_t_21 = log_prob_tm1[:, 0] + torch.log(transition_probs[:, 0, 1] + 1e-9) + log_lik_21
                log_prob_t_22 = log_prob_tm1[:, 1] + torch.log(transition_probs[:, 1, 1] + 1e-9) + log_lik_22
                log_prob_t = torch.stack([log_prob_t_11, log_prob_t_12, log_prob_t_21, log_prob_t_22], dim=1)
                log_likelihood_t = torch.logsumexp(log_prob_t, dim=1)
                total_log_likelihood += (self.time_weights[t] * log_likelihood_t).sum()
                prob_t = torch.exp(log_prob_t - log_likelihood_t.unsqueeze(1))
                prob_t_r1 = prob_t[:, 0] + prob_t[:, 1]; prob_t_r2 = prob_t[:, 2] + prob_t[:, 3]
                W_11 = prob_t[:, 0] / (prob_t_r1 + 1e-9); W_12 = prob_t[:, 1] / (prob_t_r1 + 1e-9)
                W_21 = prob_t[:, 2] / (prob_t_r2 + 1e-9); W_22 = prob_t[:, 3] / (prob_t_r2 + 1e-9)
                K_11 = P_pred_11 @ L1_lambda.mT @ torch.linalg.pinv(F_11); eta_upd_11 = eta_pred_11 + K_11 @ v_11
                K_21 = P_pred_21 @ L2_lambda.mT @ torch.linalg.pinv(F_21); eta_upd_21 = eta_pred_21 + K_21 @ v_21
                K_22 = P_pred_22 @ L2_lambda.mT @ torch.linalg.pinv(F_22); eta_upd_22 = eta_pred_22 + K_22 @ v_22
                I_L = torch.eye(self.L, device=device)
                I_KL_11 = I_L - K_11 @ L1_lambda; P_upd_11 = I_KL_11 @ P_pred_11 @ I_KL_11.mT + K_11 @ R @ K_11.mT
                I_KL_21 = I_L - K_21 @ L2_lambda; P_upd_21 = I_KL_21 @ P_pred_21 @ I_KL_21.mT + K_21 @ R @ K_21.mT
                I_KL_22 = I_L - K_22 @ L2_lambda; P_upd_22 = I_KL_22 @ P_pred_22 @ I_KL_22.mT + K_22 @ R @ K_22.mT
                eta_marg_r1_t = eta_upd_11; P_marg_r1_t = P_upd_11
                eta_marg_r2_t = W_21.view(N, 1, 1) * eta_upd_21 + W_22.view(N, 1, 1) * eta_upd_22
                P_marg_r2_t = W_21.view(N,1,1) * (P_upd_21 + (eta_marg_r2_t-eta_upd_21) @ (eta_marg_r2_t-eta_upd_21).transpose(-1,-2)) + W_22.view(N,1,1) * (P_upd_22 + (eta_marg_r2_t-eta_upd_22) @ (eta_marg_r2_t-eta_upd_22).transpose(-1,-2))
                eta_marg_r1_tm1, P_marg_r1_tm1 = eta_marg_r1_t, P_marg_r1_t
                eta_marg_r2_tm1, P_marg_r2_tm1 = eta_marg_r2_t, P_marg_r2_t
                prob_tm1 = torch.stack([prob_t_r1, prob_t_r2], dim=1)
                self.eta_filtered_r1_history[:, t, :] = eta_marg_r1_t.squeeze(-1)
                self.eta_filtered_r2_history[:, t, :] = eta_marg_r2_t.squeeze(-1)
                self.P_filtered_r1_history[:, t, :, :] = P_marg_r1_t
                self.P_filtered_r2_history[:, t, :, :] = P_marg_r2_t
                self.filtered_probs[:, t, :] = prob_tm1
                y_pred_t = prob_tm1[:,0].view(N,1,1)* (L1_lambda @ eta_marg_r1_t) + prob_tm1[:,1].view(N,1,1)* (L2_lambda @ eta_marg_r2_t)
                self.predicted_y[:, t, :] = y_pred_t.squeeze(-1)
            return -total_log_likelihood

    # FRSモデルに渡す初期値の辞書を作成
    frs_initial_params = {
        'B1_3fac': best_params_3fac['B1'], 'B1_1fac': best_params_1fac['B1'],
        'lambda1_free_3fac': torch.cat([best_params_3fac['Lambda1'][1:3, 0], best_params_3fac['Lambda1'][4:6, 1], best_params_3fac['Lambda1'][7:9, 2]]),
        'lambda1_free_1fac': best_params_1fac['Lambda1'][1:9, 0],
        'Q_3fac': best_params_3fac['Q'], 'Q_1fac': best_params_1fac['Q'],
        'R_3fac': best_params_3fac['R'], 'R_1fac': best_params_1fac['R']
    }
    num_runs_rs = 1; num_epochs_rs = 10000; learning_rate_rs = 1e-3
    patience = 100

    if os.path.exists(rs_kf_results_file):
        print(f"Loading pre-computed RS-KF model results from '{rs_kf_results_file}'...")
        saved_data = torch.load(rs_kf_results_file, weights_only=False)
        best_model_rs_state = saved_data['model_state_dict']
        final_loss_rs = saved_data['loss']; rmse_rs = saved_data['rmse']; sens_rs = saved_data['sens']; spec_rs = saved_data['spec']; ssr_eta_rs = saved_data.get('ssr_eta', float('nan'))
        duration_rs = saved_data.get('duration', 0.0); failed_runs_rs = 0; history_rs = {}
        print("RS-KF Loading complete.")
    else:
        start_time_rs = time.time()
        best_loss_rs = float('inf'); best_model_rs_state = None; failed_runs_rs = 0
        param_names_to_track_rs = ['B_G', 'B_T', 'B_B', 'B_W', 'lambda_r1_free', 'lambda_r2_free', 'q_diag', 'r_diag', 'gamma1', 'gamma2_learnable']
        history_rs = {}
        
        for run in range(num_runs_rs):
            print(f"Starting Run {run+1}/{num_runs_rs}...")
            # ▼▼▼【ここを修正】▼▼▼
            # 訓練用の時間重み(time_weights_train)を渡す
            model_rs = RegimeSwitchingKF(O, time_weights_train, initial_belief=SIMULATION_MODE, init_params=frs_initial_params).to(device)
            # ▲▲▲【ここまで修正】▲▲▲            
            optimizer_rs = torch.optim.Adam(model_rs.parameters(), lr=learning_rate_rs)
            patience_counter = 0; best_loss_in_run = float('inf'); best_model_state_in_run = None
            run_history = {name: [] for name in param_names_to_track_rs}

            # ▼▼▼【ここから修正】▼▼▼
            absolute_threshold = 0.1

            for epoch in range(num_epochs_rs):
                optimizer_rs.zero_grad()
                loss = model_rs(Y_train)
                if torch.isnan(loss):
                    failed_runs_rs += 1; print(f"Run {run+1} failed due to NaN loss."); break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_rs.parameters(), 1.0)
                optimizer_rs.step()
                
                if best_loss_in_run - loss.item() > absolute_threshold:
                    best_loss_in_run = loss.item()
                    patience_counter = 0
                    best_model_state_in_run = model_rs.state_dict()
                else:
                    patience_counter += 1

                if (epoch + 1) % 100 == 0:
                    print(f"   [RS-KF Run {run + 1}, Epoch {epoch + 1:04d}] loss: {loss.item():.4f} (Best in run: {best_loss_in_run:.4f})")
                    with torch.no_grad():
                        for name in run_history.keys():
                            param_value = getattr(model_rs, name)
                            run_history[name].append(param_value.clone().cpu().numpy())
                
                if patience_counter >= patience:
                    print(f"    -> Early stopping triggered at epoch {epoch + 1}.")
                    break
            # ▲▲▲【ここまで修正】▲▲▲

            if best_loss_in_run < best_loss_rs:
                best_loss_rs = best_loss_in_run
                best_model_rs_state = best_model_state_in_run
                history_rs = run_history
                print(f"-> New overall best loss for RS-KF Model found: {best_loss_rs:.4f}")
            
            duration_rs = time.time() - start_time_rs
            
            print("\n--- RS-KF Parameter Update History (Best Run) ---")
            if history_rs:
                for name, history in history_rs.items():
                    print(f"\nParameter: {name}")
                    for i, val in enumerate(history):
                        epoch_num = (i + 1) * 100
                        display_val = 1 / (1 + np.exp(-val)) if 'logit' in name else (np.exp(val) if 'log_' in name else val)
                        print(f"  Epoch {epoch_num:>4d}: {np.round(display_val.flatten(), 4)}")

            # ▼▼▼【ここから修正】▼▼▼
            # 最終モデルをインスタンス化し、学習済みベストモデルの状態を読み込む
            _final_model_rs = RegimeSwitchingKF(O, time_weights_train, initial_belief=SIMULATION_MODE, init_params=frs_initial_params).to(device)
            if best_model_rs_state: 
                _final_model_rs.load_state_dict(best_model_rs_state)
            
            # 訓練データでの最終的な損失を記録
            with torch.no_grad():
                final_loss_rs = _final_model_rs(Y_train).item()
            
            # --- テストデータでの評価 ---
            print("\n--- Evaluating RS-KF Model on Test Set ---")
            y_preds_rs_forecast, probs_rs_forecast, eta_r1_forecast, eta_r2_forecast, _, _ = forecast_rs(_final_model_rs, Y_generated, Nt_train)

            # RMSEをテストデータで評価
            rmse_rs = calculate_rmse(Y_test, y_preds_rs_forecast)
            
            # Sens/Specをテストデータで評価
            predicted_states_rs = (probs_rs_forecast[:, :, 1] > 0.5).cpu().numpy().astype(int)
            sens_rs, spec_rs = calculate_sens_spec(actual_states_test, predicted_states_rs)
            
            # SSR(eta)を予測値で計算
            ssr_eta_rs = calculate_ssr_eta(
                eta_true_history_test.cpu(),
                (eta_r1_forecast.cpu(), eta_r2_forecast.cpu()),
                actual_states_test,
                model_type='FRS'
            )
            
            print(f"Test RMSE: {rmse_rs:.4f}")
            print(f"Test Sensitivity: {sens_rs:.4f}, Specificity: {spec_rs:.4f}")
            print(f"Test Forecast SSR(eta): {ssr_eta_rs:.4f}")

            print("RS-KF model training finished.")
            print(f"Saving RS-KF model results to '{rs_kf_results_file}'...")

            # 保存するメトリクスもテストデータに対するものに更新
            results_to_save = {
                'model_state_dict': best_model_rs_state, 'loss': final_loss_rs, 
                'rmse': rmse_rs, 'sens': sens_rs, 'spec': spec_rs, 
                'duration': duration_rs, 'ssr_eta': ssr_eta_rs
            }
            torch.save(results_to_save, rs_kf_results_file)
            print("RS-KF Saving complete.")
            # ▲▲▲【ここまで修正】▲▲▲

    # --- RS-KF評価 (プロット用変数の計算) ---
    # ▼▼▼【ここから修正】▼▼▼
    # 最終モデルをロードし、テストデータでの予測確率を取得する
    final_model_rs = RegimeSwitchingKF(O, time_weights_train, initial_belief=SIMULATION_MODE, init_params=frs_initial_params).to(device)
    if 'best_model_rs_state' in locals() and best_model_rs_state:
        final_model_rs.load_state_dict(best_model_rs_state)
    else: # ファイルから読み込む場合
        saved_data = torch.load(rs_kf_results_file, weights_only=False)
        final_model_rs.load_state_dict(saved_data['model_state_dict'])

    with torch.no_grad():
        # 訓練データでのフィルタリング結果と、テストデータでの予測結果を結合する
        # 訓練期間の確率
        _ = final_model_rs(Y_train)
        probs_rs_train = final_model_rs.filtered_probs.cpu().numpy()
        
        # テスト期間の予測確率
        _, probs_rs_forecast_final, _, _, _, _ = forecast_rs(final_model_rs, Y_generated, Nt_train)
        probs_rs_test = probs_rs_forecast_final.cpu().numpy()
        
        # 全期間の確率として結合
        predicted_probs_rs = np.concatenate((probs_rs_train, probs_rs_test), axis=1)
    # ▲▲▲【ここまで修正】▲▲▲
                
    # ----------------------------------------------------------------
    # Part 7: 最終的なモデル比較
    # ----------------------------------------------------------------
    print("\n\n--- 7. Final Model Fit Comparison ---")
    
    # ファイル読み込み時にSSR(eta)が未定義の場合に備えて再取得
    if 'ssr_eta_3fac' not in locals() and os.path.exists(results_file_3fac):
        ssr_eta_3fac = torch.load(results_file_3fac, weights_only=False).get('ssr_eta', float('nan'))
    if 'ssr_eta_1fac' not in locals() and os.path.exists(results_file_1fac):
        ssr_eta_1fac = torch.load(results_file_1fac, weights_only=False).get('ssr_eta', float('nan'))
    # === ▼ 修正箇所 ▼ ===
    if 'ssr_eta_bps' not in locals() and os.path.exists(bps_results_file): # bps_metrics_file -> bps_results_file に変更
        # BPSの結果ファイルからは直接ssr_eta_bpsを読み込まず、後で計算するため、この行はコメントアウトまたは削除します。
        # ssr_eta_bps = torch.load(bps_results_file, weights_only=False).get('ssr_eta', float('nan'))
        pass # あとの評価パートで計算するのでここでは何もしません
    # === ▲ 修正ここまで ▲ ===

    table_width = 133
    print("="*table_width)
    # ▼▼▼【ここを修正】▼▼▼
    print(f"{'Model':<30} | {'Train Loss':<15} | {'Test RMSE':<15} | {'Test SSR (eta)':<15} | {'Test Sensitivity':<15} | {'Test Specificity':<15} | {'Time (s)':<25}")
    # ▲▲▲【ここまで修正】▲▲▲    
    print("-"*table_width)
    
    time_str_3fac = f"{duration_3fac:.2f} (Failed: {failed_runs_3fac}/{num_runs})"
    print(f"{'3-Factor Model':<30} | {best_loss_3fac:<15.4f} | {rmse_3fac:<15.4f} | {ssr_eta_3fac:<15.4f} | {sens_3fac:<15.4f} | {spec_3fac:<15.4f} | {time_str_3fac:<25}")
    
    time_str_1fac = f"{duration_1fac:.2f} (Failed: {failed_runs_1fac}/{num_runs})"
    print(f"{'1-Factor Model':<30} | {best_loss_1fac:<15.4f} | {rmse_1fac:<15.4f} | {ssr_eta_1fac:<15.4f} | {sens_1fac:<15.4f} | {spec_1fac:<15.4f} | {time_str_1fac:<25}")
    
    print(f"{'BPS Hybrid Personalized':<30} | {final_bps_loss:<15.4f} | {rmse_bps:<15.4f} | {ssr_eta_bps:<15.4f} | {sens_bps:<15.4f} | {spec_bps:<15.4f} | {duration_bps:<25.2f}")
    
    time_str_rs = f"{duration_rs:.2f} (Failed: {failed_runs_rs}/{num_runs_rs})"
    print(f"{'Regime-Switching KF':<30} | {final_loss_rs:<15.4f} | {rmse_rs:<15.4f} | {ssr_eta_rs:<15.4f} | {sens_rs:<15.4f} | {spec_rs:<15.4f} | {time_str_rs:<25}")
    
    print("\nNote: Loss for Baseline/RS-KF is NegLogLikelihood. Loss for BPS is a part of ELBO.")
    print("      RMSE, SSR(eta), Sensitivity, and Specificity are directly comparable metrics.")
    print("="*table_width)
    print("="*table_width)

    # ==============================================================================
    # ▼▼▼【Part 8 を以下に置き換え】▼▼▼
    # ----------------------------------------------------------------
    # Part 8: プロット用データの準備
    # ----------------------------------------------------------------
    print("\n\n--- 8. Preparing data for plotting ---")

    # BPSモデルの平均重みを計算 (Part 5の結果を直接使用)
    estimated_weights_bps_full = weights_full.mean(dim=0).cpu().numpy()

    # RS-KFモデルの平均確率を計算 (訓練期間とテスト期間の結果を結合)
    # (この部分は変更なしでOK)
    final_model_rs = RegimeSwitchingKF(O, time_weights_train, initial_belief=SIMULATION_MODE, init_params=frs_initial_params).to(device)
    if 'best_model_rs_state' in locals() and best_model_rs_state:
        final_model_rs.load_state_dict(best_model_rs_state)
    else:
        saved_data = torch.load(rs_kf_results_file, weights_only=False)
        final_model_rs.load_state_dict(saved_data['model_state_dict'])

    with torch.no_grad():
        _ = final_model_rs(Y_train)
        probs_rs_train = final_model_rs.filtered_probs.cpu().numpy()
        _, probs_rs_forecast_final, _, _, _, _ = forecast_rs(final_model_rs, Y_generated, Nt_train)
        probs_rs_test = probs_rs_forecast_final.cpu().numpy()
        predicted_probs_rs = np.concatenate((probs_rs_train, probs_rs_test), axis=1)

    # Part 8a を修正

    # ▼▼▼【Part 8a を以下に置き換え】▼▼▼
    # --- 8a. 事後サンプル生成（プロット用） ---
    # プロット用の変数を事前に初期化
    bps_lower_ci, bps_upper_ci = None, None

    if guide is not None:
        print("\n--- 8a. Generating posterior samples for BPS model uncertainty ---")
        guide.requires_grad_(False)
        
        predictive = Predictive(bps_fixed_mixture_model, guide=guide, num_samples=1000,
                                return_sites=("gamma0", "gamma_eta"))
        param_samples = predictive(Y_train, latents_3fac_train, log_lik_3fac_train, log_lik_1fac_train, SIMULATION_MODE)
        print("Generated 1000 samples from the posterior distribution of parameters.")

        num_samples = 1000
        bps_prob_samples_s2 = torch.zeros(num_samples, N, Nt, device=device)

        print("Calculating state probability trajectories for each parameter sample...")
        for i in range(num_samples):
            params_i = {'gamma0': param_samples['gamma0'][i], 'gamma_eta': param_samples['gamma_eta'][i]}
            weights_sample = evaluate_bps_fixed_mixture(
                params_i, Y_generated, latents_3fac_full, 
                log_lik_3fac_full, log_lik_1fac_full, SIMULATION_MODE
            )
            bps_prob_samples_s2[i, :, :] = weights_sample[:, :, 1]
            if (i + 1) % 100 == 0: print(f"  Processed {i+1}/{num_samples} samples...")
        print("Calculation complete.")
        
        bps_prob_samples_s2_permuted = bps_prob_samples_s2.permute(1, 2, 0)
        bps_lower_ci = torch.quantile(bps_prob_samples_s2_permuted, 0.025, dim=-1).cpu().numpy()
        bps_upper_ci = torch.quantile(bps_prob_samples_s2_permuted, 0.975, dim=-1).cpu().numpy()
    # ▲▲▲【置き換えここまで】▲▲▲

    # ----------------------------------------------------------------
    # Part 9: 比較グラフの描画
    # ----------------------------------------------------------------
    print("\n\n--- 9. Visualization of Final Results ---")
    # (このパートは変更なしでOK)
    state1_proportion = (actual_states == 1).mean(axis=0)
    # ... (plt.subplots から plt.show() までの描画コードはそのまま) ...


    # ----------------------------------------------------------------
    # Part 10: Visualization with Uncertainty Intervals
    # ----------------------------------------------------------------
    print("\n\n--- 10. Generating Plots with Uncertainty Intervals ---")

    individual_ids_to_plot = [10, 25, 50, 75]
    if N < max(individual_ids_to_plot):
        individual_ids_to_plot = random.sample(range(N), 4)

    # Part 10a を修正

    # ▼▼▼【Part 10a を以下に置き換え】▼▼▼
    # ----------------------------------------------------------------
    # Part 10a: 個人レベルのレジームスイッチ過程のプロット
    # ----------------------------------------------------------------
    print("\n--- 10a. Generating individual regime switch plot for BPS (HMM) ---")

    # --- プロット作成 ---
    weight_mean_full = weights_full.cpu().numpy()
    actual_states_binary = (actual_states == 2).astype(int)
    time_points = np.arange(Nt)

    fig, axes = plt.subplots(len(individual_ids_to_plot), 2, figsize=(16, 4 * len(individual_ids_to_plot)), sharex=True, sharey=True)
    fig.suptitle('Individual Regime Switch Trajectories (Filtered on Train, Forecast on Test)', fontsize=20)

    for i, ind_id in enumerate(individual_ids_to_plot):
        # BPSのプロット
        ax_bps = axes[i, 0]
        ax_bps.plot(time_points, weight_mean_full[ind_id, :, 1], 'g-', label='BPS Prob. (State 2)', lw=2)
        
        # 信頼区間データが存在する場合のみ、塗りつぶしを実行
        if bps_lower_ci is not None and bps_upper_ci is not None:
            ax_bps.fill_between(time_points, bps_lower_ci[ind_id, :], bps_upper_ci[ind_id, :], color='green', alpha=0.2, label='95% Credible Interval')

        ax_bps.plot(time_points, actual_states_binary[ind_id, :], 'r--', label='True State', lw=2)
        ax_bps.set_title(f'Individual #{ind_id} - BPS Filter Model')
        ax_bps.set_ylabel('Probability of State 2')
        
        # RS-KFのプロット
        ax_rs = axes[i, 1]
        ax_rs.plot(time_points, predicted_probs_rs[ind_id, :, 1], 'b-', label='RS-KF Prob. (State 2)', lw=2)
        ax_rs.plot(time_points, actual_states_binary[ind_id, :], 'r--', label='True State', lw=2)
        ax_rs.set_title(f'Individual #{ind_id} - RS-KF Model')

        # 共通設定
        ax_bps.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2)
        ax_rs.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2)
        if i == 0:
            ax_bps.legend(loc='upper left')
            ax_rs.legend(loc='upper left')

    axes[-1, 0].set_xlabel('Time Point')
    axes[-1, 1].set_xlabel('Time Point')
    plt.ylim(-0.1, 1.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("individual_switches_plot_with_ci.png")
    print("Individual switch plot with C.I. saved to individual_switches_plot_with_ci.png")
    plt.show()

    # ▲▲▲【置き換えここまで】▲▲▲

    # ----------------------------------------------------------------
    # Part 10b: 潜在変数 eta の比較プロット (信頼区間付き)
    # ----------------------------------------------------------------
    print("\n--- 10b. Generating latent variable plot with uncertainty (Filtered on Train, Forecast on Test) ---")

    # --- プロット用の全期間etaとPを準備 ---
    with torch.no_grad():
        # ベースラインモデルのetaとPを取得
        # 訓練期間（フィルター値）
        _, latents_3fac_train, P_3fac_train = get_kalman_predictions_and_latents(Y_train, best_params_3fac['b0'], best_params_3fac['B1'], best_params_3fac['Lambda1'], best_params_3fac['Q'], best_params_3fac['R'])
        # テスト期間（予測値）
        _, latents_3fac_test, P_3fac_test = generate_forecasts(best_params_3fac, Y_generated, Nt_train)
        # 結合
        latents_3fac_full = torch.cat((latents_3fac_train, latents_3fac_test), dim=1).cpu().numpy()
        P_3fac_full = torch.cat((P_3fac_train, P_3fac_test), dim=1).cpu().numpy()

        # ▼▼▼【ここから追加】▼▼▼
        # 1-FactorモデルのetaとPを取得
        # 訓練期間（フィルター値）
        _, latents_1fac_train, _ = get_kalman_predictions_and_latents(Y_train, best_params_1fac['b0'], best_params_1fac['B1'], best_params_1fac['Lambda1'], best_params_1fac['Q'], best_params_1fac['R'])
        # テスト期間（予測値）
        _, latents_1fac_test, _ = generate_forecasts(best_params_1fac, Y_generated, Nt_train)
        # 結合
        latents_1fac_full = torch.cat((latents_1fac_train, latents_1fac_test), dim=1).cpu().numpy()
        # ▲▲▲【ここまで追加】▲▲▲

        # RS-KFモデルのetaとPを取得
        # 訓練期間（フィルター値）
        _ = final_model_rs(Y_train)
        eta_r1_train_rs = final_model_rs.eta_filtered_r1_history
        P_r1_train_rs = final_model_rs.P_filtered_r1_history
        eta_r2_train_rs = final_model_rs.eta_filtered_r2_history
        P_r2_train_rs = final_model_rs.P_filtered_r2_history
        # テスト期間（予測値）
        _, _, eta_r1_test_rs, eta_r2_test_rs, P_r1_test_rs, P_r2_test_rs = forecast_rs(final_model_rs, Y_generated, Nt_train)
        # 結合
        eta_r1_full_rs = torch.cat((eta_r1_train_rs, eta_r1_test_rs), dim=1).cpu().numpy()
        P_r1_full_rs = torch.cat((P_r1_train_rs, P_r1_test_rs), dim=1).cpu().numpy()
        eta_r2_full_rs = torch.cat((eta_r2_train_rs, eta_r2_test_rs), dim=1).cpu().numpy()
        P_r2_full_rs = torch.cat((P_r2_train_rs, P_r2_test_rs), dim=1).cpu().numpy()

    eta_true_hist_numpy = eta_true_history.cpu().numpy()
    time_points = np.arange(Nt)

    # --- プロット作成 ---
    fig, axes = plt.subplots(len(individual_ids_to_plot), 4, figsize=(7 * 4, 4 * len(individual_ids_to_plot)), sharex=True)
    fig.suptitle('Latent Variable Trajectories (Filtered on Train, Forecast on Test with 95% CI)', fontsize=20)

    for i, ind_id in enumerate(individual_ids_to_plot):
        mask_true_r1 = ~np.isnan(eta_true_hist_numpy[ind_id, :, 1])

        # --- 3因子モデルの潜在変数 (3列) ---
        for l in range(L1_state1):
            ax = axes[i, l]
            
            # RS-KFの軌道と信頼区間
            mean_rs = eta_r1_full_rs[ind_id, :, l]
            std_dev_rs = np.sqrt(P_r1_full_rs[ind_id, :, l, l])
            ax.plot(time_points, mean_rs, 'b-', label='RS-KF', lw=1.5)
            ax.fill_between(time_points, mean_rs - 1.96 * std_dev_rs, mean_rs + 1.96 * std_dev_rs, color='blue', alpha=0.15)
            
            # 3-Factorモデルの軌道 (BPS Input)
            ax.plot(time_points, latents_3fac_full[ind_id, :, l], 'g-', label='BPS Input (3-Fac)', lw=1.5)
            
            # 真値
            if np.any(mask_true_r1):
                ax.plot(time_points[mask_true_r1], eta_true_hist_numpy[ind_id, :, l][mask_true_r1], 'r--', label='True', lw=2)
                
            ax.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=1.5)
            ax.set_title(f'Individual #{ind_id} - $\eta_{l+1}$ (3-Factor)')
            if i == 0: ax.legend()
            if l == 0: ax.set_ylabel('Value')

        # --- 1因子モデルの潜在変数 (4列目) ---
        ax = axes[i, 3]
        
        # RS-KFの軌道と信頼区間 (eta_Wは4番目の潜在変数)
        mean_rs_w = eta_r2_full_rs[ind_id, :, 3]
        std_dev_rs_w = np.sqrt(P_r2_full_rs[ind_id, :, 3, 3])
        ax.plot(time_points, mean_rs_w, 'b-', label='RS-KF', lw=1.5)
        ax.fill_between(time_points, mean_rs_w - 1.96 * std_dev_rs_w, mean_rs_w + 1.96 * std_dev_rs_w, color='blue', alpha=0.15)
        ax.plot(time_points, latents_1fac_full[ind_id, :, 0], 'g-', label='BPS Input (1-Fac)', lw=1.5)

        # 真値
        mask_true_r2 = ~mask_true_r1
        if np.any(mask_true_r2):
            ax.plot(time_points[mask_true_r2], eta_true_hist_numpy[ind_id, :, 0][mask_true_r2], 'r--', label='True', lw=2)

        ax.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(f'Individual #{ind_id} - $\eta$ (1-Factor)')
        if i == 0: ax.legend()

    for l in range(4):
        axes[-1, l].set_xlabel('Time Point')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("eta_trajectories_forecast_with_ci.png")
    print("\nLatent variable forecast plot with C.I. saved to eta_trajectories_forecast_with_ci.png")
    plt.show()

    # ==============================================================================
    # ▼▼▼【ここから追加】▼▼▼
    # ----------------------------------------------------------------
    # Part 11: 集計レベルでのモデル性能比較プロット
    # ----------------------------------------------------------------
    print("\n--- 11. Generating Aggregate Model Comparison Plot ---")

    # --- プロット用データを準備 ---
    # 1. 実際の状態の割合を計算
    state1_proportion_actual = (actual_states == 1).mean(axis=0)
    # 2. BPSモデルの平均重みを計算
    avg_weights_bps = weights_full.mean(dim=0).cpu().numpy()
    # 3. RS-KFモデルの平均確率を計算
    avg_probs_rs = predicted_probs_rs.mean(axis=0)
    # 4. 時間軸を作成
    time_points = np.arange(Nt)

    # --- プロット作成 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle('Model Comparison: BPS Hybrid vs. Regime-Switching KF (Aggregate Performance)', fontsize=20)

    # --- BPSモデルのプロット (左側) ---
    ax1.plot(time_points, avg_weights_bps[:, 0], 'o-', color='royalblue', label='Avg. Weight for 3-Factor Model (State 1)', zorder=3)
    ax1.plot(time_points, avg_weights_bps[:, 1], 's-', color='firebrick', label='Avg. Weight for 1-Factor Model (State 2)', zorder=3)
    ax1.set_title('BPS Hybrid Personalized Model', fontsize=16)
    ax1.set_xlabel('Time Point', fontsize=12)
    ax1.set_ylabel('Estimated Model Weight', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_ylim(-0.05, 1.05)
    ax1.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2, label='Train-Test Split')

    # 右側のY軸（実際の割合）
    ax2 = ax1.twinx()
    ax2.bar(time_points, state1_proportion_actual, color='grey', alpha=0.2, label='Proportion in State 1 (Actual)')
    ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False) # 右軸のグリッドは非表示

    # --- RS-KFモデルのプロット (右側) ---
    ax3.plot(time_points, avg_probs_rs[:, 0], 'o-', color='royalblue', label='Avg. Prob. of Regime 1 (3-Factor)', zorder=3)
    ax3.plot(time_points, avg_probs_rs[:, 1], 's-', color='firebrick', label='Avg. Prob. of Regime 2 (1-Factor)', zorder=3)
    ax3.set_title('Regime-Switching KF Model', fontsize=16)
    ax3.set_xlabel('Time Point', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2, label='Train-Test Split')

    # 右側のY軸（実際の割合）
    ax4 = ax3.twinx()
    ax4.bar(time_points, state1_proportion_actual, color='grey', alpha=0.2, label='Proportion in State 1 (Actual)')
    ax4.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("model_comparison_plot.png")
    print("Aggregate model comparison plot saved to model_comparison_plot.png")
    plt.show()

    # ▲▲▲【追加ここまで】▲▲▲
    # ==============================================================================