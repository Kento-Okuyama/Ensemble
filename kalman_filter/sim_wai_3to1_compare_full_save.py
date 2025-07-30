import math
import random
import sys
import time
import os

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

    N = 100; Nt = 20; O = 9; J = 2
    L1_state1 = 3; L1_state2 = 1
    
    B1_true_state1 = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.5]], dtype=torch.float32, device=device)
    Lambda1_state1_matrix = np.zeros((O, L1_state1), dtype=np.float32)
    Lambda1_state1_matrix[0:3, 0]=[1.0,0.8,0.7]; Lambda1_state1_matrix[3:6, 1]=[1.0,1.2,0.8]; Lambda1_state1_matrix[6:9, 2]=[1.0,0.9,0.7]
    Lambda1_true_state1 = torch.tensor(Lambda1_state1_matrix, device=device)
    B1_true_state2 = torch.tensor([[0.6]], dtype=torch.float32, device=device)
    Lambda1_state2_matrix = np.zeros((O, L1_state2), dtype=np.float32)
    Lambda1_state2_matrix[0:9, 0] = [0.9, 0.8, 0.7, 1.0, 1.1, 0.9, 0.9, 0.8, 0.7]
    Lambda1_true_state2 = torch.tensor(Lambda1_state2_matrix, device=device)
    Q_state1=torch.eye(L1_state1, device=device); Q_state2=torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept=-2.5; gamma_task=0.1; gamma_goal=0.1; gamma_bond=0.1

    # ----------------------------------------------------------------
    # Part 1: データ生成
    # ----------------------------------------------------------------
    print("\n--- 1. Generating Simulation Data from Regime-Switching Model ---")
    Y_generated = torch.zeros(N, Nt, O, device=device)
    actual_states = np.zeros((N, Nt))
    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)
    for i in range(N):
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
            eta_history = eta_t
    print("Simulation data generated.")

    # ----------------------------------------------------------------
    # Part 2: 共通の関数定義
    # ----------------------------------------------------------------
    print("\n--- 2. Defining Common Functions ---")
    
    def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
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
            total_log_likelihood += log_likelihood_t.sum()
            
            eta_prev = eta_updated
            P_prev = P_updated
            
        return total_log_likelihood

    def get_kalman_predictions_and_latents(Y, B1, Lambda1, Q, R):
        N, Nt, O1 = Y.shape; L1 = B1.shape[0]
        y_pred_series = torch.zeros_like(Y); eta_series = torch.zeros(N, Nt, L1, device=device)
        b0 = torch.zeros(L1, 1, device=device)
        
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
            eta_prev = eta_updated
            P_prev = P_updated
            
        return y_pred_series, eta_series
        
    def calculate_rmse(y_true, y_pred): return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
    
    def calculate_sens_spec(actual_states, predicted_states_binary):
        ground_truth_binary = (actual_states == 2).astype(int)
        TP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 1)); FN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 1))
        TN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 0)); FP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 0))
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0; specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        return sensitivity, specificity

# ----------------------------------------------------------------
# Part 3: ベースラインモデルのパラメータ推定と評価
# ----------------------------------------------------------------
print("\n--- 3. Estimation and Evaluation of Baseline Models ---")
num_runs = 3; learning_rate = 0.001; num_epochs = 10000; l2_decay = 0.01

results_file_3fac = 'baseline_3fac_results.pt'
results_file_1fac = 'baseline_1fac_results.pt'

# --- 3a. 3-Factor Model ---
print("\n--- 3a. 3-Factor Model ---")
if os.path.exists(results_file_3fac):
    print(f"Loading pre-computed 3-factor model results from '{results_file_3fac}'...")
    saved_data = torch.load(results_file_3fac, weights_only=False)
    best_loss_3fac = saved_data['best_loss']; best_params_3fac = saved_data['best_params']
    rmse_3fac = saved_data['rmse']; sens_3fac = saved_data['sens']; spec_3fac = saved_data['spec']
    duration_3fac = 0.0; failed_runs_3fac = 0
    print("Loading complete.")
else:
    print(f"No pre-computed file found. Running 3-factor model estimation...")
    start_time_3fac = time.time()
    best_loss_3fac = float('inf'); best_params_3fac = {}; failed_runs_3fac = 0
    Q_est_3fac = torch.eye(L1_state1, device=device); R_est_3fac = torch.eye(O, device=device)
    for run in range(num_runs):
        print(f"Starting Run {run+1}/{num_runs}...")
        b0 = torch.randn(L1_state1, 1, requires_grad=True, device=device); b1_free_params = torch.randn(L1_state1, requires_grad=True, device=device)
        lambda1_free_params = torch.randn(6, requires_grad=True, device=device)
        optimizer = torch.optim.AdamW([b0, b1_free_params, lambda1_free_params], lr=learning_rate, weight_decay=l2_decay)
        
        # === ▼▼▼ 3因子モデルの履歴記録を追加 ▼▼▼ ===
        history_3fac = {'b1_free_params': [], 'lambda1_free_params': []}
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            B1 = torch.diag(b1_free_params); Lambda1 = torch.zeros(O, L1_state1, device=device)
            Lambda1[0,0]=1; Lambda1[1,0]=lambda1_free_params[0]; Lambda1[2,0]=lambda1_free_params[1]
            Lambda1[3,1]=1; Lambda1[4,1]=lambda1_free_params[2]; Lambda1[5,1]=lambda1_free_params[3]
            Lambda1[6,2]=1; Lambda1[7,2]=lambda1_free_params[4]; Lambda1[8,2]=lambda1_free_params[5]
            loss = -kalman_filter_torch_loss(Y_generated, b0, B1, Lambda1, Q_est_3fac, R_est_3fac, torch.zeros(L1_state1, 1, device=device), torch.eye(L1_state1, device=device) * 1e3)
            if torch.isnan(loss): failed_runs_3fac += 1; print(f"  Run {run+1} failed due to NaN loss."); break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([b0, b1_free_params, lambda1_free_params], 1.0); optimizer.step()
            
            if (epoch + 1) % (num_epochs // 10) == 0:
                print(f"  [Run {run + 1}, Epoch {epoch + 1:04d}] loss: {loss.item():.4f}")
                with torch.no_grad():
                    history_3fac['b1_free_params'].append(b1_free_params.clone().cpu().numpy())
                    history_3fac['lambda1_free_params'].append(lambda1_free_params.clone().cpu().numpy())
        
        if not torch.isnan(loss) and loss.item() < best_loss_3fac:
            best_loss_3fac = loss.item()
            best_params_3fac = {'b0': b0.detach(), 'B1': B1.detach(), 'Lambda1': Lambda1.detach()}
            print(f"-> New best loss for 3-Factor Model: {best_loss_3fac:.4f}")

            # ベストスコア更新時に履歴を表示
            print("\n--- 3-Factor Model Parameter Update History (Best Run) ---")
            for name, history in history_3fac.items():
                print(f"\nParameter: {name}")
                for i, val in enumerate(history):
                    progress = (i + 1) * 10
                    print(f"  {progress:>3d}%: {np.round(val.flatten(), 4)}")
        # === ▲▲▲ 履歴記録ここまで ▲▲▲ ===

    duration_3fac = time.time() - start_time_3fac
    preds_3fac_est, _ = get_kalman_predictions_and_latents(Y_generated, best_params_3fac['B1'], best_params_3fac['Lambda1'], Q_est_3fac, R_est_3fac)
    rmse_3fac = calculate_rmse(Y_generated, preds_3fac_est)
    sens_3fac, spec_3fac = calculate_sens_spec(actual_states, np.zeros_like(actual_states))
    print(f"Saving 3-factor model results to '{results_file_3fac}'...")
    results_to_save = {'best_loss': best_loss_3fac, 'best_params': best_params_3fac, 'rmse': rmse_3fac, 'sens': sens_3fac, 'spec': spec_3fac}
    torch.save(results_to_save, results_file_3fac)
    print("Saving complete.")

# --- 3b. 1-Factor Model ---
print("\n--- 3b. 1-Factor Model ---")
if os.path.exists(results_file_1fac):
    print(f"Loading pre-computed 1-factor model results from '{results_file_1fac}'...")
    saved_data = torch.load(results_file_1fac, weights_only=False)
    best_loss_1fac = saved_data['best_loss']; best_params_1fac = saved_data['best_params']
    rmse_1fac = saved_data['rmse']; sens_1fac = saved_data['sens']; spec_1fac = saved_data['spec']
    duration_1fac = 0.0; failed_runs_1fac = 0
    print("Loading complete.")
else:
    print(f"No pre-computed file found. Running 1-factor model estimation...")
    start_time_1fac = time.time()
    best_loss_1fac = float('inf'); best_params_1fac = {}; failed_runs_1fac = 0
    Q_est_1fac = torch.eye(L1_state2, device=device); R_est_1fac = torch.eye(O, device=device)
    for run in range(num_runs):
        print(f"Starting Run {run+1}/{num_runs}...")
        b0 = torch.randn(L1_state2, 1, requires_grad=True, device=device); B1 = torch.randn(L1_state2, L1_state2, requires_grad=True, device=device)
        lambda1_free_params = torch.randn(8, requires_grad=True, device=device)
        optimizer = torch.optim.AdamW([b0, B1, lambda1_free_params], lr=learning_rate, weight_decay=l2_decay)
        
        # === ▼▼▼ 1因子モデルの履歴記録を追加 ▼▼▼ ===
        history_1fac = {'B1': [], 'lambda1_free_params': []}
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            Lambda1 = torch.zeros(O, L1_state2, device=device)
            Lambda1[0,0]=1.0; Lambda1[1:9,0] = lambda1_free_params[0:8]
            loss = -kalman_filter_torch_loss(Y_generated, b0, B1, Lambda1, Q_est_1fac, R_est_1fac, torch.zeros(L1_state2, 1, device=device), torch.eye(L1_state2, device=device) * 1e3)
            if torch.isnan(loss): failed_runs_1fac += 1; print(f"  Run {run+1} failed due to NaN loss."); break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([b0, B1, lambda1_free_params], 1.0); optimizer.step()
            
            if (epoch + 1) % (num_epochs // 10) == 0:
                print(f"  [Run {run + 1}, Epoch {epoch + 1:04d}] loss: {loss.item():.4f}")
                with torch.no_grad():
                    history_1fac['B1'].append(B1.clone().cpu().numpy())
                    history_1fac['lambda1_free_params'].append(lambda1_free_params.clone().cpu().numpy())

        if not torch.isnan(loss) and loss.item() < best_loss_1fac:
            best_loss_1fac = loss.item()
            best_params_1fac = {'b0': b0.detach(), 'B1': B1.detach(), 'Lambda1': Lambda1.detach()}
            print(f"-> New best loss for 1-Factor Model: {best_loss_1fac:.4f}")
            
            print("\n--- 1-Factor Model Parameter Update History (Best Run) ---")
            for name, history in history_1fac.items():
                print(f"\nParameter: {name}")
                for i, val in enumerate(history):
                    progress = (i + 1) * 10
                    print(f"  {progress:>3d}%: {np.round(val.flatten(), 4)}")
        # === ▲▲▲ 履歴記録ここまで ▲▲▲ ===

    duration_1fac = time.time() - start_time_1fac
    preds_1fac_est, _ = get_kalman_predictions_and_latents(Y_generated, best_params_1fac['B1'], best_params_1fac['Lambda1'], Q_est_1fac, R_est_1fac)
    rmse_1fac = calculate_rmse(Y_generated, preds_1fac_est)
    sens_1fac, spec_1fac = calculate_sens_spec(actual_states, np.ones_like(actual_states))
    print(f"Saving 1-factor model results to '{results_file_1fac}'...")
    results_to_save = {'best_loss': best_loss_1fac, 'best_params': best_params_1fac, 'rmse': rmse_1fac, 'sens': sens_1fac, 'spec': spec_1fac}
    torch.save(results_to_save, results_file_1fac)
    print("Saving complete.")

# ----------------------------------------------------------------
# Part 4 & 5: BPSモデルの定義、学習、評価、可視化
# ----------------------------------------------------------------

print("\n--- 4a. Defining the Prior Trajectory for BPS Model ---")
start_val = 3.0
end_val = 0.8
time_vec_3fac = torch.linspace(start_val, -end_val, Nt, device=device)
time_vec_1fac = torch.linspace(-start_val, end_val, Nt, device=device)
prior_loc_over_time = torch.stack([time_vec_3fac, time_vec_1fac], dim=1)


def guided_rw_model(y_obs, y_pred_expert1, y_pred_expert2, latents_e1, latents_e2, prior_loc):
    N, Nt, O = y_obs.shape; J=2; L1=latents_e1.shape[2]; L2=latents_e2.shape[2]
    with pyro.plate("individuals", N) as ind:
        theta_3fac = pyro.sample("theta_3fac", dist.Normal(0., 1.).expand([L1]).to_event(1)); theta_1fac = pyro.sample("theta_1fac", dist.Normal(0., 1.).expand([L2]).to_event(1))
        theta_inter1 = pyro.sample("theta_inter1", dist.Normal(0., 1.).expand([L1, L2]).to_event(2)); theta_inter2 = pyro.sample("theta_inter2", dist.Normal(0., 1.).expand([L1, L2]).to_event(2))
        alpha_t0 = pyro.sample("alpha_t0", dist.Normal(0., 1.).expand([O]).to_event(1)); tau_alpha = pyro.sample("tau_alpha", dist.LogNormal(0.0, 1.0))
        sigma = pyro.sample("sigma", dist.LogNormal(0.0, 1.0)); beta_offset = pyro.sample("beta_offset", dist.Normal(0., 1.).expand([J]).to_event(1))
        tau_beta = pyro.sample("tau_beta", dist.LogNormal(0.0, 1.0)); alpha_t = alpha_t0
        for t in pyro.markov(range(Nt)):
            if t > 0:
                latents1_tm1 = latents_e1[ind, t-1, :]; latents2_tm1 = latents_e2[ind, t-1, :]
                drift_main_3fac = (latents1_tm1 * theta_3fac).sum(dim=-1); drift_main_1fac = (latents2_tm1 * theta_1fac).sum(dim=-1)
                interaction_features = latents1_tm1.unsqueeze(2) * latents2_tm1.unsqueeze(1)
                drift_inter_3fac = (interaction_features * theta_inter1).sum(dim=(1, 2)); drift_inter_1fac = (interaction_features * theta_inter2).sum(dim=(1, 2))
                data_drift = torch.stack([drift_main_3fac + drift_inter_3fac, drift_main_1fac + drift_inter_1fac], dim=-1)
            else: data_drift = 0.0
            loc = prior_loc[t] + beta_offset + data_drift
            beta_latent_t = pyro.sample(f"beta_latent_{t}", dist.Normal(loc, tau_beta.unsqueeze(-1)).to_event(1))
            alpha_t = pyro.sample(f"alpha_{t}", dist.Normal(alpha_t, tau_alpha.unsqueeze(-1)).to_event(1))
            beta_t = torch.softmax(beta_latent_t, dim=-1)
            y_pred_mixed = beta_t[:, 0].unsqueeze(-1) * y_pred_expert1[:, t, :] + beta_t[:, 1].unsqueeze(-1) * y_pred_expert2[:, t, :]
            y_mean = y_pred_mixed + alpha_t
            pyro.sample(f"obs_{t}", dist.Normal(y_mean, sigma.unsqueeze(-1)).to_event(1), obs=y_obs[:, t, :])

def guided_rw_guide(y_obs, y_pred_expert1, y_pred_expert2, latents_e1, latents_e2, prior_loc):
    N, Nt, O=y_obs.shape; J=2; L1=latents_e1.shape[2]; L2=latents_e2.shape[2]
    with pyro.plate("individuals", N):
        theta_3fac_loc = pyro.param('theta_3fac_loc', torch.zeros(N, L1)); theta_3fac_scale = pyro.param('theta_3fac_scale', torch.ones(N, L1), constraint=dist.constraints.positive); pyro.sample("theta_3fac", dist.Normal(theta_3fac_loc, theta_3fac_scale).to_event(1))
        theta_1fac_loc = pyro.param('theta_1fac_loc', torch.zeros(N, L2)); theta_1fac_scale = pyro.param('theta_1fac_scale', torch.ones(N, L2), constraint=dist.constraints.positive); pyro.sample("theta_1fac", dist.Normal(theta_1fac_loc, theta_1fac_scale).to_event(1))
        theta_inter1_loc = pyro.param('theta_inter1_loc', torch.zeros(N, L1, L2)); theta_inter1_scale = pyro.param('theta_inter1_scale', torch.ones(N, L1, L2), constraint=dist.constraints.positive); pyro.sample("theta_inter1", dist.Normal(theta_inter1_loc, theta_inter1_scale).to_event(2))
        theta_inter2_loc = pyro.param('theta_inter2_loc', torch.zeros(N, L1, L2)); theta_inter2_scale = pyro.param('theta_inter2_scale', torch.ones(N, L1, L2), constraint=dist.constraints.positive); pyro.sample("theta_inter2", dist.Normal(theta_inter2_loc, theta_inter2_scale).to_event(2))
        alpha_t0_loc = pyro.param('alpha_t0_loc', torch.zeros(N, O)); alpha_t0_scale = pyro.param('alpha_t0_scale', torch.ones(N, O), constraint=dist.constraints.positive); pyro.sample("alpha_t0", dist.Normal(alpha_t0_loc, alpha_t0_scale).to_event(1))
        tau_alpha_loc = pyro.param('tau_alpha_loc', torch.zeros(N)); tau_alpha_scale = pyro.param('tau_alpha_scale', torch.ones(N), constraint=dist.constraints.positive); pyro.sample("tau_alpha", dist.LogNormal(tau_alpha_loc, tau_alpha_scale))
        sigma_loc = pyro.param('sigma_loc', torch.zeros(N)); sigma_scale = pyro.param('sigma_scale', torch.ones(N), constraint=dist.constraints.positive); pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))
        beta_offset_loc = pyro.param('beta_offset_loc', torch.zeros(N, J)); beta_offset_scale = pyro.param('beta_offset_scale', torch.ones(N, J), constraint=dist.constraints.positive); pyro.sample("beta_offset", dist.Normal(beta_offset_loc, beta_offset_scale).to_event(1))
        tau_beta_loc = pyro.param('tau_beta_loc', torch.zeros(N)); tau_beta_scale = pyro.param('tau_beta_scale', torch.ones(N), constraint=dist.constraints.positive); pyro.sample("tau_beta", dist.LogNormal(tau_beta_loc, tau_beta_scale))
        for t in pyro.markov(range(Nt)):
            beta_latent_loc = pyro.param(f'beta_latent_loc_{t}', torch.zeros(N, J)); beta_latent_scale = pyro.param(f'beta_latent_scale_{t}', torch.ones(N, J), constraint=dist.constraints.positive); pyro.sample(f"beta_latent_{t}", dist.Normal(beta_latent_loc, beta_latent_scale).to_event(1))
            alpha_loc = pyro.param(f'alpha_loc_{t}', torch.zeros(N, O)); alpha_scale = pyro.param(f'alpha_scale_{t}', torch.ones(N, O), constraint=dist.constraints.positive); pyro.sample(f"alpha_{t}", dist.Normal(alpha_loc, alpha_scale).to_event(1))

bps_params_file = 'bps_params.pt'
bps_metrics_file = 'bps_metrics.pt'

if os.path.exists(bps_params_file) and os.path.exists(bps_metrics_file):
    print(f"\n\n--- 4b. Loading Pre-computed BPS Model Results ---")
    pyro.clear_param_store()
    print(f"Loading BPS parameters from '{bps_params_file}'..."); saved_params_state = torch.load(bps_params_file, weights_only=False); pyro.get_param_store().set_state(saved_params_state)
    print(f"Loading BPS metrics from '{bps_metrics_file}'..."); saved_metrics = torch.load(bps_metrics_file, weights_only=False)
    final_bps_loss = saved_metrics['loss']; rmse_bps = saved_metrics['rmse']; sens_bps = saved_metrics['sens']; spec_bps = saved_metrics['spec']
    duration_bps = 0.0; print("BPS Loading complete.")
else:
    print("\n\n--- 4b. Defining and Training the BPS Guided Random Walk Model ---")
    start_time_bps = time.time()
    preds_3fac_true, latents_3fac_true = get_kalman_predictions_and_latents(Y_generated, B1_true_state1, Lambda1_true_state1, Q_state1, R_true)
    preds_1fac_true, latents_1fac_true = get_kalman_predictions_and_latents(Y_generated, B1_true_state2, Lambda1_true_state2, Q_state2, R_true)
    
    pyro.clear_param_store()
    svi = SVI(guided_rw_model, guided_rw_guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
    num_iterations_bps = 10000

    # === ▼▼▼ BPSモデルの履歴記録を追加 ▼▼▼ ===
    param_history_bps = {
        'theta_3fac_loc': [], 'theta_1fac_loc': [], 'theta_inter1_loc': [], 'theta_inter2_loc': [],
        'beta_offset_loc': [], 'alpha_t0_loc': [], 'tau_alpha_loc': [], 'tau_beta_loc': []
    }
    
    print("Starting BPS model training and recording parameter history...")
    for j in range(num_iterations_bps):
        loss = svi.step(Y_generated, preds_3fac_true, preds_1fac_true, latents_3fac_true, latents_1fac_true, prior_loc_over_time)
        
        if (j + 1) % (num_iterations_bps // 10) == 0 or j == 0:
            print(f"[BPS iter {j+1:04d}] loss: {loss:.4f}")
            with torch.no_grad():
                for name in param_history_bps.keys():
                    param_value = pyro.param(name)
                    mean_param_value = param_value.mean(dim=0).cpu().numpy()
                    param_history_bps[name].append(mean_param_value)

    final_bps_loss = loss
    print("BPS model training finished.")
    duration_bps = time.time() - start_time_bps

    print("\n--- BPS Parameter Update History ---")
    for name, history in param_history_bps.items():
        print(f"\nParameter: {name}")
        for i, val in enumerate(history):
            iteration_step = 1 if i == 0 else (i) * (num_iterations_bps // 10)
            print(f"  Iter {iteration_step:>5d}: {np.round(val.flatten(), 5)}")
    # === ▲▲▲ 履歴記録ここまで ▲▲▲ ===

    individual_latent_weights = torch.zeros(N, Nt, J, device=device)
    for t in range(Nt): individual_latent_weights[:, t, :] = pyro.param(f'beta_latent_loc_{t}').detach()
    individual_weights = torch.softmax(individual_latent_weights, dim=-1)
    predicted_states_bps = (individual_weights[:, :, 1] > 0.5).cpu().numpy().astype(int)
    sens_bps, spec_bps = calculate_sens_spec(actual_states, predicted_states_bps)
    _estimated_weights_bps = individual_weights.mean(dim=0).cpu().numpy()
    weights_tensor_bps = torch.tensor(_estimated_weights_bps, dtype=torch.float32, device=device)
    y_preds_bps = torch.zeros_like(Y_generated)
    for t in range(Nt):
        alpha_t_mean = pyro.param(f'alpha_loc_{t}').detach().mean(dim=0)
        y_preds_bps[:, t, :] = weights_tensor_bps[t, 0] * preds_3fac_true[:, t, :] + weights_tensor_bps[t, 1] * preds_1fac_true[:, t, :] + alpha_t_mean
    rmse_bps = calculate_rmse(Y_generated, y_preds_bps)

    print(f"\nSaving BPS parameters to '{bps_params_file}'..."); pyro.get_param_store().save(bps_params_file)
    print(f"Saving BPS metrics to '{bps_metrics_file}'..."); metrics_to_save = {'loss': final_bps_loss, 'rmse': rmse_bps, 'sens': sens_bps, 'spec': spec_bps}; torch.save(metrics_to_save, bps_metrics_file)
    print("BPS Saving complete.")

print("\n--- 5. Evaluating BPS Model for Visualization ---")
individual_latent_weights = torch.zeros(N, Nt, J, device=device)
for t in range(Nt): individual_latent_weights[:, t, :] = pyro.param(f'beta_latent_loc_{t}').detach()
individual_weights = torch.softmax(individual_latent_weights, dim=-1)
estimated_weights_bps = individual_weights.mean(dim=0).cpu().numpy()

# ----------------------------------------------------------------
# Part 6: レジームスイッチング・カルマンフィルターモデルの定義と学習
# ----------------------------------------------------------------
rs_kf_results_file = 'rs_kf_results.pt'
print("\n\n--- 6. Defining and Training the Regime-Switching Kalman Filter Model ---")
class RegimeSwitchingKF(torch.nn.Module):
    def __init__(self, L1, L2, O, initial_belief='informative'):
        super().__init__(); self.initial_belief = initial_belief
        self.b1_r1 = torch.nn.Parameter(torch.randn(L1)); self.lambda_r1_free = torch.nn.Parameter(torch.randn(6)); self.q_r1_diag = torch.nn.Parameter(torch.rand(L1))
        self.b1_r2 = torch.nn.Parameter(torch.randn(L2, L2)); self.lambda_r2_free = torch.nn.Parameter(torch.randn(O - 1)); self.q_r2_diag = torch.nn.Parameter(torch.rand(L2))
        self.r_diag = torch.nn.Parameter(torch.rand(O)); initial_persistence_logit = 4.0
        self.gamma1 = torch.nn.Parameter(torch.tensor([initial_persistence_logit])); self.gamma2 = torch.nn.Parameter(torch.randn(L1))
        self.gamma3 = torch.nn.Parameter(torch.tensor([initial_persistence_logit])); self.gamma4 = torch.nn.Parameter(torch.randn(L2))
    def forward(self, y):
        N, Nt, O = y.shape; L1, L2 = self.b1_r1.shape[0], self.b1_r2.shape[0]
        B_r1 = torch.diag(self.b1_r1); Lambda_r1 = torch.zeros(O, L1, device=device); Lambda_r1[0,0]=1; Lambda_r1[1,0]=self.lambda_r1_free[0]; Lambda_r1[2,0]=self.lambda_r1_free[1]; Lambda_r1[3,1]=1; Lambda_r1[4,1]=self.lambda_r1_free[2]; Lambda_r1[5,1]=self.lambda_r1_free[3]; Lambda_r1[6,2]=1; Lambda_r1[7,2]=self.lambda_r1_free[4]; Lambda_r1[8,2]=self.lambda_r1_free[5]
        Q_r1 = torch.diag(self.q_r1_diag.abs() + 1e-4); B_r2 = self.b1_r2; Lambda_r2 = torch.zeros(O, L2, device=device); Lambda_r2[0,0]=1; Lambda_r2[1:,0]=self.lambda_r2_free
        Q_r2 = torch.diag(self.q_r2_diag.abs() + 1e-4); R = torch.diag(self.r_diag.abs() + 1e-4)
        eta_filtered_r1 = torch.zeros(N, L1, 1, device=device); P_filtered_r1 = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
        eta_filtered_r2 = torch.zeros(N, L2, 1, device=device); P_filtered_r2 = torch.eye(L2, device=device).expand(N, -1, -1) * 1e3
        m_prob_filtered = torch.zeros(N, 2, device=device)
        if self.initial_belief == 'informative': m_prob_filtered[:, 0] = 0.99; m_prob_filtered[:, 1] = 0.01
        else: m_prob_filtered[:, 0] = 0.5; m_prob_filtered[:, 1] = 0.5
        total_log_likelihood = 0.0
        self.filtered_probs = torch.zeros(N, Nt, 2, device=device); self.predicted_y = torch.zeros_like(y)
        for t in range(Nt):
            y_t = y[:, t, :].unsqueeze(-1)
            p11 = torch.sigmoid(self.gamma1 + (eta_filtered_r1.squeeze(-1) * self.gamma2).sum(-1)); p22 = torch.sigmoid(self.gamma3 + (eta_filtered_r2.squeeze(-1) * self.gamma4).sum(-1))
            transition_matrix = torch.stack([torch.stack([p11, 1-p11], dim=1), torch.stack([1-p22, p22], dim=1)], dim=1)
            m_prob_pred = (transition_matrix.transpose(1, 2) @ m_prob_filtered.unsqueeze(-1)).squeeze(-1)
            eta_pred_r1 = B_r1 @ eta_filtered_r1; P_pred_r1 = B_r1 @ P_filtered_r1 @ B_r1.T + Q_r1; v_r1 = y_t - Lambda_r1 @ eta_pred_r1; F_r1 = Lambda_r1 @ P_pred_r1 @ Lambda_r1.T + R
            dist_r1 = torch.distributions.MultivariateNormal(loc=y_t.squeeze(-1), covariance_matrix=F_r1); log_lik_r1 = dist_r1.log_prob((Lambda_r1 @ eta_pred_r1).squeeze(-1))
            eta_pred_r2 = B_r2 @ eta_filtered_r2; P_pred_r2 = B_r2 @ P_filtered_r2 @ B_r2.T + Q_r2; v_r2 = y_t - Lambda_r2 @ eta_pred_r2; F_r2 = Lambda_r2 @ P_pred_r2 @ Lambda_r2.T + R
            dist_r2 = torch.distributions.MultivariateNormal(loc=y_t.squeeze(-1), covariance_matrix=F_r2); log_lik_r2 = dist_r2.log_prob((Lambda_r2 @ eta_pred_r2).squeeze(-1))
            log_liks = torch.stack([log_lik_r1, log_lik_r2], dim=1); log_pred_probs = torch.log(m_prob_pred + 1e-9)
            log_joint_lik = log_liks + log_pred_probs; log_marginal_lik_t = torch.logsumexp(log_joint_lik, dim=1)
            total_log_likelihood += log_marginal_lik_t.sum()
            m_prob_filtered = torch.exp(log_joint_lik - log_marginal_lik_t.unsqueeze(-1)); self.filtered_probs[:, t, :] = m_prob_filtered
            y_pred_r1_t = Lambda_r1 @ eta_pred_r1; y_pred_r2_t = Lambda_r2 @ eta_pred_r2
            y_pred_t = m_prob_filtered[:, 0].unsqueeze(-1).unsqueeze(-1) * y_pred_r1_t + m_prob_filtered[:, 1].unsqueeze(-1).unsqueeze(-1) * y_pred_r2_t
            self.predicted_y[:, t, :] = y_pred_t.squeeze(-1)
            F_inv_r1 = torch.linalg.pinv(F_r1); K_r1 = P_pred_r1 @ Lambda_r1.T @ F_inv_r1; eta_filtered_r1 = eta_pred_r1 + K_r1 @ v_r1
            I_minus_KL_r1 = torch.eye(L1, device=device) - K_r1 @ Lambda_r1; P_filtered_r1 = I_minus_KL_r1 @ P_pred_r1 @ I_minus_KL_r1.transpose(-1, -2) + K_r1 @ R @ K_r1.transpose(-1, -2)
            F_inv_r2 = torch.linalg.pinv(F_r2); K_r2 = P_pred_r2 @ Lambda_r2.T @ F_inv_r2; eta_filtered_r2 = eta_pred_r2 + K_r2 @ v_r2
            I_minus_KL_r2 = torch.eye(L2, device=device) - K_r2 @ Lambda_r2; P_filtered_r2 = I_minus_KL_r2 @ P_pred_r2 @ I_minus_KL_r2.transpose(-1, -2) + K_r2 @ R @ K_r2.transpose(-1, -2)
        return -total_log_likelihood
num_runs_rs = 3
if os.path.exists(rs_kf_results_file):
    print(f"Loading pre-computed RS-KF model results from '{rs_kf_results_file}'...")
    saved_data = torch.load(rs_kf_results_file, weights_only=False)
    best_model_rs_state = saved_data['model_state_dict']
    final_loss_rs = saved_data['loss']; rmse_rs = saved_data['rmse']; sens_rs = saved_data['sens']; spec_rs = saved_data['spec']
    duration_rs = 0.0; failed_runs_rs = 0; print("RS-KF Loading complete.")
else:
    start_time_rs = time.time()
    best_loss_rs = float('inf'); best_model_rs_state = None; failed_runs_rs = 0
    num_epochs_rs = 10000

    # === ▼▼▼ RS-KFモデルの履歴記録を追加 ▼▼▼ ===
    history_rs = {'gamma1': [], 'gamma2': [], 'gamma3': [], 'gamma4': []}
    
    for run in range(num_runs_rs):
        print(f"Starting Run {run+1}/{num_runs_rs}...")
        model_rs = RegimeSwitchingKF(L1_state1, L1_state2, O, initial_belief=SIMULATION_MODE).to(device)
        optimizer_rs = torch.optim.Adam(model_rs.parameters(), lr=1e-3)

        # 各runの履歴を一時的に保存
        run_history = {name: [] for name in history_rs.keys()}

        for epoch in range(num_epochs_rs):
            optimizer_rs.zero_grad(); loss = model_rs(Y_generated)
            if torch.isnan(loss): failed_runs_rs += 1; print(f"Run {run+1} failed due to NaN loss."); break
            loss.backward(); torch.nn.utils.clip_grad_norm_(model_rs.parameters(), 1.0); optimizer_rs.step()
            if (epoch + 1) % (num_epochs_rs // 10) == 0:
                print(f"   [RS-KF Run {run + 1}, Epoch {epoch + 1:04d}] loss: {loss.item():.4f}")
                with torch.no_grad():
                    for name in run_history.keys():
                        param_value = getattr(model_rs, name)
                        run_history[name].append(param_value.clone().cpu().numpy())

        if not torch.isnan(loss) and loss.item() < best_loss_rs:
            best_loss_rs = loss.item(); best_model_rs_state = model_rs.state_dict()
            history_rs = run_history # ベストスコアのrunの履歴を保持
            print(f"-> New best loss for RS-KF Model: {best_loss_rs:.4f}")
    
    duration_rs = time.time() - start_time_rs
    
    print("\n--- RS-KF Parameter Update History (Best Run) ---")
    for name, history in history_rs.items():
        print(f"\nParameter: {name}")
        for i, val in enumerate(history):
            progress = (i + 1) * 10
            print(f"  {progress:>3d}%: {np.round(val.flatten(), 4)}")
    # === ▲▲▲ 履歴記録ここまで ▲▲▲ ===
    
    _final_model_rs = RegimeSwitchingKF(L1_state1, L1_state2, O, initial_belief=SIMULATION_MODE).to(device)
    if best_model_rs_state: _final_model_rs.load_state_dict(best_model_rs_state)
    with torch.no_grad():
        final_loss_rs = _final_model_rs(Y_generated).item()
        predicted_y_rs = _final_model_rs.predicted_y
        rmse_rs = calculate_rmse(Y_generated, predicted_y_rs)
        _final_model_rs(Y_generated) 
        _predicted_probs_rs = _final_model_rs.filtered_probs.cpu().numpy()
        _predicted_states_rs = (_predicted_probs_rs[:, :, 1] > 0.5).astype(int)
    sens_rs, spec_rs = calculate_sens_spec(actual_states, _predicted_states_rs)
    print("RS-KF model training finished.")
    print(f"Saving RS-KF model results to '{rs_kf_results_file}'...")
    results_to_save = {'model_state_dict': best_model_rs_state, 'loss': final_loss_rs, 'rmse': rmse_rs, 'sens': sens_rs, 'spec': spec_rs}
    torch.save(results_to_save, rs_kf_results_file)
    print("RS-KF Saving complete.")

# --- RS-KF評価 (プロット用変数の計算) ---
print("\n--- 7. Evaluating RS-KF Model for Visualization ---")
final_model_rs = RegimeSwitchingKF(L1_state1, L1_state2, O, initial_belief=SIMULATION_MODE).to(device)
if best_model_rs_state:
    final_model_rs.load_state_dict(best_model_rs_state)
with torch.no_grad():
    final_model_rs(Y_generated)
    predicted_probs_rs = final_model_rs.filtered_probs.cpu().numpy()
    
# ----------------------------------------------------------------
# Part 7: 最終的なモデル比較
# ----------------------------------------------------------------
print("\n\n--- 6. Final Model Fit Comparison ---")
print("="*115)
print(f"{'Model':<30} | {'Loss':<15} | {'RMSE':<15} | {'Sensitivity':<15} | {'Specificity':<15} | {'Time (s)':<25}")
print("-"*115)
time_str_3fac = f"{duration_3fac:.2f} (Failed: {failed_runs_3fac}/{num_runs})"
print(f"{'3-Factor Model':<30} | {best_loss_3fac:<15.4f} | {rmse_3fac:<15.4f} | {sens_3fac:<15.4f} | {spec_3fac:<15.4f} | {time_str_3fac:<25}")
time_str_1fac = f"{duration_1fac:.2f} (Failed: {failed_runs_1fac}/{num_runs})"
print(f"{'1-Factor Model':<30} | {best_loss_1fac:<15.4f} | {rmse_1fac:<15.4f} | {sens_1fac:<15.4f} | {spec_1fac:<15.4f} | {time_str_1fac:<25}")
print(f"{'BPS Hybrid Personalized':<30} | {final_bps_loss:<15.4f} | {rmse_bps:<15.4f} | {sens_bps:<15.4f} | {spec_bps:<15.4f} | {duration_bps:<25.2f}")
time_str_rs = f"{duration_rs:.2f} (Failed: {failed_runs_rs}/{num_runs_rs})"
print(f"{'Regime-Switching KF':<30} | {final_loss_rs:<15.4f} | {rmse_rs:<15.4f} | {sens_rs:<15.4f} | {spec_rs:<15.4f} | {time_str_rs:<25}")
print("="*115)

# ----------------------------------------------------------------
# Part 8: レジームスイッチング・カルマンフィルターモデルのパラメータ可視化
# ----------------------------------------------------------------
print("\n\n--- 8. RS-KF Model Parameter Visualization ---")

# === ▼▼▼ ここから変更 ▼▼▼ ===
# 評価指標の再計算は行わず、Part 7で準備されたfinal_model_rsからパラメータを表示するだけにする

# RS-KFモデルのパラメータ推定値を表示
print("\nRS-KF Estimated Gamma Intercepts (p11, p22):")
# final_model_rsはPart 7で準備済みなので、そのまま使用できる
if best_model_rs_state:
    print(f"  gamma1: {final_model_rs.gamma1.item():.4f}, gamma3: {final_model_rs.gamma3.item():.4f}")
else:
    print("  Model was not trained or loaded, cannot display parameters.")

# === ▲▲▲ ここまで変更 ▲▲▲ ===

# ----------------------------------------------------------------
# Part 9: 最終的なモデル比較
# ----------------------------------------------------------------
print("\n\n--- 9. Final Model Fit Comparison ---")
print("="*95)
print(f"{'Model':<30} | {'Loss':<15} | {'RMSE':<15} | {'Sensitivity':<15} | {'Specificity':<15}")
print("-"*95)
print(f"{'3-Factor Model (Estimated)':<30} | {best_loss_3fac:<15.4f} | {rmse_3fac:<15.4f} | {sens_3fac:<15.4f} | {spec_3fac:<15.4f}")
print(f"{'1-Factor Model (Estimated)':<30} | {best_loss_1fac:<15.4f} | {rmse_1fac:<15.4f} | {sens_1fac:<15.4f} | {spec_1fac:<15.4f}")
print(f"{'BPS Hybrid Personalized':<30} | {final_bps_loss:<15.4f} | {rmse_bps:<15.4f} | {sens_bps:<15.4f} | {spec_bps:<15.4f}")
print(f"{'Regime-Switching KF':<30} | {final_loss_rs:<15.4f} | {rmse_rs:<15.4f} | {sens_rs:<15.4f} | {spec_rs:<15.4f}")
print("="*95)
print("\nNote: Loss for Baseline/RS-KF is NegLogLikelihood. Loss for BPS is ELBO.")
print("      RMSE, Sensitivity, and Specificity are directly comparable metrics.")

# ----------------------------------------------------------------
# Part 10: 比較グラフの描画
# ----------------------------------------------------------------
print("\n\n--- 10. Visualization of Final Results ---")

state1_proportion = (actual_states == 1).mean(axis=0)
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
fig.suptitle('Model Comparison: BPS Hybrid vs. Regime-Switching KF', fontsize=20)

# --- プロット1: BPS Hybrid Model ---
ax1.plot(range(Nt), estimated_weights_bps[:, 0], 'o-', label='Weight for 3-Factor Model', color='royalblue', lw=2.5, zorder=10)
ax1.plot(range(Nt), estimated_weights_bps[:, 1], 's--', label='Weight for 1-Factor Model', color='firebrick', lw=2.5, zorder=10)
ax1.set_xlabel('Time Point', fontsize=14); ax1.set_ylabel('Estimated Model Weight', fontsize=14)
ax1.set_title('BPS Hybrid Personalized Model', fontsize=16); ax1.legend(loc='best')
ax1.set_ylim(-0.05, 1.05)
ax1_twin = ax1.twinx()
ax1_twin.bar(range(Nt), state1_proportion, alpha=0.2, color='gray', label='Proportion in State 1 (Actual)')
ax1_twin.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=14, color='gray'); ax1_twin.tick_params(axis='y', labelcolor='gray')
ax1_twin.set_ylim(-0.05, 1.05); ax1_twin.legend(loc='upper right')

# --- プロット2: Regime-Switching KF Model ---
avg_prob_rs_r1 = predicted_probs_rs[:, :, 0].mean(axis=0)
avg_prob_rs_r2 = predicted_probs_rs[:, :, 1].mean(axis=0)
ax2.plot(range(Nt), avg_prob_rs_r1, 'o-', label='Avg. Prob. of Regime 1 (3-Factor)', color='royalblue', lw=2.5, zorder=10)
ax2.plot(range(Nt), avg_prob_rs_r2, 's--', label='Avg. Prob. of Regime 2 (1-Factor)', color='firebrick', lw=2.5, zorder=10)
ax2.set_xlabel('Time Point', fontsize=14); ax2.set_ylabel('Estimated Regime Probability', fontsize=14)
ax2.set_title('Regime-Switching KF Model', fontsize=16); ax2.legend(loc='best')
ax2_twin = ax2.twinx()
ax2_twin.bar(range(Nt), state1_proportion, alpha=0.2, color='gray', label='Proportion in State 1 (Actual)')
ax2_twin.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=14, color='gray'); ax2_twin.tick_params(axis='y', labelcolor='gray')
ax2_twin.set_ylim(-0.05, 1.05); ax2_twin.legend(loc='upper right')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("model_comparison_plot.png")
print("\nCombined results plot saved to model_comparison_plot.png")

plt.show()
