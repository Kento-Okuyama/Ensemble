import math
import random
import sys

# 可視化
import matplotlib.pyplot as plt
import numpy as np

# Pyro (確率的プログラミングライブラリ)
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal


# Teeクラス (コンソールとファイルへの同時出力)
class Tee:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

# Teeクラスを使って全体の処理を囲む
file_path = r"sim_wai_3to1_with_prior_and_comparison.txt"
with Tee(file_path):
    # ----------------------------------------------------------------
    # Part 0: モデルパラメータの定義
    # ----------------------------------------------------------------
    print("--- 0. Defining Parameters for the 3-to-1 Factor Regime-Switching Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N = 100; Nt = 20; O = 9; J = 2 # J: モデルの数

    L1_state1 = 3
    B1_true_state1 = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.5]], dtype=torch.float32, device=device)
    Lambda1_state1_matrix = np.zeros((O, L1_state1), dtype=np.float32)
    Lambda1_state1_matrix[0:3, 0] = [1.0, 0.8, 0.7]; Lambda1_state1_matrix[3:6, 1] = [1.0, 1.2, 0.8]; Lambda1_state1_matrix[6:9, 2] = [1.0, 0.9, 0.7]
    Lambda1_true_state1 = torch.tensor(Lambda1_state1_matrix, device=device)

    L1_state2 = 1
    B1_true_state2 = torch.tensor([[0.6]], dtype=torch.float32, device=device)
    Lambda1_state2_matrix = np.zeros((O, L1_state2), dtype=np.float32)
    Lambda1_state2_matrix[0:9, 0] = [0.9, 0.8, 0.7, 1.0, 1.1, 0.9, 0.9, 0.8, 0.7]
    Lambda1_true_state2 = torch.tensor(Lambda1_state2_matrix, device=device)

    Q_state1 = torch.eye(L1_state1, device=device); Q_state2 = torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept = -2.5; gamma_task = 0.1; gamma_goal = 0.1; gamma_bond = 0.1

    # ----------------------------------------------------------------
    # Part 1: データ生成
    # ----------------------------------------------------------------
    print("\n--- 1. Generating Simulation Data from Regime-Switching Model ---")
    Y_generated = torch.zeros(N, Nt, O, device=device)
    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)
    actual_states = np.zeros((N, Nt))
    for i in range(N):
        eta_history = torch.zeros(L1_state1, 1, device=device)
        current_state = 1
        for t in range(Nt):
            actual_states[i, t] = current_state
            if current_state == 1 and t > 0:
                task_prev, goal_prev, bond_prev = eta_history.squeeze().tolist()
                z = gamma_intercept + gamma_task * task_prev + gamma_goal * goal_prev + gamma_bond * bond_prev
                if random.random() < (1 / (1 + math.exp(-z))):
                    current_state = 2
                    eta_history = torch.tensor([sum(eta_history)/3], device=device).reshape(L1_state2, 1)
            if current_state == 1:
                eta_t = (B1_true_state1 @ eta_history) + q_dist_s1.sample().reshape(L1_state1, 1)
                y_mean_t = Lambda1_true_state1 @ eta_t
            else:
                eta_t = (B1_true_state2 @ eta_history) + q_dist_s2.sample().reshape(L1_state2, 1)
                y_mean_t = Lambda1_true_state2 @ eta_t
            Y_generated[i, t, :] = (y_mean_t + r_dist.sample().reshape(O, 1)).squeeze()
            eta_history = eta_t
    print("Simulation data generated.")

    # ----------------------------------------------------------------
    # Part 2: 共通の関数定義
    # ----------------------------------------------------------------
    print("\n--- 2. Defining Common Functions ---")
    def kalman_filter_torch_loss(Y1, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
        N, Nt, O1 = Y1.shape; L1 = B1.shape[0]
        total_log_likelihood = torch.tensor(0.0, device=Y1.device)
        I_mat = torch.eye(L1, device=Y1.device)
        for i in range(N):
            eta_prev = eta1_i0_0.clone(); P_prev = P_i0_0.clone()
            for t in range(Nt):
                y1_it = Y1[i, t, :].reshape(O1, 1)
                eta_pred = b0 + B1 @ eta_prev
                P_pred = B1 @ P_prev @ B1.T + Q
                v_it = y1_it - Lambda1 @ eta_pred
                F_it = Lambda1 @ P_pred @ Lambda1.T + R
                F_it_inv = torch.linalg.pinv(F_it)
                K_it = P_pred @ Lambda1.T @ F_it_inv
                eta_updated = eta_pred + K_it @ v_it
                term1 = I_mat - K_it @ Lambda1
                P_updated = term1 @ P_pred @ term1.T + K_it @ R @ K_it.T
                log_det_F_it = torch.logdet(F_it)
                exponent_term = -0.5 * v_it.T @ F_it_inv @ v_it
                total_log_likelihood += (-0.5 * O1 * math.log(2 * math.pi) - 0.5 * log_det_F_it + exponent_term).squeeze()
                eta_prev = eta_updated; P_prev = P_updated
        return total_log_likelihood

    def get_kalman_predictions(Y1, B1, Lambda1, Q, R):
        N, Nt, O1 = Y1.shape; L1 = B1.shape[0]
        y_pred_series = torch.zeros_like(Y1)
        b0 = torch.zeros(L1, 1, device=Y1.device)
        for i in range(N):
            eta_prev = torch.zeros(L1, 1, device=Y1.device)
            P_prev = torch.eye(L1, device=Y1.device) * 1e3
            for t in range(Nt):
                eta_pred = b0 + B1 @ eta_prev
                y_pred_series[i, t, :] = (Lambda1 @ eta_pred).squeeze()
                v_it = Y1[i, t, :].reshape(O1, 1) - (Lambda1 @ eta_pred)
                P_pred = B1 @ P_prev @ B1.T + Q
                F_it = Lambda1 @ P_pred @ Lambda1.T + R
                K_it = P_pred @ Lambda1.T @ torch.linalg.pinv(F_it)
                eta_prev = eta_pred + K_it @ v_it
                P_prev = (torch.eye(L1, device=Y1.device) - K_it @ Lambda1) @ P_pred
        return y_pred_series
        
    def calculate_rmse(y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()

    # ----------------------------------------------------------------
    # Part 3: 3因子モデルの評価（真値を使用）
    # ----------------------------------------------------------------
    print("\n\n--- 3. 3-Factor Model Evaluation (using True Parameters) ---")
    loss_3fac = -kalman_filter_torch_loss(Y_generated, torch.zeros(L1_state1, 1, device=device), B1_true_state1, Lambda1_true_state1, Q_state1, R_true,
        eta1_i0_0=torch.zeros(L1_state1, 1, device=device), P_i0_0=torch.eye(L1_state1, device=device) * 1e3).item()
    preds_3fac_true = get_kalman_predictions(Y_generated, B1_true_state1, Lambda1_true_state1, Q_state1, R_true)
    rmse_3fac = calculate_rmse(Y_generated, preds_3fac_true)
    print(f"3-Factor Model Loss: {loss_3fac:.4f}, RMSE: {rmse_3fac:.4f}")

    # ----------------------------------------------------------------
    # Part 4: 1因子モデルの評価（真値を使用）
    # ----------------------------------------------------------------
    print("\n\n--- 4. 1-Factor Model Evaluation (using True Parameters) ---")
    loss_1fac = -kalman_filter_torch_loss(Y_generated, torch.zeros(L1_state2, 1, device=device), B1_true_state2, Lambda1_true_state2, Q_state2, R_true,
        eta1_i0_0=torch.zeros(L1_state2, 1, device=device), P_i0_0=torch.eye(L1_state2, device=device) * 1e3).item()
    preds_1fac_true = get_kalman_predictions(Y_generated, B1_true_state2, Lambda1_true_state2, Q_state2, R_true)
    rmse_1fac = calculate_rmse(Y_generated, preds_1fac_true)
    print(f"1-Factor Model Loss: {loss_1fac:.4f}, RMSE: {rmse_1fac:.4f}")

    # ----------------------------------------------------------------
    # Part 5: ★事前知識（Prior）の定義（修正版）★
    # ----------------------------------------------------------------
    print("\n--- 5. Defining the Prior for Weight Transition (75% switch assumption) ---")
    
    # 潜在空間での遷移の開始点と終了点を定義
    start_val = 2.5  # 初期時点での3因子モデルの強さ（潜在値）
    
    # 最終時点で重みが約 25% / 75% になるように潜在値の終着点を設定
    # z1 - z2 = log(0.25/0.75) => z1=-0.55, z2=0.55
    end_val = 0.55
    
    # 3因子モデルの重み(潜在)：高(start_val) -> 中(-end_val)
    time_vec_3fac = torch.linspace(start_val, -end_val, Nt, device=device)
    
    # 1因子モデルの重み(潜在)：低(-start_val) -> 高(end_val)
    time_vec_1fac = torch.linspace(-start_val, end_val, Nt, device=device)

    # 形状: (Nt, J) = (20, 2)
    prior_loc_over_time = torch.stack([time_vec_3fac, time_vec_1fac], dim=1)
    print(f"Prior assumes transition to a 25% / 75% final state.")
    
    # ----------------------------------------------------------------
    # Part 6: PyroによるBPSモデルの実装 (事前知識を反映)
    # ----------------------------------------------------------------
    print("\n--- 6. Implementing and Training the BPS Model with Prior ---")
    
    def bps_model_with_prior(y_obs, y_pred_expert1, y_pred_expert2, prior_loc):
        N, Nt, O = y_obs.shape; J = 2
        with pyro.plate("individuals", N):
            beta_offset = pyro.sample("beta_offset", dist.Normal(0., 1.).expand([J]).to_event(1))
            tau_t = pyro.sample("tau_t", dist.LogNormal(0.0, 1.0))
            sigma = pyro.sample("sigma", dist.LogNormal(0.0, 1.0))
            for t in range(Nt):
                current_loc = prior_loc[t] + beta_offset
                beta_latent_t = pyro.sample(f"beta_latent_{t}", dist.Normal(current_loc, tau_t.unsqueeze(-1)).to_event(1))
                beta_t = torch.softmax(beta_latent_t, dim=-1)
                y_pred_mixed = beta_t[:, 0].unsqueeze(-1) * y_pred_expert1[:, t, :] + \
                               beta_t[:, 1].unsqueeze(-1) * y_pred_expert2[:, t, :]
                pyro.sample(f"obs_{t}", dist.Normal(y_pred_mixed, sigma.unsqueeze(-1)).to_event(1), obs=y_obs[:, t, :])

    def bps_guide_with_prior(y_obs, y_pred_expert1, y_pred_expert2, prior_loc):
        N, Nt, O = y_obs.shape; J = 2
        with pyro.plate("individuals", N):
            beta_offset_loc = pyro.param('beta_offset_loc', torch.zeros(N, J))
            beta_offset_scale = pyro.param('beta_offset_scale', torch.ones(N, J), constraint=dist.constraints.positive)
            pyro.sample("beta_offset", dist.Normal(beta_offset_loc, beta_offset_scale).to_event(1))
            tau_t_loc = pyro.param('tau_t_loc', torch.tensor(0.0))
            tau_t_scale = pyro.param('tau_t_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
            pyro.sample("tau_t", dist.LogNormal(tau_t_loc, tau_t_scale))
            sigma_loc = pyro.param('sigma_loc', torch.zeros(N))
            sigma_scale = pyro.param('sigma_scale', torch.ones(N), constraint=dist.constraints.positive)
            pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))
            for t in range(Nt):
                beta_latent_loc = pyro.param(f'beta_latent_loc_{t}', torch.zeros(N, J))
                beta_latent_scale = pyro.param(f'beta_latent_scale_{t}', torch.ones(N, J), constraint=dist.constraints.positive)
                pyro.sample(f"beta_latent_{t}", dist.Normal(beta_latent_loc, beta_latent_scale).to_event(1))

    pyro.clear_param_store()
    svi = SVI(bps_model_with_prior, bps_guide_with_prior, Adam({"lr": 0.01}), loss=Trace_ELBO())
    num_iterations = 2000
    final_bps_loss = 0.0
    print("Starting SVI training...")
    for j in range(num_iterations):
        loss = svi.step(Y_generated, preds_3fac_true, preds_1fac_true, prior_loc_over_time)
        if (j + 1) % 500 == 0:
            print(f"[iteration {j+1:04d}] loss: {loss:.4f}")
    final_bps_loss = loss
    print("SVI training finished.")

    # ----------------------------------------------------------------
    # Part 7: BPSモデルのRMSE計算と可視化
    # ----------------------------------------------------------------
    print("\n--- 7. BPS Model RMSE Calculation and Visualization ---")
    estimated_weights = np.zeros((Nt, 2))
    for t in range(Nt):
        beta_latent_mean = pyro.param(f"beta_latent_loc_{t}").detach().mean(dim=0).cpu().numpy()
        exp_beta = np.exp(beta_latent_mean); estimated_weights[t, :] = exp_beta / np.sum(exp_beta)
    
    weights_tensor = torch.tensor(estimated_weights, dtype=torch.float32, device=device)
    y_preds_bps = torch.zeros_like(Y_generated)
    for t in range(Nt):
        y_preds_bps[:, t, :] = weights_tensor[t, 0] * preds_3fac_true[:, t, :] + \
                               weights_tensor[t, 1] * preds_1fac_true[:, t, :]
    rmse_bps = calculate_rmse(Y_generated, y_preds_bps)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(range(Nt), estimated_weights[:, 0], 'o-', label='Weight for 3-Factor Model', color='royalblue', lw=2.5)
    ax1.plot(range(Nt), estimated_weights[:, 1], 's--', label='Weight for 1-Factor Model', color='firebrick', lw=2.5)
    prior_weights = torch.softmax(prior_loc_over_time, dim=-1).cpu().numpy()
    ax1.plot(range(Nt), prior_weights[:, 0], '-', label='Prior Trend (3-Factor)', color='royalblue', lw=1.5, alpha=0.5)
    ax1.plot(range(Nt), prior_weights[:, 1], '--', label='Prior Trend (1-Factor)', color='firebrick', lw=1.5, alpha=0.5)
    ax1.set_xlabel('Time Point', fontsize=14); ax1.set_ylabel('Estimated Model Weight', fontsize=14)
    ax1.set_ylim(0, 1); ax1.set_title('Dynamic Model Weighting with Prior', fontsize=16); ax1.legend(loc='best')
    ax2 = ax1.twinx()
    state1_proportion = (actual_states == 1).mean(axis=0)
    ax2.bar(range(Nt), state1_proportion, alpha=0.2, color='gray', label='Proportion in State 1 (Actual)')
    ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=14, color='gray'); ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1); ax2.legend(loc='upper right')
    fig.tight_layout(); plt.savefig("bps_weights_with_prior.png")
    print("\nResults plot saved to bps_weights_with_prior.png")

    # ----------------------------------------------------------------
    # Part 8: 最終的なモデル比較
    # ----------------------------------------------------------------
    print("\n\n--- 8. Final Model Fit Comparison ---")
    print("="*50)
    print(f"{'Model':<20} | {'Loss (Lower is Better)':<25} | {'RMSE (Lower is Better)':<25}")
    print("-"*50)
    
    if 'loss_3fac' in locals():
        print(f"{'3-Factor Model':<20} | {loss_3fac:<25.4f} | {rmse_3fac:<25.4f}")
    else:
        print("3-Factor Model evaluation was not performed.")
        
    if 'loss_1fac' in locals():
        print(f"{'1-Factor Model':<20} | {loss_1fac:<25.4f} | {rmse_1fac:<25.4f}")
    else:
        print("1-Factor Model evaluation was not performed.")

    if 'final_bps_loss' in locals() and final_bps_loss != 0.0:
        print(f"{'BPS Model (w/ Prior)':<20} | {final_bps_loss:<25.4f} | {rmse_bps:<25.4f}")
    else:
        print("BPS Model training failed or did not run.")
    print("="*50)
    print("\nNote: Loss for 3/1-Factor models is NegLogLikelihood. Loss for BPS is ELBO.")
    print("      These loss types are not directly comparable. RMSE is a directly comparable metric.")


plt.show()