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
file_path = r"sim_wai_3to1_true_params_ver.txt"
with Tee(file_path):
    # ----------------------------------------------------------------
    # Part 0: モデルパラメータの定義
    # ----------------------------------------------------------------
    print("--- 0. Defining Parameters for the 3-to-1 Factor Regime-Switching Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N = 100; Nt = 20; O = 9

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
    switch_points = []; actual_states = np.zeros((N, Nt))
    for i in range(N):
        eta_history = torch.zeros(L1_state1, 1, device=device)
        current_state = 1
        has_switched = False
        for t in range(Nt):
            actual_states[i, t] = current_state
            if current_state == 1 and t > 0:
                task_prev, goal_prev, bond_prev = eta_history.squeeze().tolist()
                z = gamma_intercept + gamma_task * task_prev + gamma_goal * goal_prev + gamma_bond * bond_prev
                switch_prob = 1 / (1 + math.exp(-z))
                if random.random() < switch_prob:
                    current_state = 2
                    if not has_switched: switch_points.append(t); has_switched = True
                    wai_prev = (task_prev + goal_prev + bond_prev) / 3
                    eta_history = torch.tensor([wai_prev], device=device).reshape(L1_state2, 1)
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
        eta1_i0_0 = torch.zeros(L1, 1, device=Y1.device)
        P_i0_0 = torch.eye(L1, device=Y1.device) * 1e3
        b0 = torch.zeros(L1, 1, device=Y1.device)
        for i in range(N):
            eta_prev = eta1_i0_0.clone(); P_prev = P_i0_0.clone()
            for t in range(Nt):
                eta_pred = b0 + B1 @ eta_prev
                y_pred_series[i, t, :] = (Lambda1 @ eta_pred).squeeze()
                v_it = Y1[i, t, :].reshape(O1, 1) - (Lambda1 @ eta_pred)
                P_pred = B1 @ P_prev @ B1.T + Q
                F_it = Lambda1 @ P_pred @ Lambda1.T + R
                K_it = P_pred @ Lambda1.T @ torch.linalg.pinv(F_it)
                eta_updated = eta_pred + K_it @ v_it
                P_updated = (torch.eye(L1, device=Y1.device) - K_it @ Lambda1) @ P_pred
                eta_prev = eta_updated; P_prev = P_updated
        return y_pred_series

    def calculate_rmse(y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()

    # ----------------------------------------------------------------
    # Part 3: 3因子モデルの評価（真値を使用）
    # ----------------------------------------------------------------
    print("\n\n--- 3. 3-Factor Model Evaluation (using True Parameters) ---")
    # 真値パラメータを使って損失（負の対数尤度）を計算
    loss_3fac = -kalman_filter_torch_loss(Y_generated, torch.zeros(L1_state1, 1, device=device), B1_true_state1, Lambda1_true_state1, Q_state1, R_true,
        eta1_i0_0=torch.zeros(L1_state1, 1, device=device), P_i0_0=torch.eye(L1_state1, device=device) * 1e3).item()
    print(f"3-Factor Model Loss (NegLogLike) with true params: {loss_3fac:.4f}")

    # 真値パラメータを使って予測とRMSEを計算
    preds_3fac_true = get_kalman_predictions(Y_generated, B1_true_state1, Lambda1_true_state1, Q_state1, R_true)
    rmse_3fac = calculate_rmse(Y_generated, preds_3fac_true)
    print(f"3-Factor Model RMSE with true params: {rmse_3fac:.4f}")


    # ----------------------------------------------------------------
    # Part 4: 1因子モデルの評価（真値を使用）
    # ----------------------------------------------------------------
    print("\n\n--- 4. 1-Factor Model Evaluation (using True Parameters) ---")
    # 真値パラメータを使って損失（負の対数尤度）を計算
    loss_1fac = -kalman_filter_torch_loss(Y_generated, torch.zeros(L1_state2, 1, device=device), B1_true_state2, Lambda1_true_state2, Q_state2, R_true,
        eta1_i0_0=torch.zeros(L1_state2, 1, device=device), P_i0_0=torch.eye(L1_state2, device=device) * 1e3).item()
    print(f"1-Factor Model Loss (NegLogLike) with true params: {loss_1fac:.4f}")

    # 真値パラメータを使って予測とRMSEを計算
    preds_1fac_true = get_kalman_predictions(Y_generated, B1_true_state2, Lambda1_true_state2, Q_state2, R_true)
    rmse_1fac = calculate_rmse(Y_generated, preds_1fac_true)
    print(f"1-Factor Model RMSE with true params: {rmse_1fac:.4f}")

    # Part 5はPart 3, 4で計算済みのため不要

    # ----------------------------------------------------------------
    # Part 6: PyroによるBPSモデルの実装
    # ----------------------------------------------------------------
    print("\n--- 6. Implementing and Training the BPS Model with Pyro ---")
    Y_centered = Y_generated - Y_generated.mean(dim=(0, 1))

    def bps_model(y_obs, y_pred_expert1, y_pred_expert2):
        N, Nt, O = y_obs.shape; J = 2
        tau_b = pyro.sample("tau_b", dist.LogNormal(0.0, 1.0))
        with pyro.plate("individuals", N):
            beta_latent_t0 = pyro.sample("beta_latent_t0", dist.Normal(0., 1.).expand([N, J]).to_event(1))
            sigma = pyro.sample("sigma", dist.LogNormal(0.0, 1.0))
            beta_latent_t = beta_latent_t0
            for t in pyro.markov(range(Nt)):
                beta_latent_t = pyro.sample(f"beta_latent_{t}", dist.Normal(beta_latent_t, tau_b).to_event(1))
                beta_t = torch.softmax(beta_latent_t, dim=-1)
                y_pred_mixed = beta_t[:, 0].unsqueeze(-1) * y_pred_expert1[:, t, :] + \
                               beta_t[:, 1].unsqueeze(-1) * y_pred_expert2[:, t, :]
                pyro.sample(f"obs_{t}", dist.Normal(y_pred_mixed, sigma.unsqueeze(-1)).to_event(1), obs=y_obs[:, t, :])

    def bps_guide(y_obs, y_pred_expert1, y_pred_expert2):
        N, Nt, O = y_obs.shape; J = 2
        tau_b_loc = pyro.param('tau_b_loc', torch.tensor(0.0))
        tau_b_scale = pyro.param('tau_b_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
        pyro.sample("tau_b", dist.LogNormal(tau_b_loc, tau_b_scale))
        with pyro.plate("individuals", N):
            beta_latent_t0_loc = pyro.param('beta_latent_t0_loc', torch.zeros(N, J))
            beta_latent_t0_scale = pyro.param('beta_latent_t0_scale', torch.ones(N, J), constraint=dist.constraints.positive)
            pyro.sample("beta_latent_t0", dist.Normal(beta_latent_t0_loc, beta_latent_t0_scale).to_event(1))
            sigma_loc = pyro.param('sigma_loc', torch.zeros(N))
            sigma_scale = pyro.param('sigma_scale', torch.ones(N), constraint=dist.constraints.positive)
            pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))
            beta_latent_prev_loc = beta_latent_t0_loc
            for t in pyro.markov(range(Nt)):
                beta_latent_loc = pyro.param(f'beta_latent_loc_{t}', torch.zeros(N, J))
                beta_latent_scale = pyro.param(f'beta_latent_scale_{t}', torch.ones(N, J), constraint=dist.constraints.positive)
                pyro.sample(f"beta_latent_{t}", dist.Normal(beta_latent_loc, beta_latent_scale).to_event(1))
                beta_latent_prev_loc = beta_latent_loc

    pyro.clear_param_store()
    svi = SVI(bps_model, bps_guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
    num_iterations = 2000
    final_bps_loss = 0.0
    print("Starting SVI training...")
    for j in range(num_iterations):
        loss = svi.step(Y_centered, preds_3fac_true, preds_1fac_true)
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
    
    # BPSモデルの予測値を計算
    weights_tensor = torch.tensor(estimated_weights, dtype=torch.float32, device=device)
    y_preds_bps = torch.zeros_like(Y_generated)
    for t in range(Nt):
        y_preds_bps[:, t, :] = weights_tensor[t, 0] * preds_3fac_true[:, t, :] + \
                               weights_tensor[t, 1] * preds_1fac_true[:, t, :]
    rmse_bps = calculate_rmse(Y_generated, y_preds_bps)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(range(Nt), estimated_weights[:, 0], 'o-', label='BPS Weight for 3-Factor Model', color='royalblue', lw=2.5)
    ax1.plot(range(Nt), estimated_weights[:, 1], 's--', label='BPS Weight for 1-Factor Model', color='firebrick', lw=2.5)
    ax1.set_xlabel('Time Point', fontsize=14); ax1.set_ylabel('Estimated Model Weight', fontsize=14)
    ax1.set_ylim(0, 1); ax1.set_title('Dynamic Model Weighting by BPS Model', fontsize=16); ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    state1_proportion = (actual_states == 1).mean(axis=0)
    ax2.bar(range(Nt), state1_proportion, alpha=0.2, color='gray', label='Proportion of Individuals in State 1 (Actual)')
    ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=14, color='gray'); ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1); ax2.legend(loc='upper right')
    fig.tight_layout(); plt.savefig("bps_weights_over_time_true_params.png")
    print("\nResults plot saved to bps_weights_over_time_true_params.png")

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
        print(f"{'BPS Model':<20} | {final_bps_loss:<25.4f} | {rmse_bps:<25.4f}")
    else:
        print("BPS Model training failed or did not run.")
    print("="*50)
    print("\nNote: Loss for 3/1-Factor models is NegLogLikelihood. Loss for BPS is ELBO.")
    print("      These loss types are not directly comparable. RMSE is a directly comparable metric.")


plt.show()