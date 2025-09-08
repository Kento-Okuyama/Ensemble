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
from pyro.infer.autoguide import AutoDiagonalNormal

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
file_path = r"model_comparison_random_walk_y_hat.txt"
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

    print("Running in-sample evaluation mode (no train-test split).")

    # ----------------------------------------------------------------
    # Part 1: データ生成
    # ----------------------------------------------------------------
    print("\n--- 1. Generating Simulation Data ---")
    data_file = 'simulation_data_rw.pt'
    if os.path.exists(data_file):
        print(f"Loading pre-computed simulation data from '{data_file}'...")
        saved_data = torch.load(data_file, weights_only=False)
        Y_generated = saved_data['Y_generated'].to(device)
        actual_states = saved_data['actual_states']
        print("Data loading complete.")
    else:
        print("No pre-computed data file found. Generating new simulation data...")
        Y_generated = torch.zeros(N, Nt, O, device=device)
        actual_states = np.zeros((N, Nt))
        q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1); q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
        r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)
        for i in range(N):
            eta_history = torch.randn(L1_state1, 1, device=device); current_state = 1; has_switched = False
            for t in range(Nt):
                actual_states[i, t] = current_state
                if current_state == 1 and t > 0:
                    z = gamma_intercept + (eta_history[0]*gamma_task + eta_history[1]*gamma_goal + eta_history[2]*gamma_bond)
                    if random.random() < (1 / (1 + math.exp(-z))): current_state = 2
                if current_state == 1:
                    eta_t = (B1_true_state1 @ eta_history) + q_dist_s1.sample().reshape(L1_state1, 1); y_mean_t = Lambda1_true_state1 @ eta_t
                else: 
                    if not has_switched: eta_history = torch.tensor([eta_history.mean()], device=device).reshape(L1_state2, 1); has_switched = True
                    eta_t = (B1_true_state2 @ eta_history) + q_dist_s2.sample().reshape(L1_state2, 1); y_mean_t = Lambda1_true_state2 @ eta_t
                Y_generated[i, t, :] = (y_mean_t + r_dist.sample().reshape(O, 1)).squeeze()
                eta_history = eta_t
        print("Simulation data generated."); print(f"Saving simulation data to '{data_file}'...")
        data_to_save = {'Y_generated': Y_generated.cpu(), 'actual_states': actual_states}
        torch.save(data_to_save, data_file); print("Data saving complete.")

    # ----------------------------------------------------------------
    # Part 2: 共通の関数定義
    # ----------------------------------------------------------------
    print("\n--- 2. Defining Common Functions ---")
    def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
        N, Nt, O1 = Y.shape; L1 = B1.shape[0]
        eta_prev = eta1_i0_0.expand(N, -1, -1); P_prev = P_i0_0.expand(N, -1, -1)
        total_log_likelihood = 0.0
        for t in range(Nt):
            y_t = Y[:, t, :].unsqueeze(-1)
            eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
            P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
            v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
            F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
            dist = torch.distributions.MultivariateNormal(loc=torch.zeros(O1, device=device), covariance_matrix=F_t)
            total_log_likelihood += dist.log_prob(v_t.squeeze(-1)).sum()
            K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
            eta_updated = eta_pred + torch.bmm(K_t, v_t)
            I_KL = torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
            P_updated = torch.bmm(torch.bmm(I_KL, P_pred), I_KL.transpose(-1, -2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(-1, -2))
            eta_prev = eta_updated; P_prev = P_updated
        return total_log_likelihood

    def get_kalman_predictions_and_latents(Y, b0, B1, Lambda1, Q, R):
        N, Nt, O1 = Y.shape; L1 = B1.shape[0]
        y_pred_series = torch.zeros_like(Y); eta_series = torch.zeros(N, Nt, L1, device=device)
        P_series = torch.zeros(N, Nt, L1, L1, device=device)
        eta_prev = torch.zeros(N, L1, 1, device=device); P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
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
            eta_series[:, t, :] = eta_updated.squeeze(-1); P_series[:, t, :, :] = P_updated
            eta_prev = eta_updated; P_prev = P_updated
        return y_pred_series, eta_series, P_series
        
    def calculate_rmse(y_true, y_pred): return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
    
    def calculate_sens_spec(actual_states, predicted_states_binary):
        ground_truth_binary = (actual_states == 2).astype(int)
        TP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 1)); FN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 1))
        TN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 0)); FP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 0))
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0; specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        return sensitivity, specificity

    # ----------------------------------------------------------------
    # Part 3: ベースラインモデルのパラメータ推定
    # ----------------------------------------------------------------
    print("\n--- 3. Estimation of Baseline Models ---")
    results_file_3fac = 'baseline_3fac_results.pt'
    results_file_1fac = 'baseline_1fac_results.pt'
    
    # --- 3a. 3-Factor Model ---
    if os.path.exists(results_file_3fac):
        print(f"\n--- 3a. Loading Pre-computed 3-Factor Model ---")
        saved_data = torch.load(results_file_3fac, weights_only=False)
        best_loss_3fac = saved_data['loss']
        best_params_3fac = saved_data['params']
        duration_3fac = saved_data['duration']
        print("Loading complete.")
    else:
        print("\n--- 3a. 3-Factor Model ---")
        start_time_3fac = time.time(); best_loss_3fac = float('inf'); best_params_3fac = {}
        b0_3fac = torch.randn(L1_state1, 1, requires_grad=True, device=device); b1_free_params_3fac = torch.randn(L1_state1, requires_grad=True, device=device)
        lambda1_free_params_3fac = torch.randn(6, requires_grad=True, device=device); log_q_diag_3fac = torch.zeros(L1_state1, requires_grad=True, device=device)
        log_r_diag_3fac = torch.zeros(O, requires_grad=True, device=device)
        params_to_learn_3fac = [b0_3fac, b1_free_params_3fac, lambda1_free_params_3fac, log_q_diag_3fac, log_r_diag_3fac]
        optimizer_3fac = torch.optim.AdamW(params_to_learn_3fac, lr=0.001, weight_decay=0.01)
        patience_counter = 0;
        for epoch in range(10000):
            optimizer_3fac.zero_grad()
            Q_est_3fac = torch.diag(torch.exp(log_q_diag_3fac)); R_est_3fac = torch.diag(torch.exp(log_r_diag_3fac))
            B1_3fac = torch.diag(b1_free_params_3fac); Lambda1_3fac = torch.zeros(O, L1_state1, device=device)
            Lambda1_3fac[0,0]=1; Lambda1_3fac[1,0]=lambda1_free_params_3fac[0]; Lambda1_3fac[2,0]=lambda1_free_params_3fac[1]; Lambda1_3fac[3,1]=1; Lambda1_3fac[4,1]=lambda1_free_params_3fac[2]
            Lambda1_3fac[5,1]=lambda1_free_params_3fac[3]; Lambda1_3fac[6,2]=1; Lambda1_3fac[7,2]=lambda1_free_params_3fac[4]; Lambda1_3fac[8,2]=lambda1_free_params_3fac[5]
            loss = -kalman_filter_torch_loss(Y_generated, b0_3fac, B1_3fac, Lambda1_3fac, Q_est_3fac, R_est_3fac, torch.zeros(L1_state1, 1, device=device), torch.eye(L1_state1, device=device) * 1e3)
            if torch.isnan(loss): print(f"  Run failed due to NaN loss."); break
            loss.backward(); optimizer_3fac.step()
            if (epoch + 1) % 100 == 0: print(f"  [Epoch {epoch + 1:04d}] loss: {loss.item():.4f} (Best: {best_loss_3fac:.4f})")
            if loss.item() < best_loss_3fac: best_loss_3fac = loss.item(); patience_counter = 0; best_params_3fac = {'b0': b0_3fac.detach(), 'B1': B1_3fac.detach(), 'Lambda1': Lambda1_3fac.detach(), 'Q': Q_est_3fac.detach(), 'R': R_est_3fac.detach()}
            else: patience_counter += 1
            if patience_counter >= 100: print(f"    -> Early stopping triggered at epoch {epoch + 1}."); break
        duration_3fac = time.time() - start_time_3fac
        print(f"Saving 3-factor model results to '{results_file_3fac}'...")
        torch.save({'loss': best_loss_3fac, 'params': best_params_3fac, 'duration': duration_3fac}, results_file_3fac)
        print("Saving complete.")

    # --- 3b. 1-Factor Model ---
    if os.path.exists(results_file_1fac):
        print(f"\n--- 3b. Loading Pre-computed 1-Factor Model ---")
        saved_data = torch.load(results_file_1fac, weights_only=False)
        best_loss_1fac = saved_data['loss']
        best_params_1fac = saved_data['params']
        duration_1fac = saved_data['duration']
        print("Loading complete.")
    else:
        print("\n--- 3b. 1-Factor Model ---")
        start_time_1fac = time.time(); best_loss_1fac = float('inf'); best_params_1fac = {}
        b0_1fac = torch.randn(L1_state2, 1, requires_grad=True, device=device); B1_1fac = torch.randn(L1_state2, L1_state2, requires_grad=True, device=device)
        lambda1_free_params_1fac = torch.randn(8, requires_grad=True, device=device); log_q_diag_1fac = torch.zeros(L1_state2, requires_grad=True, device=device)
        log_r_diag_1fac = torch.zeros(O, requires_grad=True, device=device)
        params_to_learn_1fac = [b0_1fac, B1_1fac, lambda1_free_params_1fac, log_q_diag_1fac, log_r_diag_1fac]
        optimizer_1fac = torch.optim.AdamW(params_to_learn_1fac, lr=0.001, weight_decay=0.01)
        patience_counter = 0
        for epoch in range(10000):
            optimizer_1fac.zero_grad()
            Q_est_1fac = torch.diag(torch.exp(log_q_diag_1fac)); R_est_1fac = torch.diag(torch.exp(log_r_diag_1fac))
            Lambda1_1fac = torch.zeros(O, L1_state2, device=device); Lambda1_1fac[0,0]=1.0; Lambda1_1fac[1:9,0] = lambda1_free_params_1fac[0:8]
            loss = -kalman_filter_torch_loss(Y_generated, b0_1fac, B1_1fac, Lambda1_1fac, Q_est_1fac, R_est_1fac, torch.zeros(L1_state2, 1, device=device), torch.eye(L1_state2, device=device) * 1e3)
            if torch.isnan(loss): print(f"  Run failed due to NaN loss."); break
            loss.backward(); optimizer_1fac.step()
            if (epoch + 1) % 100 == 0: print(f"  [Epoch {epoch + 1:04d}] loss: {loss.item():.4f} (Best: {best_loss_1fac:.4f})")
            if loss.item() < best_loss_1fac: best_loss_1fac = loss.item(); patience_counter = 0; best_params_1fac = {'b0': b0_1fac.detach(), 'B1': B1_1fac.detach(), 'Lambda1': Lambda1_1fac.detach(), 'Q': Q_est_1fac.detach(), 'R': R_est_1fac.detach()}
            else: patience_counter += 1
            if patience_counter >= 100: print(f"    -> Early stopping triggered at epoch {epoch + 1}."); break
        duration_1fac = time.time() - start_time_1fac
        print(f"Saving 1-factor model results to '{results_file_1fac}'...")
        torch.save({'loss': best_loss_1fac, 'params': best_params_1fac, 'duration': duration_1fac}, results_file_1fac)
        print("Saving complete.")

    # ----------------------------------------------------------------
    # Part 4: 新しいBPSモデル（ランダムウォーク）の定義と学習
    # ----------------------------------------------------------------
    print("\n\n--- 4. Defining and Training BPS Random-Walk Model ---")
    results_file_bps = 'bps_rw_results.pt'
    
    print("Pre-calculating inputs for BPS model (using predicted y_hat)...")
    with torch.no_grad():
        preds_3fac_est, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
        preds_1fac_est, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_1fac)
        factors_f = torch.stack((preds_3fac_est, preds_1fac_est), dim=3)
    print("Pre-calculation complete.")

    def bps_random_walk_model(y_obs, factors_f):
        N, Nt, O, J = factors_f.shape
        log_v_diag = pyro.sample("log_v_diag", dist.Normal(-2.0, 1.0).expand([J]).to_event(1))
        V = torch.diag(torch.exp(log_v_diag))
        log_tau = pyro.sample("log_tau", dist.Normal(-2.0, 1.0))
        tau = torch.exp(log_tau)
        log_sigma_diag = pyro.sample("log_sigma_diag", dist.Normal(0.0, 1.0).expand([O]).to_event(1))
        beta_prev = pyro.sample("beta_0", dist.Normal(0., 1.).expand([J]).to_event(1))
        alpha_prev = pyro.sample("alpha_0", dist.Normal(0., 1.).expand([O]).to_event(1))
        for t in range(Nt):
            beta_t = pyro.sample(f"beta_{t}", dist.MultivariateNormal(beta_prev, V))
            alpha_t = pyro.sample(f"alpha_{t}", dist.MultivariateNormal(alpha_prev, torch.eye(O, device=device) * tau**2))
            factors_t = factors_f[:, t, :, :]
            y_mean = alpha_t.unsqueeze(0) + torch.einsum('noj,j->no', factors_t, beta_t)
            with pyro.plate("individuals", N):
                pyro.sample(f"obs_{t}", dist.Normal(y_mean, torch.exp(log_sigma_diag)).to_event(1), obs=y_obs[:, t, :])
            beta_prev = beta_t; alpha_prev = alpha_t
    
    guide = AutoDiagonalNormal(bps_random_walk_model) # Guideはifブロックの外で定義
    
    if os.path.exists(results_file_bps):
        print(f"Loading Pre-computed BPS Model results from '{results_file_bps}'...")
        saved_data = torch.load(results_file_bps, weights_only=False)
        final_bps_loss = saved_data['loss']
        duration_bps = saved_data['duration']
        pyro.clear_param_store()
        pyro.get_param_store().set_state(saved_data['param_store_state'])
        print("Loading complete.")
    else:
        start_time_bps = time.time()
        pyro.clear_param_store()
        optimizer = Adam({"lr": 0.005})
        svi = SVI(bps_random_walk_model, guide, optimizer, loss=Trace_ELBO())
        n_steps = 5000; patience_bps = 200; patience_counter_bps = 0; best_loss_bps = float('inf')
        print("Starting BPS model training with VI...")
        for step in range(n_steps):
            loss = svi.step(Y_generated, factors_f) / N
            if torch.isnan(torch.tensor(loss)): print("Loss became NaN. Stopping training."); break
            if loss < best_loss_bps:
                best_loss_bps = loss; patience_counter_bps = 0
            else:
                patience_counter_bps += 1
            if (step + 1) % 100 == 0: print(f" [SVI step {step+1:04d}] ELBO loss: {loss:.4f} (Best: {best_loss_bps:.4f})")
            if patience_counter_bps >= patience_bps: print(f"    -> Early stopping triggered at step {step + 1}."); break
        duration_bps = time.time() - start_time_bps
        final_bps_loss = best_loss_bps
        print(f"BPS VI training finished. Duration: {duration_bps:.2f}s")
        print(f"Saving BPS model results to '{results_file_bps}'...")
        param_store_state = pyro.get_param_store().get_state()
        torch.save({'loss': final_bps_loss, 'duration': duration_bps, 'param_store_state': param_store_state}, results_file_bps)
        print("Saving complete.")

    # ----------------------------------------------------------------
    # Part 5: 全モデルの評価と可視化
    # ----------------------------------------------------------------
    print("\n\n--- 5. In-Sample Model Evaluation and Visualization ---")
    
    # --- ベースラインモデルの評価 ---
    preds_3fac, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
    rmse_3fac = calculate_rmse(Y_generated, preds_3fac)
    preds_1fac, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_1fac)
    rmse_1fac = calculate_rmse(Y_generated, preds_1fac)
    
    # --- BPSモデルの評価 ---
    median_params = guide.median()
    estimated_betas = torch.stack([median_params[f'beta_{t}'] for t in range(Nt)]).cpu().numpy()
    estimated_alphas = torch.stack([median_params[f'alpha_{t}'] for t in range(Nt)]).cpu().numpy()
    y_preds_bps = np.zeros_like(Y_generated.cpu().numpy())
    factors_f_numpy = factors_f.cpu().numpy()
    y_preds_bps = estimated_alphas[np.newaxis, :, :] + np.einsum('ntoj,tj->nto', factors_f_numpy, estimated_betas)
    rmse_bps = calculate_rmse(Y_generated, torch.tensor(y_preds_bps, device=device))
    predicted_states_bps_agg = (estimated_betas[:, 1] > estimated_betas[:, 0]).astype(int) + 1 # 状態1 or 2
    predicted_states_bps = np.tile(predicted_states_bps_agg, (N, 1))
    sens_bps, spec_bps = calculate_sens_spec(actual_states, predicted_states_bps)
    
    # --- 比較表の表示 ---
    print("\n--- Final Model In-Sample Fit Comparison ---")
    print("="*100)
    print(f"{'Model':<30} | {'Loss':<15} | {'In-Sample RMSE':<20} | {'Sensitivity':<15} | {'Specificity':<15}")
    print("-"*100)
    print(f"{'3-Factor Model':<30} | {best_loss_3fac:<15.4f} | {rmse_3fac:<20.4f} | {'N/A':<15} | {'N/A':<15}")
    print(f"{'1-Factor Model':<30} | {best_loss_1fac:<15.4f} | {rmse_1fac:<20.4f} | {'N/A':<15} | {'N/A':<15}")
    print(f"{'BPS Random-Walk Model':<30} | {final_bps_loss:<15.4f} | {rmse_bps:<20.4f} | {sens_bps:<15.4f} | {spec_bps:<15.4f}")
    print("="*100)
    
    # --- 可視化 ---
    print("\n--- Visualizing BPS Random-Walk Weights ---")
    time_points = np.arange(Nt)
    state1_proportion = (actual_states == 1).mean(axis=0)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time_points, estimated_betas[:, 0], 'o-', color='royalblue', label='Weight for 3-Factor Model (beta_1)')
    ax1.plot(time_points, estimated_betas[:, 1], 's-', color='firebrick', label='Weight for 1-Factor Model (beta_2)')
    ax1.axhline(0.5, color='grey', linestyle='--', label='Equal Weight (0.5)') # 0.5の基準線
    ax1.set_xlabel('Time Point', fontsize=12)
    ax1.set_ylabel('Beta Weight Value', fontsize=12)
    ax1.set_title('Estimated Beta Weights Over Time', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.bar(time_points, state1_proportion, color='grey', alpha=0.3, label='Proportion in State 1 (Actual)')
    ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("bps_random_walk_y_hat_betas.png")
    plt.show()