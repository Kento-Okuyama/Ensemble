# BPS/bps.py (MODIFIED)

import math
import os
import sys
import time
import matplotlib
matplotlib.use('Agg') # 'Agg'バックエンドを指定
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange
from scipy.special import logsumexp
import argparse
import random
import re # ★★★ この行を追加 ★★★
from scipy.stats import gaussian_kde # ★★★ 追加: 最頻値(mode)の計算に使う ★★★
import seaborn as sns

# Teeクラスの定義
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    # close()メソッドを追加
    def close(self):
        for f in self.files:
            if hasattr(f, 'close'):
                f.close()

# --- コマンドライン引数の解析 ---
parser = argparse.ArgumentParser(description='Run BPS analysis with specified parameters.')
parser.add_argument('-m', '--mode', type=str, required=True, choices=['IMPLEMENT', 'IMPLEMENT 2.0'], help='DGP mode to use.')
parser.add_argument('-r', '--realization', type=int, required=True, help='Realization number for naming results.')
parser.add_argument('-s', '--seed', type=int, required=True, help='Random seed for reproducibility.')
args = parser.parse_args()

# --- 乱数シードの設定 ---
SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
pyro.set_rng_seed(SEED)

# --- その他の関数定義はそのまま ---
def get_kalman_predictive_distribution(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    device = Y.device
    y_pred_mean_series = torch.zeros(N, Nt, O1, device=device)
    y_pred_cov_series = torch.zeros(N, Nt, O1, O1, device=device)
    # ★★★ 修正点: テンソルのデータ型をfloatに統一 ★★★
    eta_prev = torch.zeros(N, L1, 1, device=device).float()
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1).float() * 1e3
    for t in range(Nt):
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        P_pred = 0.5 * (P_pred + P_pred.transpose(-1, -2)) # Enforce symmetry
        y_pred_mean = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        y_pred_cov = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        y_pred_mean_series[:, t, :] = y_pred_mean.squeeze(-1)
        y_pred_cov_series[:, t, :, :] = y_pred_cov
        y_t = Y[:, t, :].unsqueeze(-1)
        v_t = y_t - y_pred_mean
        y_pred_cov_jitter = y_pred_cov + torch.eye(O1, device=device).float() * 1e-6
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(y_pred_cov_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1).float()
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))
        eta_prev, P_prev = eta_updated, P_updated
    return y_pred_mean_series, y_pred_cov_series

def get_kalman_predictions_and_latents(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    device = Y.device
    eta_series = torch.zeros(N, Nt, L1, device=device).float()
    # ★★★ 修正点: テンソルのデータ型をfloatに統一 ★★★
    eta_prev = torch.zeros(N, L1, 1, device=device).float()
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1).float() * 1e3
    for t in range(Nt):
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        y_pred = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        v_t = Y[:, t, :].unsqueeze(-1) - y_pred
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        F_t_jitter = F_t + torch.eye(O1, device=device).float() * 1e-6
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1).float()
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))
        eta_series[:, t, :] = eta_updated.squeeze(-1)
        eta_prev, P_prev = eta_updated, P_updated
    return eta_series

def get_per_time_point_log_likelihood(Y, b0, B1, Lambda1, Q, R):
    device = Y.device
    # ★★★ 修正点: 全てのパラメータを.float()に変換して統一 ★★★
    b0 = b0.to(device).float()
    B1 = B1.to(device).float()
    Lambda1 = Lambda1.to(device).float()
    Q = Q.to(device).float()
    R = R.to(device).float()
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    if b0.dim() == 1:
        b0 = b0.unsqueeze(-1)
    # ★★★ 修正点: テンソルのデータ型をfloatに統一 ★★★
    eta_prev = torch.zeros(N, L1, 1, device=device).float()
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1).float() * 1e3
    log_likelihoods_over_time = torch.zeros(N, Nt, device=device).float()
    for t in range(Nt):
        y_t = Y[:, t, :].unsqueeze(-1).float()
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
        y_pred_mean = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
        F_t_jitter = F_t + torch.eye(O1, device=device).float() * 1e-6
        dist_t = MultivariateNormal(loc=y_pred_mean.squeeze(-1), covariance_matrix=F_t_jitter)
        log_likelihoods_over_time[:, t] = dist_t.log_prob(y_t.squeeze(-1))
        v_t = y_t - y_pred_mean
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1).float()
        P_updated = torch.bmm(I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1)), P_pred)
        eta_prev, P_prev = eta_updated, P_updated
    return log_likelihoods_over_time.cpu()

# --- bps_filter_model関数全体を置き換え ---
def bps_filter_model(y_obs, m_factors, v_factors, eta_3fac, P_3fac, log_liks_stacked):
    device = y_obs.device
    N, Nt, O = y_obs.shape
    L1_state1 = eta_3fac.shape[-1]
    
    gamma_intercept = pyro.sample("gamma_intercept", dist.Normal(torch.tensor(-2.0, device=device).float(), torch.tensor(0.1, device=device).float()))
    gamma_coeffs = pyro.sample("gamma_coeffs", dist.Normal(torch.tensor(0.5, device=device).float(), torch.tensor(0.1, device=device).float()).expand([L1_state1]).to_event(1))
    p21 = 0.0

    log_lik_s1 = log_liks_stacked[:, :, 0].float()
    log_lik_s2 = log_liks_stacked[:, :, 1].float()

    # 解析的近似で使う定数
    kappa = torch.tensor(np.sqrt(np.pi / 8), device=device, dtype=torch.float32)

    with pyro.plate("individuals", N):
        pi2_prev_updated = torch.zeros(N, device=device).float()
        total_log_lik = 0
        for t in range(Nt):
            if t > 0:
                # --- etaの不確かさを考慮した遷移確率の計算 ---
                eta_hat_prev = eta_3fac[:, t - 1, :].float()
                P_prev = P_3fac[:, t - 1, :, :].float()
                
                # z の平均と分散を計算
                mu_z = gamma_intercept + torch.einsum('bi,i->b', eta_hat_prev, gamma_coeffs)
                sigma2_z = torch.einsum('i,bij,j->b', gamma_coeffs, P_prev, gamma_coeffs)
                
                # 解析的近似式で遷移確率の期待値を計算
                arg = (kappa * mu_z) / torch.sqrt(1 + (kappa**2) * sigma2_z)
                p12_t = 0.5 * (1.0 + torch.erf(arg / math.sqrt(2.0)))
                # ---------------------------------------------
                
                pi2_t_predicted = pi2_prev_updated * (1 - p21) + (1 - pi2_prev_updated) * p12_t
            else:
                pi2_t_predicted = pi2_prev_updated

            # 尤度計算
            log_numerator_s1 = log_lik_s1[:, t] + torch.log(torch.clamp(1 - pi2_t_predicted, 1e-9, 1-1e-9))
            log_numerator_s2 = log_lik_s2[:, t] + torch.log(torch.clamp(pi2_t_predicted, 1e-9, 1-1e-9))
            log_denominator = torch.logsumexp(torch.stack([log_numerator_s1, log_numerator_s2]), dim=0)
            
            # 状態確率の更新
            pi2_t_updated = torch.exp(log_numerator_s2 - log_denominator)
            pi2_prev_updated = pi2_t_updated
            total_log_lik += log_denominator.sum()

    # pyro.factorを使ってモデルの対数尤度を定義
    pyro.factor("log_likelihood", total_log_lik)

# --- 評価指標計算用の関数 (BPSと共通化) ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_sens_spec(actual, predicted_prob_s2):
    actual_binary = (actual == 2).astype(int)
    predicted_binary = (predicted_prob_s2 > 0.5).astype(int)
    TP = np.sum((predicted_binary == 1) & (actual_binary == 1))
    FN = np.sum((predicted_binary == 0) & (actual_binary == 1))
    TN = np.sum((predicted_binary == 0) & (actual_binary == 0))
    FP = np.sum((predicted_binary == 1) & (actual_binary == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return sensitivity, specificity

# --- メイン実行部 ---
if __name__ == '__main__':
    # ----------------------------------------------------------------
    # Part 0: パラメータとファイルパスの設定
    # ----------------------------------------------------------------
    DGP_MODE = args.mode
    REALIZATION_NUM = args.realization

    RESULTS_DIR = os.path.join('results', f"{DGP_MODE.replace(' 2.0', '_2.0')}", f"bps_results_run_{REALIZATION_NUM}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    DGP_RESULTS_DIR = os.path.join('data', f"{DGP_MODE.replace(' 2.0', '_2.0')}", f"dgp_data_run_{REALIZATION_NUM}")
    VAR_RESULTS_DIR = os.path.join('results', f"{DGP_MODE.replace(' 2.0', '_2.0')}", f"var_results_run_{REALIZATION_NUM}")
    
    DATA_FILE = os.path.join(DGP_RESULTS_DIR, f'simulation_data_{DGP_MODE}.pt')
    MODEL_3FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_3fac_model_{DGP_MODE}.pt')
    MODEL_1FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_1fac_model_{DGP_MODE}.pt')
    
    log_filename = os.path.join(RESULTS_DIR, f"bps_log_{DGP_MODE}_run_{REALIZATION_NUM}.txt")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, open(log_filename, 'w', encoding='utf-8'))

    print("--- 0. BPS Model Setup ---")
    # ★★★ 修正: 計算時間の公平な比較のため、デバイスをCPUに明示的に固定 ★★★
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (explicitly set for fair comparison)")
    print(f"Running in '{DGP_MODE}' mode for realization #{REALIZATION_NUM}.")
    print(f"All output files will be saved to the '{RESULTS_DIR}/' directory.")

    BPS_MCMC_RESULTS_FILE = os.path.join(RESULTS_DIR, f'bps_mcmc_results_{DGP_MODE}.pt')

    # ----------------------------------------------------------------
    # Part 1: データの読み込み
    # ----------------------------------------------------------------
    print("\n--- 1. Loading Data and Pre-trained Models ---")
    try:
        saved_data = torch.load(DATA_FILE, weights_only=False, map_location=device)
        Y_generated = saved_data['Y_generated']
        actual_states = saved_data['actual_states']
        N, Nt, O = Y_generated.shape

        model_3fac_data = torch.load(MODEL_3FAC_FILE, weights_only=False, map_location=device)
        best_params_3fac = model_3fac_data['params']
        eta_3fac = model_3fac_data['eta_series'].to(device) # <- etaをここから読み込む
        P_3fac = model_3fac_data['P_series'].to(device)     # <- Pをここから読み込む

        model_1fac_data = torch.load(MODEL_1FAC_FILE, weights_only=False, map_location=device)
        best_params_1fac = model_1fac_data['params']

        print("All necessary files loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n[Error] Could not find a required file: {e.filename}")
        print("Please ensure you have run 'dgp.py' and 'var.py' first.")
        sys.exit()

    num_bps_steps = 500
    num_bps_burnin = 500

    # ----------------------------------------------------------------
    # Part 2: BPSモデルの事前計算
    # ----------------------------------------------------------------
    print("\n--- 2. Pre-calculating inputs for BPS model ---")
    with torch.no_grad():
        # ★★★ 修正点: 入力データをfloatに変換 ★★★
        m_3fac, v_3fac = get_kalman_predictive_distribution(Y_generated.float(), **best_params_3fac)
        m_1fac, v_1fac = get_kalman_predictive_distribution(Y_generated.float(), **best_params_1fac)
        m_factors = torch.stack((m_3fac, m_1fac), dim=3)
        v_factors = torch.stack((v_3fac, v_1fac), dim=4)
    print("Pre-calculation complete.")

    # ----------------------------------------------------------------
    # Part 3: MCMCによるBPSモデルの学習
    # ----------------------------------------------------------------
    print("\n--- 3. Training BPS Model with MCMC ---")
    duration_bps = 0.0
    if os.path.exists(BPS_MCMC_RESULTS_FILE):
        print(f"Loading pre-computed BPS MCMC results from '{BPS_MCMC_RESULTS_FILE}'...")
        saved_mcmc = torch.load(BPS_MCMC_RESULTS_FILE, map_location=device)
        posterior_samples_mcmc = saved_mcmc['posterior_samples']
        if 'duration' in saved_mcmc:
            duration_bps = saved_mcmc['duration']
    else:
        print("Starting BPS model inference with MCMC (NUTS)...")
        start_time_bps = time.time()
        
        with torch.no_grad():
            log_lik_3fac_full = get_per_time_point_log_likelihood(
                Y=Y_generated.float(), b0=best_params_3fac['b0'], B1=best_params_3fac['B1'],
                Lambda1=best_params_3fac['Lambda1'], Q=best_params_3fac['Q'], R=best_params_3fac['R']
            )
            log_lik_1fac_full = get_per_time_point_log_likelihood(
                Y=Y_generated.float(), b0=best_params_1fac['b0'], B1=best_params_1fac['B1'],
                Lambda1=best_params_1fac['Lambda1'], Q=best_params_1fac['Q'], R=best_params_1fac['R']
            )
        log_liks_stacked_tensor = torch.stack([log_lik_3fac_full, log_lik_1fac_full], dim=-1).to(device)
        
        # ★★★ この2行を修正 ★★★
        nuts_kernel = NUTS(bps_filter_model)
        mcmc = MCMC(nuts_kernel, num_samples=num_bps_steps, warmup_steps=num_bps_burnin, num_chains=4) # warmupを0に、サンプル数をBPSのステップ数に設定

        # ★★★ 修正点: 入力データをfloatに変換 ★★★
        mcmc.run(Y_generated.clone().float(), m_factors.clone().float(), v_factors.clone().float(), eta_3fac.clone().float(), P_3fac.clone().float(), log_liks_stacked_tensor.float())
        
        duration_bps = time.time() - start_time_bps
        print(f"MCMC finished. Duration: {duration_bps:.2f}s")
        
        print("\n--- MCMC Convergence Diagnostics (R-hat) ---")
        mcmc.summary()
        
        posterior_samples_mcmc = mcmc.get_samples()
        
        print(f"Saving BPS MCMC results to '{BPS_MCMC_RESULTS_FILE}'...")
        torch.save({'posterior_samples': posterior_samples_mcmc, 'duration': duration_bps}, BPS_MCMC_RESULTS_FILE)
        print("Saving complete.")
        saved_mcmc = torch.load(BPS_MCMC_RESULTS_FILE, map_location=device)
        posterior_samples_mcmc = saved_mcmc['posterior_samples']

    # ----------------------------------------------------------------
    # Part 4: モデル評価 (etaとgammaの不確実性を統合)
    # ----------------------------------------------------------------
    print("\n--- 4. Evaluating Final BPS Model (propagating uncertainty from both eta and gamma) ---")
    posterior_intercepts = posterior_samples_mcmc['gamma_intercept'].reshape(-1).cpu().numpy()
    posterior_coeffs = posterior_samples_mcmc['gamma_coeffs'].reshape(
        -1, posterior_samples_mcmc['gamma_coeffs'].shape[-1]
    ).cpu().numpy()
    num_samples = len(posterior_intercepts)

    with torch.no_grad():
        log_lik_3fac_full = get_per_time_point_log_likelihood(Y=Y_generated.float(), **best_params_3fac)
        log_lik_1fac_full = get_per_time_point_log_likelihood(Y=Y_generated.float(), **best_params_1fac)
    log_liks_stacked = np.stack([log_lik_3fac_full, log_lik_1fac_full], axis=-1)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    #  FIX: etaとgammaの不確実性を統合するロジックを実装
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    all_pi2_trajectories = np.zeros((num_samples, N, Nt))
    all_log_liks = []
    eta_3fac_np = eta_3fac.cpu().numpy()
    P_3fac_np = P_3fac.cpu().numpy()

    # 正規分布の累積分布関数(CDF) = プロビット関数
    from scipy.stats import norm
    kappa = np.sqrt(np.pi / 8)
    eta_3fac_np = eta_3fac.cpu().numpy()
    P_3fac_np = P_3fac.cpu().numpy()

    for i in trange(num_samples, desc="Processing MCMC samples with eta uncertainty"):
        gamma_intercept_sample = posterior_intercepts[i]
        gamma_coeffs_sample = posterior_coeffs[i]
        pi2_prev_updated = np.zeros(N)
        
        pi2_trajectories_sample = np.zeros((N, Nt))
        log_lik_per_time_sample = np.zeros((N, Nt))

        for t in range(Nt):
            if t > 0:
                # --- etaの不確かさを考慮した遷移確率の計算 ---
                eta_hat_prev = eta_3fac_np[:, t-1, :]
                P_prev = P_3fac_np[:, t-1, :, :]
                
                mu_z = gamma_intercept_sample + np.einsum('ij,j->i', eta_hat_prev, gamma_coeffs_sample)
                sigma2_z = np.einsum('j,ijk,k->i', gamma_coeffs_sample, P_prev, gamma_coeffs_sample)
                
                p12_t = norm.cdf( (kappa * mu_z) / np.sqrt(1 + (kappa**2) * np.abs(sigma2_z)) )
                # ---------------------------------------------

                pi2_t_predicted = pi2_prev_updated * (1 - 0.0) + (1 - pi2_prev_updated) * p12_t
            else:
                pi2_t_predicted = pi2_prev_updated
            
            log_numerator_s2 = log_liks_stacked[:, t, 1] + np.log(np.clip(pi2_t_predicted, 1e-9, 1-1e-9))
            log_numerator_s1 = log_liks_stacked[:, t, 0] + np.log(np.clip(1 - pi2_t_predicted, 1e-9, 1-1e-9))
            log_denominator = logsumexp(np.stack([log_numerator_s1, log_numerator_s2]), axis=0)
            
            pi2_t_updated = np.exp(log_numerator_s2 - log_denominator)
            
            pi2_trajectories_sample[:, t] = pi2_t_updated
            log_lik_per_time_sample[:, t] = log_denominator
            
            pi2_prev_updated = pi2_t_updated

        all_pi2_trajectories[i, :, :] = pi2_trajectories_sample
        all_log_liks.append(log_lik_per_time_sample.sum())

    # 状態確率の中央値と信用区間を計算
    pi2_median = np.median(all_pi2_trajectories, axis=0)
    pi2_lower_ci = np.percentile(all_pi2_trajectories, 2.5, axis=0)
    pi2_upper_ci = np.percentile(all_pi2_trajectories, 97.5, axis=0)

    # MCMCサンプル全体の平均対数尤度を計算
    total_log_lik_bps = np.mean(all_log_liks)

    # ★★★ ここからRMSE計算の修正 ★★★
    print("  -> Calculating posterior mode of state probabilities using KDE...")
    pi2_mode = np.zeros_like(pi2_median)
    kde_grid = np.linspace(0, 1, 501)
    for n in trange(N, desc="Calculating Mode (Individuals)"):
        for t in range(Nt):
            samples = all_pi2_trajectories[:, n, t]
            if np.all(samples == samples[0]):
                pi2_mode[n, t] = samples[0]
            else:
                kde = gaussian_kde(samples)
                density = kde(kde_grid)
                pi2_mode[n, t] = kde_grid[np.argmax(density)]

    m_factors_np = m_factors.cpu().numpy()
    
    # MedianベースのRMSE
    estimated_weights_median = np.stack([1 - pi2_median, pi2_median], axis=-1)
    y_pred_median = np.sum(m_factors_np * np.expand_dims(estimated_weights_median, axis=2), axis=3)
    rmse_bps_median = calculate_rmse(Y_generated.cpu().numpy(), y_pred_median)
    
    # ModeベースのRMSE
    estimated_weights_mode = np.stack([1 - pi2_mode, pi2_mode], axis=-1)
    y_pred_mode = np.sum(m_factors_np * np.expand_dims(estimated_weights_mode, axis=2), axis=3)
    rmse_bps_mode = calculate_rmse(Y_generated.cpu().numpy(), y_pred_mode)

    sens_bps, spec_bps = calculate_sens_spec(actual_states, estimated_weights_median[:, :, 1])

    print("\n--- BPS Model Performance Summary ---")
    table_width = 100
    print("=" * table_width)
    print(f"{'Metric':<30} | {'Value'}")
    print("-" * table_width)
    print(f"{'Final Log-Likelihood':<30} | {total_log_lik_bps:.2f} (Averaged over MCMC samples)")
    print(f"{'Y Prediction RMSE (Median)':<30} | {rmse_bps_median:.4f}")
    print(f"{'Y Prediction RMSE (Mode)':<30} | {rmse_bps_mode:.4f}")
    print(f"{'State Detection Sensitivity':<30} | {sens_bps:.4f} (Correctly identifying State 2)")
    print(f"{'State Detection Specificity':<30} | {spec_bps:.4f} (Correctly identifying State 1)")
    print(f"{'Training Duration (s)':<30} | {duration_bps:.2f}")
    print("=" * table_width)

    print("\n--- Estimated vs. True Transition Parameters (using Posterior Median) ---")
    true_transition_params = {'gamma_intercept': -2.0, 'gamma_coeffs': [0.5, 0.5, 0.5]}
    est_gamma_intercept = np.median(posterior_intercepts)
    est_gamma_coeffs = np.median(posterior_coeffs, axis=0)
    print(f"  gamma_intercept: Estimated = {est_gamma_intercept:.4f}, True = {true_transition_params['gamma_intercept']}")
    for i, name in enumerate(['task', 'goal', 'bond']):
        print(f"  gamma_{name}:      Estimated = {est_gamma_coeffs[i]:.4f}, True = {true_transition_params['gamma_coeffs'][i]}")
    
    # ----------------------------------------------------------------
    # Part 5: Generating Visualizations
    # ----------------------------------------------------------------
    print("\n--- 5. Generating Visualizations ---")
    avg_weights_bps_over_time = estimated_weights_median.mean(axis=0)
    state1_proportion_actual = (actual_states == 1).mean(axis=0)
    time_points_vis = np.arange(Nt)
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 集計プロット (変更なし) ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(time_points_vis, avg_weights_bps_over_time[:, 0], 'o-', color='royalblue', label='Avg. Weight for 3-Factor Model (State 1)', zorder=3)
    ax1.plot(time_points_vis, avg_weights_bps_over_time[:, 1], 's-', color='firebrick', label='Avg. Weight for 1-Factor Model (State 2)', zorder=3)
    ax1.set_title(f'BPS Regression Model - Aggregate Performance ({DGP_MODE})', fontsize=16)
    ax1.set_xlabel('Time Point', fontsize=12)
    ax1.set_ylabel('Estimated Model Weight (Median)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(time_points_vis, state1_proportion_actual, color='grey', alpha=0.2, label='Proportion in State 1 (Actual)')
    ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)
    plt.tight_layout()
    plot_filename = os.path.join(RESULTS_DIR, f"bps_aggregate_plot_{DGP_MODE}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Aggregate plot saved to '{plot_filename}'")

    # Define the color palette at the top of the visualization section
    color_palette = {
        'BPS': sns.color_palette("muted")[0],
        'FRS': sns.color_palette("muted")[1],
        'VAR_1fac': sns.color_palette("muted")[3],
        'VAR_3fac': sns.color_palette("muted")[2]
    }

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    #  FIX: 個人プロットでUncertainty Bandを描画するロジックを復活させる
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    individuals_per_file = 10
    num_files = math.ceil(N / individuals_per_file)
    actual_states_binary = (actual_states == 2).astype(int)
    # --- Part 6の個人プロット作成ループ全体を置き換え ---
    print(f"Generating {num_files} files for individual-level plots with uncertainty bands...")
    for file_idx in range(num_files):
        start_idx = file_idx * individuals_per_file
        end_idx = min((file_idx + 1) * individuals_per_file, N)
        ids_to_plot = range(start_idx, end_idx)
        fig, axes = plt.subplots(len(ids_to_plot), 1, figsize=(10, 2.5 * len(ids_to_plot)), sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f'BPS Individual State Probabilities (Individuals {start_idx+1}-{end_idx})', fontsize=16)
        
        for i, ind_id in enumerate(ids_to_plot):
            ax = axes[i, 0]
            # ★★★ CHANGE 1: Use the BPS color from the palette for the median line ★★★
            ax.plot(time_points_vis, pi2_median[ind_id, :], color=color_palette['BPS'], ls='-', label='BPS Prob. (Median)', lw=2)
            # ★★★ CHANGE 2: Use the BPS color for the credible interval fill ★★★
            ax.fill_between(time_points_vis, pi2_lower_ci[ind_id, :], pi2_upper_ci[ind_id, :], color=color_palette['BPS'], alpha=0.2, label='95% Credible Interval')
            # ★★★ CHANGE 3: Change the true state line to black ★★★
            ax.plot(time_points_vis, actual_states_binary[ind_id, :], color='black', ls='--', label='True State', lw=1.5)
            
            ax.set_title(f'Individual #{ind_id+1}', fontsize=10)
            ax.set_ylabel('Prob. of State 2')
            if i == 0:
                ax.legend(loc='upper left')
                
        axes[-1, 0].set_xlabel('Time Point')
        plt.ylim(-0.1, 1.1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_filename = os.path.join(RESULTS_DIR, f"bps_individual_plots_{DGP_MODE}_part_{file_idx+1}.png")
        plt.savefig(save_filename)
        plt.close(fig)

    print(f"Individual-level plots saved.")
    print(f"\nBPS analysis complete. Log saved to '{log_filename}'")

    sys.stdout.close()
    sys.stdout = original_stdout
