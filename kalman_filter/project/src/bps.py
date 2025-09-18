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

def bps_filter_model(y_obs, m_factors, v_factors, eta_3fac, log_liks_stacked):
    device = y_obs.device
    N, Nt, O = y_obs.shape
    L1_state1 = eta_3fac.shape[-1]
    
    # ★★★ 修正点: パラメータの型をfloatに指定 ★★★
    gamma_intercept = pyro.sample("gamma_intercept", dist.Normal(torch.tensor(-2.5, device=device).float(), torch.tensor(2.0, device=device).float()))
    gamma_coeffs = pyro.sample("gamma_coeffs", dist.Normal(torch.tensor(0.0, device=device).float(), torch.tensor(1.0, device=device).float()).expand([L1_state1]).to_event(1))
    p21 = 0.0

    # ★★★ 修正点: 入力テンソルをfloatに変換 ★★★
    log_lik_s1 = log_liks_stacked[:, :, 0].float()
    log_lik_s2 = log_liks_stacked[:, :, 1].float()

    with pyro.plate("individuals", N):
        # ★★★ 修正点: テンソルのデータ型をfloatに統一 ★★★
        pi2_prev_updated = torch.zeros(N, device=device).float()
        for t in pyro.markov(range(Nt)):
            if t > 0:
                eta_prev = eta_3fac[:, t - 1, :].float()
                logit_p12_t = gamma_intercept + (eta_prev * gamma_coeffs).sum(dim=-1)
                p12_t = torch.sigmoid(logit_p12_t)
                pi2_t_predicted = pi2_prev_updated * (1 - p21) + (1 - pi2_prev_updated) * p12_t
            else:
                pi2_t_predicted = pi2_prev_updated
            mixture_weights = torch.stack([1 - pi2_t_predicted, pi2_t_predicted], dim=-1)
            # ★★★ 修正点: 入力テンソルをfloatに変換 ★★★
            m_t_components = m_factors[:, t, :, :].permute(0, 2, 1).float()
            v_t_components = v_factors[:, t, :, :, :].permute(0, 3, 1, 2).float()
            v_t_components = v_t_components + torch.eye(O, device=device).float() * 1e-6
            component_dist = dist.MultivariateNormal(m_t_components, covariance_matrix=v_t_components)
            mixture_dist = dist.MixtureSameFamily(dist.Categorical(mixture_weights), component_dist)
            # ★★★ 修正点: 観測データもfloatに変換 ★★★
            pyro.sample(f"obs_{t}", mixture_dist, obs=y_obs[:, t, :].float())
            log_numerator_s1 = log_lik_s1[:, t] + torch.log(1 - pi2_t_predicted + 1e-9)
            log_numerator_s2 = log_lik_s2[:, t] + torch.log(pi2_t_predicted + 1e-9)
            log_denominator = torch.logsumexp(torch.stack([log_numerator_s1, log_numerator_s2]), dim=0)
            pi2_t_updated = torch.exp(log_numerator_s2 - log_denominator)
            pi2_prev_updated = pi2_t_updated

def get_bps_transition_trajectories(posterior_samples, eta_3fac):
    print("  -> Calculating BPS trajectories using median parameters from MCMC samples...")
    # ★★★ 修正点: MCMCサンプルをfloat型で取得 ★★★
    flat_intercept = posterior_samples["gamma_intercept"].reshape(-1).float()
    gamma_intercept = flat_intercept.median()
    coeffs_tensor = posterior_samples["gamma_coeffs"]
    flat_coeffs = coeffs_tensor.reshape(-1, coeffs_tensor.shape[-1]).float()
    gamma_coeffs = flat_coeffs.median(dim=0).values
    p21 = 0
    N, Nt, _ = eta_3fac.shape
    device = eta_3fac.device
    pi2_trajectories = torch.zeros(N, Nt, device=device).float()
    pi2_prev = torch.zeros(N, device=device).float()
    with torch.no_grad():
        for t in range(Nt):
            if t > 0:
                eta_prev = eta_3fac[:, t - 1, :].float()
                logit_p12_t = gamma_intercept + (eta_prev * gamma_coeffs).sum(dim=-1)
                p12_t = torch.sigmoid(logit_p12_t)
                pi2_t = pi2_prev * (1 - p21) + (1 - pi2_prev) * p12_t
            else:
                pi2_t = pi2_prev
            pi2_trajectories[:, t] = pi2_t
            pi2_prev = pi2_t
    return torch.stack([1 - pi2_trajectories, pi2_trajectories], dim=-1).cpu().numpy()

# --- 評価指標計算用の関数 (FRSと共通化) ---
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
    device = "cpu"
    print(f"Using device: {device}")
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
        model_1fac_data = torch.load(MODEL_1FAC_FILE, weights_only=False, map_location=device)
        best_params_1fac = model_1fac_data['params']
        print("All necessary files loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n[Error] Could not find a required file: {e.filename}")
        print("Please ensure you have run 'dgp.py' and 'var.py' first.")
        sys.exit()

    # ----------------------------------------------------------------
    # Part 2: BPSモデルの事前計算
    # ----------------------------------------------------------------
    print("\n--- 2. Pre-calculating inputs for BPS model ---")
    with torch.no_grad():
        # ★★★ 修正点: 入力データをfloatに変換 ★★★
        m_3fac, v_3fac = get_kalman_predictive_distribution(Y_generated.float(), **best_params_3fac)
        m_1fac, v_1fac = get_kalman_predictive_distribution(Y_generated.float(), **best_params_1fac)
        eta_3fac = get_kalman_predictions_and_latents(Y_generated.float(), **best_params_3fac)
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
        
        nuts_kernel = NUTS(bps_filter_model)
        mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=4)

        # ★★★ 修正点: 入力データをfloatに変換 ★★★
        mcmc.run(Y_generated.clone().float(), m_factors.clone().float(), v_factors.clone().float(), eta_3fac.clone().float(), log_liks_stacked_tensor.float())
        
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
    # Part 4: モデル評価 (MCMCの全サンプルを使用)
    # ----------------------------------------------------------------
    print("\n--- 4. Evaluating Final BPS Model (using full posterior) ---")
    print("  -> Calculating evaluation metrics using the full posterior distribution...")
    posterior_intercepts = posterior_samples_mcmc['gamma_intercept'].reshape(-1).cpu().numpy()
    posterior_coeffs = posterior_samples_mcmc['gamma_coeffs'].reshape(
        -1, posterior_samples_mcmc['gamma_coeffs'].shape[-1]
    ).cpu().numpy()
    num_samples = len(posterior_intercepts)
    all_log_liks = []
    all_y_preds = []
    all_weights = []
    with torch.no_grad():
        log_lik_3fac_full = get_per_time_point_log_likelihood(
            Y=Y_generated.float(), b0=best_params_3fac['b0'], B1=best_params_3fac['B1'],
            Lambda1=best_params_3fac['Lambda1'], Q=best_params_3fac['Q'], R=best_params_3fac['R']
        )
        log_lik_1fac_full = get_per_time_point_log_likelihood(
            Y=Y_generated.float(), b0=best_params_1fac['b0'], B1=best_params_1fac['B1'],
            Lambda1=best_params_1fac['Lambda1'], Q=best_params_1fac['Q'], R=best_params_1fac['R']
        )
    log_liks_stacked = np.stack([log_lik_3fac_full, log_lik_1fac_full], axis=-1)
    m_factors_np = m_factors.cpu().numpy()
    p21 = 0.0
    for i in trange(num_samples, desc="Processing MCMC samples"):
        gamma_intercept_sample = posterior_intercepts[i]
        gamma_coeffs_sample = posterior_coeffs[i]
        pi2_trajectories = np.zeros((N, Nt))
        pi2_prev_updated = np.zeros(N)
        for t in range(Nt):
            if t > 0:
                eta_prev = eta_3fac[:, t - 1, :].cpu().numpy()
                logit_p12_t = gamma_intercept_sample + (eta_prev * gamma_coeffs_sample).sum(axis=-1)
                p12_t = 1 / (1 + np.exp(-logit_p12_t))
                pi2_t_predicted = pi2_prev_updated * (1 - p21) + (1 - pi2_prev_updated) * p12_t
            else:
                pi2_t_predicted = pi2_prev_updated
            log_lik_s1_t = log_liks_stacked[:, t, 0]
            log_lik_s2_t = log_liks_stacked[:, t, 1]
            log_numerator_s1 = log_lik_s1_t + np.log(1 - pi2_t_predicted + 1e-9)
            log_numerator_s2 = log_lik_s2_t + np.log(pi2_t_predicted + 1e-9)
            log_denominator = logsumexp(np.stack([log_numerator_s1, log_numerator_s2]), axis=0)
            pi2_t_updated = np.exp(log_numerator_s2 - log_denominator)
            pi2_trajectories[:, t] = pi2_t_updated
            pi2_prev_updated = pi2_t_updated
        estimated_weights_sample = np.stack([1 - pi2_trajectories, pi2_trajectories], axis=-1)
        all_weights.append(estimated_weights_sample)
        log_weights_sample = np.log(estimated_weights_sample + 1e-12)
        marginal_log_lik_k_sample = log_weights_sample + log_liks_stacked
        log_lik_per_time = logsumexp(marginal_log_lik_k_sample, axis=-1)
        total_log_lik_sample = log_lik_per_time.sum()
        all_log_liks.append(total_log_lik_sample)
        weights_expanded_sample = np.expand_dims(estimated_weights_sample, axis=2)
        y_pred_sample = np.sum(m_factors_np * weights_expanded_sample, axis=3)
        all_y_preds.append(y_pred_sample)
    print("  -> Aggregating results from all samples...")
    estimated_weights_bps = np.mean(np.array(all_weights), axis=0)
    y_pred_bps = np.mean(np.array(all_y_preds), axis=0)
    total_log_lik_bps = np.mean(all_log_liks)
    rmse_bps = calculate_rmse(Y_generated.cpu().numpy(), y_pred_bps)
    sens_bps, spec_bps = calculate_sens_spec(actual_states, estimated_weights_bps[:, :, 1])
    print("\n--- BPS Model Performance Summary ---")
    table_width = 100
    print("=" * table_width)
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * table_width)
    print(f"{'Final Log-Likelihood':<25} | {total_log_lik_bps:.2f} (Averaged over MCMC samples)")
    print(f"{'Y Prediction RMSE':<25} | {rmse_bps:.4f}")
    print(f"{'State Detection Sensitivity':<25} | {sens_bps:.4f} (Correctly identifying State 2)")
    print(f"{'State Detection Specificity':<25} | {spec_bps:.4f} (Correctly identifying State 1)")
    print(f"{'Training Duration (s)':<25} | {duration_bps:.2f}")
    print("=" * table_width)
    print("\n--- Estimated vs. True Transition Parameters (using Posterior Median) ---")
    true_params = {'gamma_intercept': -2.0, 'gamma_coeffs': [0.5, 0.5, 0.5], 'p21': 0.0}
    g0_med = np.median(posterior_intercepts)
    print(f"  gamma_intercept: Estimated = {g0_med:.4f}, True = {true_params['gamma_intercept']}")
    gc_tensor = posterior_samples_mcmc['gamma_coeffs'].reshape(-1, posterior_samples_mcmc['gamma_coeffs'].shape[-1])
    gc_med = np.median(posterior_coeffs, axis=0)
    for i, name in enumerate(['task', 'goal', 'bond']):
        print(f"  gamma_{name}:      Estimated = {gc_med[i]:.4f}, True = {true_params['gamma_coeffs'][i]}")
    print("\n--- 5. Generating Visualizations ---")
    avg_weights_bps_over_time = estimated_weights_bps.mean(axis=0)
    state1_proportion_actual = (actual_states == 1).mean(axis=0)
    time_points_vis = np.arange(Nt)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(time_points_vis, avg_weights_bps_over_time[:, 0], 'o-', color='royalblue', label='Avg. Weight for 3-Factor Model (State 1)', zorder=3)
    ax1.plot(time_points_vis, avg_weights_bps_over_time[:, 1], 's-', color='firebrick', label='Avg. Weight for 1-Factor Model (State 2)', zorder=3)
    ax1.set_title(f'BPS Regression Model - Aggregate Performance ({DGP_MODE})', fontsize=16)
    ax1.set_xlabel('Time Point', fontsize=12)
    ax1.set_ylabel('Estimated Model Weight', fontsize=12)
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
    individuals_per_file = 10
    num_files = math.ceil(N / individuals_per_file)
    actual_states_binary = (actual_states == 2).astype(int)
    print(f"Generating {num_files} files for individual-level plots...")
    for file_idx in range(num_files):
        start_idx = file_idx * individuals_per_file
        end_idx = min((file_idx + 1) * individuals_per_file, N)
        ids_to_plot = range(start_idx, end_idx)
        fig, axes = plt.subplots(len(ids_to_plot), 1, figsize=(10, 2.5 * len(ids_to_plot)), sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f'BPS Individual State Probabilities (Individuals {start_idx+1}-{end_idx})', fontsize=16)
        for i, ind_id in enumerate(ids_to_plot):
            ax = axes[i, 0]
            ax.plot(time_points_vis, estimated_weights_bps[ind_id, :, 1], 'g-', label='BPS Prob. (State 2)', lw=2)
            ax.plot(time_points_vis, actual_states_binary[ind_id, :], 'r--', label='True State', lw=1.5)
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
    sys.stdout = original_stdout
    print(f"\nBPS analysis complete. Log saved to '{log_filename}'")