# BPS/bps.py (MODIFIED)

import math
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange
from scipy.special import logsumexp # ★★★ この行を追加 ★★★

# --- 共通のユーティリティ ---
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


def get_kalman_predictive_distribution(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    device = Y.device
    y_pred_mean_series = torch.zeros(N, Nt, O1, device=device)
    y_pred_cov_series = torch.zeros(N, Nt, O1, O1, device=device)
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    for t in range(Nt):
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        y_pred_mean = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        y_pred_cov = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        y_pred_mean_series[:, t, :] = y_pred_mean.squeeze(-1)
        y_pred_cov_series[:, t, :, :] = y_pred_cov
        y_t = Y[:, t, :].unsqueeze(-1)
        v_t = y_t - y_pred_mean
        y_pred_cov_jitter = y_pred_cov + torch.eye(O1, device=device) * 1e-6
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(y_pred_cov_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        
        # ★★★ 修正点: Joseph Formに更新 ★★★
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))

        eta_prev, P_prev = eta_updated, P_updated
    return y_pred_mean_series, y_pred_cov_series

def get_kalman_predictions_and_latents(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    device = Y.device
    eta_series = torch.zeros(N, Nt, L1, device=device)
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    for t in range(Nt):
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        y_pred = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        v_t = Y[:, t, :].unsqueeze(-1) - y_pred
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        F_t_jitter = F_t + torch.eye(O1, device=device) * 1e-6
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        
        # ★★★ 修正点: Joseph Formに更新 ★★★
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))

        eta_series[:, t, :] = eta_updated.squeeze(-1)
        eta_prev, P_prev = eta_updated, P_updated
    return eta_series

def get_per_time_point_log_likelihood(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    device = Y.device
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    log_likelihoods_over_time = torch.zeros(N, Nt, device=device)
    for t in range(Nt):
        y_t = Y[:, t, :].unsqueeze(-1)
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        try:
            F_t_jitter = F_t + torch.eye(O1, device=device) * 1e-6
            dist_t = MultivariateNormal(loc=v_t.squeeze(-1), covariance_matrix=F_t_jitter)
            log_likelihoods_over_time[:, t] = dist_t.log_prob(torch.zeros_like(v_t.squeeze(-1)))
        except torch.linalg.LinAlgError:
            log_likelihoods_over_time[:, t] = -torch.inf
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        
        # ★★★ 修正点: Joseph Formに更新 ★★★
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))
                    
        eta_prev, P_prev = eta_updated, P_updated
    return log_likelihoods_over_time

# bps.py (MODIFIED bps_transition_model function)

def bps_transition_model(y_obs, m_factors, v_factors, eta_3fac):
    device = y_obs.device
    N, Nt, O, J = m_factors.shape
    L1_state1 = eta_3fac.shape[-1]
    
    gamma_intercept = pyro.sample("gamma_intercept", dist.Normal(torch.tensor(-2.5, device=device), torch.tensor(2.0, device=device)))
    gamma_coeffs = pyro.sample("gamma_coeffs", dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).expand([L1_state1]).to_event(1))
    p21 = 0.0

    with pyro.plate("individuals", N):
        pi2_prev = torch.zeros(N, device=device)
        for t in pyro.markov(range(Nt)):
            if t > 0:

                eta_prev = eta_3fac[:, t - 1, :]
                logit_p12_t = gamma_intercept + (eta_prev * gamma_coeffs).sum(dim=-1)
                p12_t = torch.sigmoid(logit_p12_t)
                pi2_t = pi2_prev * (1 - p21) + (1 - pi2_prev) * p12_t

            else:
                pi2_t = pi2_prev
            
            # ★★★ 修正点: このブロック全体をforループの中に正しく配置する ★★★
            mixture_weights = torch.stack([1 - pi2_t, pi2_t], dim=-1)
            m_t_components = m_factors[:, t, :, :].permute(0, 2, 1)
            v_t_components = v_factors[:, t, :, :, :].permute(0, 3, 1, 2)
            
            # 対角成分にジッターを加えて数値安定性を確保
            v_t_components = v_t_components + torch.eye(O, device=device) * 1e-6
            
            component_dist = dist.MultivariateNormal(m_t_components, covariance_matrix=v_t_components)
            mixture_dist = dist.MixtureSameFamily(dist.Categorical(mixture_weights), component_dist)
            
            pyro.sample(f"obs_{t}", mixture_dist, obs=y_obs[:, t, :])
            
            pi2_prev = pi2_t

def get_bps_transition_trajectories(posterior_samples, eta_3fac):
    print("  -> Calculating BPS trajectories using median parameters from MCMC samples...")

    # --- ここから修正 ---
    # チェーンとサンプルの次元を統合 (例: (4, 1000) -> (4000))
    flat_intercept = posterior_samples["gamma_intercept"].reshape(-1)
    gamma_intercept = flat_intercept.median()

    # 同様に係数も統合 (例: (4, 1000, 3) -> (4000, 3))
    coeffs_tensor = posterior_samples["gamma_coeffs"]
    flat_coeffs = coeffs_tensor.reshape(-1, coeffs_tensor.shape[-1])
    gamma_coeffs = flat_coeffs.median(dim=0).values
    # --- ここまで修正 ---
    p21 = 0
    
    N, Nt, _ = eta_3fac.shape
    device = eta_3fac.device
    
    pi2_trajectories = torch.zeros(N, Nt, device=device)
    pi2_prev = torch.zeros(N, device=device)

    with torch.no_grad():
        for t in range(Nt):
            if t > 0:
                eta_prev = eta_3fac[:, t - 1, :]
                # ★★★ 変更点: 'gamma_0' を 'gamma_intercept' に変更 ★★★
                logit_p12_t = gamma_intercept + (eta_prev * gamma_coeffs).sum(dim=-1)
                p12_t = torch.sigmoid(logit_p12_t)
                pi2_t = pi2_prev * (1 - p21) + (1 - pi2_prev) * p12_t
            else:
                pi2_t = pi2_prev
            pi2_trajectories[:, t] = pi2_t
            pi2_prev = pi2_t
            
    return torch.stack([1 - pi2_trajectories, pi2_trajectories], dim=-1).cpu().numpy()

# --- ★★★ 評価指標計算用の関数 (FRSと共通化) ★★★ ---
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
    DGP_MODE = 'IMPLEMENT 2.0'
    
    # ★★★ CHANGE: Create results directory and set up logging ★★★
    RESULTS_DIR = f"results_{DGP_MODE.replace(' 2.0', '_2.0')}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    log_filename = os.path.join(RESULTS_DIR, f"bps_log_{DGP_MODE}.txt")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, open(log_filename, 'w', encoding='utf-8'))

    print("--- 0. BPS Model Setup ---")
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running in '{DGP_MODE}' mode.")
    print(f"All output files will be saved to the '{RESULTS_DIR}/' directory.")

    # ★★★ CHANGE: Update file paths to use results subdirectories ★★★
    # Input files (reading from results subdirectories in DGP and VAR)
    DGP_RESULTS_DIR = os.path.join('..', 'DGP', RESULTS_DIR)
    VAR_RESULTS_DIR = os.path.join('..', 'VAR', RESULTS_DIR)
    DATA_FILE = os.path.join(DGP_RESULTS_DIR, f'simulation_data_{DGP_MODE}.pt')
    MODEL_3FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_3fac_model_{DGP_MODE}.pt')
    MODEL_1FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_1fac_model_{DGP_MODE}.pt')

    # Output file (saving to the local results directory)
    BPS_MCMC_RESULTS_FILE = os.path.join(RESULTS_DIR, f'bps_mcmc_results_{DGP_MODE}.pt')

    # ----------------------------------------------------------------
    # Part 1: データの読み込み
    # ----------------------------------------------------------------
    print("\n--- 1. Loading Data and Pre-trained Models ---")
    saved_data = torch.load(DATA_FILE, weights_only=False, map_location=device)
    Y_generated = saved_data['Y_generated']
    actual_states = saved_data['actual_states']
    N, Nt, O = Y_generated.shape

    model_3fac_data = torch.load(MODEL_3FAC_FILE, weights_only=False, map_location=device)
    best_params_3fac = model_3fac_data['params']

    model_1fac_data = torch.load(MODEL_1FAC_FILE, weights_only=False, map_location=device)
    best_params_1fac = model_1fac_data['params']
    
    print("All necessary files loaded successfully.")

    # ----------------------------------------------------------------
    # Part 2: BPSモデルの事前計算
    # ----------------------------------------------------------------
    print("\n--- 2. Pre-calculating inputs for BPS model ---")
    with torch.no_grad():
        m_3fac, v_3fac = get_kalman_predictive_distribution(Y_generated, **best_params_3fac)
        m_1fac, v_1fac = get_kalman_predictive_distribution(Y_generated, **best_params_1fac)
        eta_3fac = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
        
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
        nuts_kernel = NUTS(bps_transition_model)
        
        # ★★★ 修正点1: num_chains を1から4に変更 ★★★
        mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=50, num_chains=4)

        mcmc.run(Y_generated.clone(), m_factors.clone(), v_factors.clone(), eta_3fac.clone())
        duration_bps = time.time() - start_time_bps
        print(f"MCMC finished. Duration: {duration_bps:.2f}s")
        
        # ★★★ 修正点: mcmc.summary()を呼び出すだけにする ★★★
        print("\n--- MCMC Convergence Diagnostics (R-hat) ---")
        mcmc.summary() # この行だけでサマリーが表示される
        
        posterior_samples_mcmc = mcmc.get_samples()
        
        print(f"Saving BPS MCMC results to '{BPS_MCMC_RESULTS_FILE}'...")
        torch.save({'posterior_samples': posterior_samples_mcmc, 'duration': duration_bps}, BPS_MCMC_RESULTS_FILE)
        print("Saving complete.")

        # --- ★★★ この3行を追加してください ★★★ ---
        print("  -> Reloading samples from file to ensure a clean state...")
        saved_mcmc = torch.load(BPS_MCMC_RESULTS_FILE, map_location=device)
        posterior_samples_mcmc = saved_mcmc['posterior_samples']
        # --- ★★★ ここまで ★★★

    # ----------------------------------------------------------------
    # Part 4: モデル評価 (MCMCの全サンプルを使用)
    # ----------------------------------------------------------------
    print("\n--- 4. Evaluating Final BPS Model (using full posterior) ---")

    # --- 4a. 全MCMCサンプルを用いた評価指標の計算 ---
    print("  -> Calculating evaluation metrics using the full posterior distribution...")

    # 事後分布からサンプルを抽出
    posterior_intercepts = posterior_samples_mcmc['gamma_intercept'].reshape(-1)
    posterior_coeffs = posterior_samples_mcmc['gamma_coeffs'].reshape(-1, posterior_samples_mcmc['gamma_coeffs'].shape[-1])
    num_samples = len(posterior_intercepts)

    # 各サンプルの結果を格納するための空のリストを準備
    all_log_liks = []
    all_y_preds = []
    all_weights = []

    # 各状態の対数尤度を事前に計算（遷移パラメータには依存しない）
    with torch.no_grad():
        log_lik_3fac_full = get_per_time_point_log_likelihood(Y_generated, **best_params_3fac).cpu().numpy()
        log_lik_1fac_full = get_per_time_point_log_likelihood(Y_generated, **best_params_1fac).cpu().numpy()
    log_liks_stacked = np.stack([log_lik_3fac_full, log_lik_1fac_full], axis=-1)
    m_factors_np = m_factors.cpu().numpy()
    p21 = 0.0

    # 全MCMCサンプルに対してループ処理を実行
    for i in trange(num_samples, desc="Processing MCMC samples"):
        gamma_intercept_sample = posterior_intercepts[i]
        gamma_coeffs_sample = posterior_coeffs[i]

        # --- このサンプルにおける状態遷移確率を計算 ---
        N, Nt, _ = eta_3fac.shape
        device = eta_3fac.device
        pi2_trajectories = torch.zeros(N, Nt, device=device)
        pi2_prev = torch.zeros(N, device=device)

        with torch.no_grad():
            for t in range(Nt):
                if t > 0:
                    eta_prev = eta_3fac[:, t - 1, :]
                    logit_p12_t = gamma_intercept_sample + (eta_prev * gamma_coeffs_sample).sum(dim=-1)
                    p12_t = torch.sigmoid(logit_p12_t)
                    pi2_t = pi2_prev * (1 - p21) + (1 - pi2_prev) * p12_t
                else:
                    pi2_t = pi2_prev
                pi2_trajectories[:, t] = pi2_t
                pi2_prev = pi2_t
                
        estimated_weights_sample = torch.stack([1 - pi2_trajectories, pi2_trajectories], dim=-1).cpu().numpy()
        all_weights.append(estimated_weights_sample)

        # --- このサンプルにおける対数尤度を計算 ---
        log_weights_sample = np.log(estimated_weights_sample + 1e-12) # log(0)を防ぐ
        marginal_log_lik_k_sample = log_weights_sample + log_liks_stacked
        # logsumexpトリックを使い、各個人の対数尤度を合計
        total_log_lik_sample = logsumexp(marginal_log_lik_k_sample, axis=-1).sum()
        all_log_liks.append(total_log_lik_sample)

        # --- このサンプルにおけるYの予測値を計算 ---
        weights_expanded_sample = np.expand_dims(estimated_weights_sample, axis=2)
        # y_pred = w1*m1 + w2*m2
        y_pred_sample = np.sum(m_factors_np * weights_expanded_sample, axis=3)
        all_y_preds.append(y_pred_sample)

    # --- 結果を集計 ---
    print("  -> Aggregating results from all samples...")
    # 最終的な重みは全サンプルの平均
    estimated_weights_bps = np.mean(np.array(all_weights), axis=0)
    # 最終的なYの予測値は全サンプルの平均
    y_pred_bps = np.mean(np.array(all_y_preds), axis=0)
    # 最終的な対数尤度は全サンプルの平均
    total_log_lik_bps = np.mean(all_log_liks)


    # --- 4b. 最終的な評価指標の計算 ---
    # 1. Y予測RMSE
    rmse_bps = calculate_rmse(Y_generated.cpu().numpy(), y_pred_bps)

    # 2. 状態検出の感度・特異度
    sens_bps, spec_bps = calculate_sens_spec(actual_states, estimated_weights_bps[:, :, 1])

    # --- 4c. パフォーマンスサマリー ---
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

    # --- 4d. 推定パラメータと真値の比較 (中央値を使用) ---
    print("\n--- Estimated vs. True Transition Parameters (using Posterior Median) ---")
    true_params = {'gamma_intercept': -2.0, 'gamma_coeffs': [0.5, 0.5, 0.5], 'p21': 0.0}

    # gamma_intercept
    g0_med = posterior_intercepts.median().item()
    print(f"  gamma_intercept: Estimated = {g0_med:.4f}, True = {true_params['gamma_intercept']}")

    # gamma_coeffs (task, goal, bond)   
    gc_tensor = posterior_samples_mcmc['gamma_coeffs'].reshape(-1, posterior_samples_mcmc['gamma_coeffs'].shape[-1])
    gc_med = gc_tensor.median(dim=0).values.cpu().numpy()
    for i, name in enumerate(['task', 'goal', 'bond']):
        print(f"  gamma_{name}:      Estimated = {gc_med[i]:.4f}, True = {true_params['gamma_coeffs'][i]}")
        
    # ----------------------------------------------------------------
    # Part 5: 可視化
    # ----------------------------------------------------------------
    print("\n--- 5. Generating Visualizations ---")
    
    # --- 5a. 集計プロット ---
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
    # ★★★ CHANGE: Save plot to the results directory ★★★
    plot_filename = os.path.join(RESULTS_DIR, f"bps_aggregate_plot_{DGP_MODE}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Aggregate plot saved to '{plot_filename}'")

    # --- 5b. 個人レベルのプロット ---
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
        
        # ★★★ CHANGE: Save plot to the results directory ★★★
        save_filename = os.path.join(RESULTS_DIR, f"bps_individual_plots_{DGP_MODE}_part_{file_idx+1}.png")
        plt.savefig(save_filename)
        plt.close(fig)
        print(f"Individual-level plots saved.")
  
    sys.stdout = original_stdout
    print(f"\nBPS analysis complete. Log saved to '{log_filename}'")