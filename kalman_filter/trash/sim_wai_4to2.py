import torch
import numpy as np
import random
import math
import sys
from torch.distributions.multivariate_normal import MultivariateNormal

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ここから追加 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
class Tee:
    """
    コンソールとファイルに同時に書き出すためのクラス。
    Rの sink(split=TRUE) と同様の機能を提供します。
    """
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
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ここまで追加 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ Teeクラスを使って全体の処理を囲む ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
with Tee("sim_wai_compared_result.txt"):
    # ----------------------------------------------------------------
    # Part 0: モデルパラメータの定義 (論文ベースの体制スイッチモデル)
    # ----------------------------------------------------------------
    print("--- 0. Defining Parameters for the Regime-Switching Model ---")

    # デバイス設定 (GPUが利用可能ならGPUを使用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 全体の設定
    N = 100
    Nt = 20
    O = 12  # 観測変数 (Alliance項目x9 + Symptom項目x3)

    # --- State 1: 4因子モデル (治療初期) ---
    L1_state1 = 4  # 潜在変数の数 (Task, Goal, Bond, Symptom)

    B1_state1_matrix = np.array([
        [0.4, 0.0, 0.0, -0.05],  # TASK(t-1) -> GOAL(t), BOND(t) への効果を0に
        [0.0, 0.45, 0.0, -0.05],  # GOAL(t-1) -> TASK(t), BOND(t) への効果を0に
        [0.0, 0.0, 0.5, -0.1],  # BOND(t-1) -> TASK(t), GOAL(t) への効果を0に
        [-0.1, -0.1, -0.1, 0.6]
    ], dtype=np.float32)

    B1_true_state1 = torch.tensor(B1_state1_matrix, device=device)
    Lambda1_state1_matrix = np.zeros((O, L1_state1), dtype=np.float32)
    Lambda1_state1_matrix[0:3, 0] = [1.0, 0.8, 0.7]
    Lambda1_state1_matrix[3:6, 1] = [1.0, 1.2, 0.8]
    Lambda1_state1_matrix[6:9, 2] = [1.0, 0.9, 0.7]
    Lambda1_state1_matrix[9:12, 3] = [1.0, 1.1, 0.9]
    Lambda1_true_state1 = torch.tensor(Lambda1_state1_matrix, device=device)

    # --- State 2: 2因子モデル (治療後期) ---
    L1_state2 = 2
    B1_state2_matrix = np.array([
        [0.6, -0.25],
        [-0.1, 0.5]
    ], dtype=np.float32)
    B1_true_state2 = torch.tensor(B1_state2_matrix, device=device)
    Lambda1_state2_matrix = np.zeros((O, L1_state2), dtype=np.float32)
    Lambda1_state2_matrix[0:9, 0] = [0.9, 0.8, 0.7, 1.0, 1.1, 0.9, 0.9, 0.8, 0.7]
    Lambda1_state2_matrix[9:12, 1] = [1.0, 1.1, 0.9]
    Lambda1_true_state2 = torch.tensor(Lambda1_state2_matrix, device=device)

    Q_state1 = torch.eye(L1_state1, device=device)
    Q_state2 = torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept = -2.5
    gamma_task = 0.1
    gamma_goal = 0.1
    gamma_bond = 0.1
    gamma_symptom = -0.2

    # ----------------------------------------------------------------
    # Part 1: データ生成 (体制スイッチモデル)
    # ----------------------------------------------------------------
    print("\n--- 1. Generating Simulation Data from Regime-Switching Model ---")

    Y_generated = torch.zeros(N, Nt, O, device=device)
    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)

    switch_points = []
    for i in range(N):
        eta_history = torch.zeros(L1_state1, 1, device=device)
        current_state = 1
        has_switched = False
        for t in range(Nt):
            if current_state == 1 and t > 0:
                task_prev, goal_prev, bond_prev, symptom_prev = eta_history.squeeze().tolist()
                z = gamma_intercept + gamma_task * task_prev + gamma_goal * goal_prev + gamma_bond * bond_prev + gamma_symptom * symptom_prev
                switch_prob = 1 / (1 + math.exp(-z))
                if random.random() < switch_prob:
                    current_state = 2
                    if not has_switched:
                        switch_points.append(t)
                        has_switched = True
                    
                    wai_prev = (task_prev + goal_prev + bond_prev) / 3
                    eta_history = torch.tensor([wai_prev, symptom_prev], device=device).reshape(L1_state2, 1)

            if current_state == 1:
                eta_mean_t = B1_true_state1 @ eta_history
                noise_q = q_dist_s1.sample().reshape(L1_state1, 1)
                eta_t = eta_mean_t + noise_q
                y_mean_t = Lambda1_true_state1 @ eta_t
            else: # current_state == 2
                eta_mean_t = B1_true_state2 @ eta_history
                noise_q = q_dist_s2.sample().reshape(L1_state2, 1)
                eta_t = eta_mean_t + noise_q
                y_mean_t = Lambda1_true_state2 @ eta_t
            
            noise_r = r_dist.sample().reshape(O, 1)
            y_t = y_mean_t + noise_r
            Y_generated[i, t, :] = y_t.squeeze()
            eta_history = eta_t

    print("Simulation data with regime-switching process generated.")
    print(f"Number of individuals switched to State 2: {len(switch_points)} / {N}")
    if switch_points:
        print(f"Average switch time point (for those who switched): {np.mean(switch_points):.4f}")


    # ----------------------------------------------------------------
    # Part 2: 共通の関数定義
    # ----------------------------------------------------------------
    print("\n--- 2. Defining Common Functions ---")

    # 修正済みのkalman_filter_torch関数
    def kalman_filter_torch(Y1, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
        N, Nt, O1 = Y1.shape
        L1 = B1.shape[0]
        
        total_log_likelihood = torch.tensor(0.0, device=Y1.device)
        I_mat = torch.eye(L1, device=Y1.device)
        
        for i in range(N):
            eta_prev = eta1_i0_0.clone()
            P_prev = P_i0_0.clone()
            y1_i = Y1[i, :, :]
            
            for t in range(Nt):
                y1_it = y1_i[t, :].reshape(O1, 1)
                
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
                log_likelihood_it = -0.5 * O1 * math.log(2 * math.pi) - 0.5 * log_det_F_it + exponent_term
                
                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ この行を修正 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                total_log_likelihood += log_likelihood_it.squeeze()
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                eta_prev = eta_updated
                P_prev = P_updated
                
        return total_log_likelihood

    def create_matrix_from_ranges(size, diag_min, diag_max, off_diag_min, off_diag_max):
        mat = np.random.uniform(off_diag_min, off_diag_max, (size, size))
        np.fill_diagonal(mat, np.random.uniform(diag_min, diag_max, size))
        return torch.tensor(mat, dtype=torch.float32)

    # --- 推定の共通設定 ---
    num_runs = 5
    learning_rate = 0.0005
    num_epochs = 2000
    l2_decay = 0.01


    # ----------------------------------------------------------------
    # Part 3: 4因子モデルによるパラメータ推定 (簡略化版)
    # ----------------------------------------------------------------
    print("\n\n--- 3. Parameter Estimation using a simplified 4-Factor Model ---")
    L1_4fac = 4
    best_loss_4fac = float('inf')
    best_params_4fac = {}
    Q_est_4fac = (torch.eye(L1_4fac) * 0.5).to(device)
    R_est_4fac = (torch.eye(O) * 1.0).to(device)

    for run in range(num_runs):
        print(f"\n--- Starting Run (4-Factor): {run + 1} / {num_runs} ---")
        
        b0 = torch.randn(L1_4fac, 1, requires_grad=True, device=device)
        b1_free_params = torch.randn(10, requires_grad=True, device=device)
        lambda1_free_params = torch.randn(8, requires_grad=True, device=device)
        
        params_to_optimize = [b0, b1_free_params, lambda1_free_params]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=l2_decay)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            B1 = torch.zeros(L1_4fac, L1_4fac, device=device)
            B1[0,0] = b1_free_params[0]; B1[1,1] = b1_free_params[1]
            B1[2,2] = b1_free_params[2]; B1[3,3] = b1_free_params[3]
            B1[3,0] = b1_free_params[4]; B1[3,1] = b1_free_params[5]
            B1[3,2] = b1_free_params[6]
            B1[0,3] = b1_free_params[7]; B1[1,3] = b1_free_params[8]
            B1[2,3] = b1_free_params[9]
            
            Lambda1 = torch.zeros(O, L1_4fac, device=device)
            Lambda1[0, 0] = 1; Lambda1[1, 0] = lambda1_free_params[0]; Lambda1[2, 0] = lambda1_free_params[1]
            Lambda1[3, 1] = 1; Lambda1[4, 1] = lambda1_free_params[2]; Lambda1[5, 1] = lambda1_free_params[3]
            Lambda1[6, 2] = 1; Lambda1[7, 2] = lambda1_free_params[4]; Lambda1[8, 2] = lambda1_free_params[5]
            Lambda1[9, 3] = 1; Lambda1[10, 3] = lambda1_free_params[6]; Lambda1[11, 3] = lambda1_free_params[7]
            
            log_likelihood = kalman_filter_torch(
                Y1=Y_generated, b0=b0, B1=B1, Lambda1=Lambda1, Q=Q_est_4fac, R=R_est_4fac,
                eta1_i0_0=torch.zeros(L1_4fac, 1, device=device),
                P_i0_0=torch.eye(L1_4fac, device=device) * 1e3
            )
            loss = -log_likelihood
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Run: {run + 1}, Epoch: {epoch + 1}, Loss is NaN/Inf. Stopping this run.")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()
            
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"Run: {run + 1}, Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
        
        if loss.item() < best_loss_4fac:
            best_loss_4fac = loss.item()
            best_params_4fac = {
                'b1_free_params': b1_free_params.detach().clone(),
                'lambda1_free_params': lambda1_free_params.detach().clone()
            }
            print(f"!!! New best result for 4-Factor Model found in run {run + 1} !!!")

    # ----------------------------------------------------------------
    # Part 4: 2因子モデルによるパラメータ推定
    # ----------------------------------------------------------------
    print("\n\n--- 4. Parameter Estimation using a 2-Factor Model ---")
    L1_2fac = 2
    best_loss_2fac = float('inf')
    best_params_2fac = {}
    Q_est_2fac = (torch.eye(L1_2fac) * 0.5).to(device)
    R_est_2fac = (torch.eye(O) * 1.0).to(device)

    for run in range(num_runs):
        print(f"\n--- Starting Run (2-Factor): {run + 1} / {num_runs} ---")
        
        b0 = torch.randn(L1_2fac, 1, requires_grad=True, device=device)
        B1_init = create_matrix_from_ranges(L1_2fac, diag_min=0.3, diag_max=0.7, off_diag_min=-0.2, off_diag_max=0.2)
        B1 = B1_init.clone().detach().to(device).requires_grad_(True)
        lambda1_free_params = torch.randn(10, requires_grad=True, device=device)
        
        params_to_optimize = [b0, B1, lambda1_free_params]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=l2_decay)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            Lambda1 = torch.zeros(O, L1_2fac, device=device)
            Lambda1[0, 0] = 1.0
            Lambda1[1:9, 0] = lambda1_free_params[0:8]
            Lambda1[9, 1] = 1.0
            Lambda1[10:12, 1] = lambda1_free_params[8:10]
            
            log_likelihood = kalman_filter_torch(
                Y1=Y_generated, b0=b0, B1=B1, Lambda1=Lambda1, Q=Q_est_2fac, R=R_est_2fac,
                eta1_i0_0=torch.zeros(L1_2fac, 1, device=device),
                P_i0_0=torch.eye(L1_2fac, device=device) * 1e3
            )
            loss = -log_likelihood
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Run: {run + 1}, Epoch: {epoch + 1}, Loss is NaN/Inf. Stopping this run.")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()
            
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"Run: {run + 1}, Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
                
        if loss.item() < best_loss_2fac:
            best_loss_2fac = loss.item()
            best_params_2fac = {
                'B1': B1.detach().clone(),
                'lambda1_free_params': lambda1_free_params.detach().clone()
            }
            print(f"!!! New best result for 2-Factor Model found in run {run + 1} !!!")

    # ----------------------------------------------------------------
    # Part 5: 結果の表示とモデル比較
    # ----------------------------------------------------------------
    print("\n\n--- 5. Final Results and Model Comparison ---\n")

    # --- 4因子モデルの最良結果 ---
    if best_params_4fac:
        print("--- Best 4-Factor Model Estimation Results ---")
        print(f"Best Negative Log-Likelihood (Loss): {best_loss_4fac:.4f}")
        
        B1_est_4fac = torch.zeros(L1_4fac, L1_4fac)
        b1_params = best_params_4fac['b1_free_params']
        B1_est_4fac[0,0]=b1_params[0]; B1_est_4fac[1,1]=b1_params[1]; B1_est_4fac[2,2]=b1_params[2]; B1_est_4fac[3,3]=b1_params[3]
        B1_est_4fac[3,0]=b1_params[4]; B1_est_4fac[3,1]=b1_params[5]; B1_est_4fac[3,2]=b1_params[6]
        B1_est_4fac[0,3]=b1_params[7]; B1_est_4fac[1,3]=b1_params[8]; B1_est_4fac[2,3]=b1_params[9]
        
        Lambda1_est_4fac = torch.zeros(O, L1_4fac)
        lambda_params = best_params_4fac['lambda1_free_params']
        Lambda1_est_4fac[0,0]=1; Lambda1_est_4fac[1,0]=lambda_params[0]; Lambda1_est_4fac[2,0]=lambda_params[1]
        Lambda1_est_4fac[3,1]=1; Lambda1_est_4fac[4,1]=lambda_params[2]; Lambda1_est_4fac[5,1]=lambda_params[3]
        Lambda1_est_4fac[6,2]=1; Lambda1_est_4fac[7,2]=lambda_params[4]; Lambda1_est_4fac[8,2]=lambda_params[5]
        Lambda1_est_4fac[9,3]=1; Lambda1_est_4fac[10,3]=lambda_params[6]; Lambda1_est_4fac[11,3]=lambda_params[7]
        
        print("True B1 (State 1):\n", B1_true_state1.cpu().numpy())
        print("Estimated B1 (4-Factor):\n", B1_est_4fac.cpu().numpy())
        print("---")
        print("True Lambda1 (State 1):\n", Lambda1_true_state1.cpu().numpy())
        print("Estimated Lambda1 (4-Factor):\n", Lambda1_est_4fac.cpu().numpy())

    # --- 2因子モデルの最良結果 ---
    if best_params_2fac:
        print("\n--- Best 2-Factor Model Estimation Results ---")
        print(f"Best Negative Log-Likelihood (Loss): {best_loss_2fac:.4f}")
        
        Lambda1_est_2fac = torch.zeros(O, L1_2fac)
        lambda_params_2fac = best_params_2fac['lambda1_free_params']
        Lambda1_est_2fac[0,0]=1
        Lambda1_est_2fac[1:9, 0] = lambda_params_2fac[0:8]
        Lambda1_est_2fac[9, 1] = 1
        Lambda1_est_2fac[10:12, 1] = lambda_params_2fac[8:10]
        
        print("True B1 (State 2):\n", B1_true_state2.cpu().numpy())
        print("Estimated B1 (2-Factor):\n", best_params_2fac['B1'].cpu().numpy())
        print("---")
        print("True Lambda1 (State 2):\n", Lambda1_true_state2.cpu().numpy())
        print("Estimated Lambda1 (2-Factor):\n", Lambda1_est_2fac.cpu().numpy())

    # --- モデル適合度の比較 ---
    if best_params_4fac and best_params_2fac:
        print("\n--- Model Fit Comparison ---")
        print(f"4-Factor Model Best Loss: {best_loss_4fac:.4f}")
        print(f"2-Factor Model Best Loss: {best_loss_2fac:.4f}")
        if best_loss_4fac < best_loss_2fac:
            print("The 4-Factor model provides a better fit to the data based on log-likelihood.")
        elif best_loss_2fac < best_loss_4fac:
            print("The 2-Factor model provides a better fit to the data based on log-likelihood.")
        else:
            print("The models have comparable fit based on log-likelihood.")

# `with`ブロックを抜けると、Teeクラスのcloseメソッドが自動的に呼ばれ、
# 出力先が元のコンソールに戻り、ファイルが閉じられます。