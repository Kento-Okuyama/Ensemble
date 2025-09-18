import subprocess
import os
import sys

# 実行するデータモードとリアライゼーション数を定義
DGP_MODES = ['IMPLEMENT']
NUM_REALIZATIONS = 1

# コントロールスクリプトのログ出力を設定
log_filename = 'run_experiments_log.txt'
original_stdout = sys.stdout
sys.stdout = open(log_filename, 'w', encoding='utf-8')

print("--- Starting Automated Experiment Runs ---")
print(f"Executing {NUM_REALIZATIONS} realizations for each mode: {DGP_MODES}")

# スクリプトの相対パス
src_path = 'src/'

for mode in DGP_MODES:
    for i in range(1, NUM_REALIZATIONS + 1):
        print(f"\n[REALIZATION {i}/{NUM_REALIZATIONS}] Running scripts for mode: {mode}")

        # 各スクリプトに渡す引数をリストで定義
        dgp_args = ['python', f'{src_path}dgp.py', '-m', mode, '-r', str(i), '-s', str(i)]
        var_args = ['python', f'{src_path}var.py', '-m', mode, '-r', str(i), '-s', str(i)]
        frs_args = ['python', f'{src_path}frs.py', '-m', mode, '-r', str(i), '-s', str(i)]
        bps_args = ['python', f'{src_path}bps.py', '-m', mode, '-r', str(i), '-s', str(i)]

        # --- ファイルパスの動的な設定 (各スクリプトの出力パスと一致させる) ---
        mode_dir = mode.replace(' 2.0', '_2.0')
        # ★★★ 修正: 出力パスの定義を一貫させる ★★★
        # ここで定義したパスが、各スクリプトの出力パスと完全に一致する必要があります
        # ★★★ 全ての出力パスを統一されたルールで定義 ★★★
        dgp_output_file = os.path.join('project/data', mode_dir, f'dgp_data_run_{i}', f'simulation_data_{mode}.pt')
        var_output_file = os.path.join('project/results', mode_dir, f'var_results_run_{i}', f'fitted_3fac_model_{mode}.pt')
        frs_output_file = os.path.join('project/results', mode_dir, f'frs_results_run_{i}', f'frs_model_results_{mode}.pt')
        bps_output_file = os.path.join('project/results', mode_dir, f'bps_results_run_{i}', f'bps_mcmc_results_{mode}.pt')
        
        # --- 実行ロジック ---
        try:
            # 1. DGPスクリプトの実行をチェック
            if not os.path.exists(dgp_output_file):
                print(f"  -> Running dgp.py with seed {i}...")
                subprocess.run(dgp_args, check=True)
            else:
                print(f"  -> dgp.py for realization {i} already exists. Skipping...")

            # 2. VARスクリプトの実行をチェック
            if not os.path.exists(var_output_file):
                print(f"  -> Running var.py with seed {i}...")
                subprocess.run(var_args, check=True)
            else:
                print(f"  -> var.py for realization {i} already exists. Skipping...")

            # 3. FRSスクリプトの実行をチェック
            if not os.path.exists(frs_output_file):
                print(f"  -> Running frs.py with seed {i}...")
                subprocess.run(frs_args, check=True)
            else:
                print(f"  -> frs.py for realization {i} already exists. Skipping...")

            # 4. BPSスクリプトの実行をチェック
            if not os.path.exists(bps_output_file):
                print(f"  -> Running bps.py with seed {i}...")
                subprocess.run(bps_args, check=True)
            else:
                print(f"  -> bps.py for realization {i} already exists. Skipping...")
            
            print(f"  [SUCCESS] Realization {i} for {mode} checked.")

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Script failed with exit code {e.returncode}. Continuing to next realization.")
            continue  # エラーが発生しても中断せず、次のリアライゼーションに進む

print("\n--- All Experiment Runs Finished ---")

# ログファイルを閉じる
sys.stdout.close()
sys.stdout = original_stdout
print(f"Automated experiment log saved to '{log_filename}'")