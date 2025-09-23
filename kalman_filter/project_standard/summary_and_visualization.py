import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- 基本設定 ---
# 解析対象のモードとデータ数を指定
DGP_MODE = 'IMPLEMENT'
NUM_REALIZATIONS = 30

# --- ディレクトリ設定 ---
# こちらのパスはご自身の環境に合わせて適宜変更してください
BASE_RESULTS_DIR = 'results' 
MODE_DIR = DGP_MODE.replace(' 2.0', '_2.0')
SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, MODE_DIR, 'summary_digest')
os.makedirs(SUMMARY_DIR, exist_ok=True)

# --- モデル名のマッピング ---
# 凡例に表示する際のモデル名を定義
MODEL_NAME_MAP = {
    'BPS': 'Bayesian Stacking',
    'FRS': 'Mixture Modeling',
    'VAR_3fac': 'VAR(3)',
    'VAR_1fac': 'VAR(1)'
}

print(f"--- スクリプトを開始します ---")
print(f"モード: {DGP_MODE}")
print(f"保存先: {SUMMARY_DIR}")

# --- データ収集用の関数 ---
def parse_log_file(log_path):
    """
    ログファイルから評価指標とモデルパラメータ（B1行列）を抽出する関数。
    """
    results = {}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return results

    # --- 評価指標の抽出 ---
    metric_patterns = {
        'Likelihood': r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)", # ★★★ この行を追加 ★★★
        'Sensitivity': r"State Detection Sensitivity\s*\|\s*(\d+\.\d+)",
        'Specificity': r"State Detection Specificity\s*\|\s*(\d+\.\d+)",
        'Duration': r"Training Duration \(s\)\s*\|\s*(\d+\.\d+)"
    }
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))

    # --- パラメータ抽出用の内部関数 ---
    def parse_matrix(matrix_str):
        cleaned_rows = [row.strip().replace('[', '').replace(']', '') for row in matrix_str.strip().split('\n')]
        return np.array([[float(v) for v in row.split()] for row in cleaned_rows if row])

    # --- B1行列の抽出 (FRSモデル) ---
    if 'frs_log' in log_path:
        if (m := re.search(r"-- Estimated B1 \(3-Factor\) --\s*\n(.*?)\n\n-- Estimated B1 \(1-Factor\) --", content, re.DOTALL)):
            results['B1_3fac'] = parse_matrix(m.group(1))
        if (m := re.search(r"-- Estimated B1 \(1-Factor\) --\s*\n\s*\[\[(.*?)\]\]", content, re.DOTALL)):
            results['B1_1fac'] = np.array([[float(m.group(1))]])
    
    # --- B1行列の抽出 (VARモデル) ---
    elif 'var_log' in log_path:
        try:
            # 3-Factorと1-Factorのセクションに分割
            s3_content = content.split("--- 3-Factor Model")[1].split("--- 1-Factor Model")[0]
            s1_content = content.split("--- 1-Factor Model")[1]
            
            if (m := re.search(r"-- Estimated B1 \(State Transition\) --\s*\n(.*?)\n-- True B1 --", s3_content, re.DOTALL)):
                results['B1_3fac'] = parse_matrix(m.group(1))
            if (m := re.search(r"-- Estimated B1 \(State Transition\): --\s*\n\s*\[\[(.*?)\]\]", s1_content, re.DOTALL)):
                results['B1_1fac'] = np.array([[float(m.group(1))]])
        except IndexError:
            pass # VARログの形式が異なる場合はスキップ

    return results

# --- データ収集メインループ ---
print("\n--- 1. 全リアライゼーションのデータを収集中... ---")
all_data = []
for i in range(1, NUM_REALIZATIONS + 1):
    # 各モデルのログファイルパスを定義
    log_files = {
        'FRS': os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'frs_results_run_{i}', f'frs_log_{DGP_MODE}_run_{i}.txt'),
        'BPS': os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'bps_results_run_{i}', f'bps_log_{DGP_MODE}_run_{i}.txt'),
        'VAR': os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'var_results_run_{i}', f'var_log_{DGP_MODE}_run_{i}.txt'),
    }

    # FRS, BPSのデータを収集 (Likelihoodを含む)
    for model in ['FRS', 'BPS']:
        # parse_log_fileはLikelihoodを収集するように事前修正済み
        parsed_data = parse_log_file(log_files[model])
        if parsed_data:
            parsed_data['realization'] = i
            parsed_data['model'] = model
            all_data.append(parsed_data)

    # ★★★ 変更点: VARの対数尤度を手動でパースする ★★★
    try:
        with open(log_files['VAR'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 3-Factorと1-Factorのセクションに分割
        s3_content = content.split("--- 3-Factor Model Performance Summary ---")[1].split("--- 1-Factor Model Performance Summary ---")[0]
        s1_content = content.split("--- 1-Factor Model Performance Summary ---")[1]

        # 共通のDurationを取得 (どちらのセクションからでも良い)
        duration_match = re.search(r"Training Duration \(s\)\s*\|\s*(\d+\.\d+)", content)
        duration = float(duration_match.group(1)) if duration_match else None

        # 3-Factorの尤度を抽出
        lik_3fac_match = re.search(r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)", s3_content)
        if lik_3fac_match:
            all_data.append({
                'realization': i, 'model': 'VAR_3fac', 'Duration': duration,
                'Likelihood': float(lik_3fac_match.group(1))
            })

        # 1-Factorの尤度を抽出
        lik_1fac_match = re.search(r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)", s1_content)
        if lik_1fac_match:
            all_data.append({
                'realization': i, 'model': 'VAR_1fac', 'Duration': duration,
                'Likelihood': float(lik_1fac_match.group(1))
            })
            
    except (FileNotFoundError, IndexError):
        print(f"警告: VARログの解析に失敗しました: {log_files['VAR']}")


df = pd.DataFrame(all_data)
df['model_label'] = df['model'].map(MODEL_NAME_MAP)
print("データ収集完了。")

# --- 真値の定義 ---
if DGP_MODE == 'IMPLEMENT 2.0':
    b1_3fac_true = np.array([[0.17, -0.06, 0.00], [0.14, 0.21, -0.10], [-0.29, -0.22, 0.11]])
    b1_1fac_true = np.array([[0.24]])
else: # デフォルトとして IMPLEMENT の値を設定
    b1_3fac_true = np.array([[0.04, 0.01, -0.11], [-0.01, 0.07, 0.13], [0.02, 0.11, 0.16]])
    b1_1fac_true = np.array([[0.50]])
    
print(f"'{DGP_MODE}' モードに対応する真値を設定しました。")

# --- スタイルとカラー設定 ---
sns.set_theme(style="whitegrid")
palette = {'Bayesian Stacking': 'steelblue', 'Mixture Modeling': 'lightcoral', 'VAR(3)': 'mediumseagreen', 'VAR(1)': 'goldenrod'}

# --- プロット①: Sensitivity / Specificity 比較 (ファセット表示版) ---
print("\n--- 2. プロット① (感度/特異度) を作成中... ---")
# Bayesian StackingとMixture Modelingのデータのみを抽出
plot_df = df[df['model_label'].isin(['Bayesian Stacking', 'Mixture Modeling'])]
plot_df = plot_df.dropna(subset=['Sensitivity', 'Specificity'])

# ファセット表示のためにデータを縦長形式（long format）に変換
melted_df = pd.melt(
    plot_df,
    id_vars=['model_label'],
    value_vars=['Sensitivity', 'Specificity'],
    var_name='Metric',
    value_name='Value'
)

# ★★★ 変更点: FacetGridの col と hue を入れ替え ★★★
# FacetGridを使用して、メトリックごとにプロットを分割
g = sns.FacetGrid(
    data=melted_df,
    col="Metric",           # 'Metric'（感度/特異度）で列を分割
    hue="model_label",      # 'model_label'で色分け
    palette=palette,
    height=5,
    aspect=1.2
)

# 各ファセットにKDEプロットを描画
g.map(sns.kdeplot, "Value", fill=True, alpha=0.6)

# 凡例のタイトルを変更し、デフォルトの位置（右側）に表示
g.add_legend(title='model_label')
# ★★★ 変更点: ここから ★★★
# 作成した凡例をプロットエリアの内側左上に移動
sns.move_legend(g, "upper left")
# ★★★ 変更点: ここまで ★★★
# タイトルと軸ラベルを設定
g.set_titles("{col_name}", size=14)
g.set_axis_labels("Value", "Density")
g.fig.suptitle('Sensitivity and Specificity Distributions', fontsize=18, y=0.95)
g.set(xlim=(0.5, 1.05))

# レイアウトを自動調整
plt.tight_layout()
save_path = os.path.join(SUMMARY_DIR, 'plot1_sensitivity_specificity.png')
plt.savefig(save_path)
plt.close()
print(f"プロットを保存しました: {save_path}")

# --- プロット②: Overall Log-Likelihood 比較 ---
print("\n--- 3. プロット② (対数尤度) を作成中... ---")

# 対数尤度のデータのみを抽出
likelihood_df = df.dropna(subset=['Likelihood'])
# 尤度の中央値でモデルをソート
order = sorted(
    likelihood_df['model_label'].unique(), 
    key=lambda x: likelihood_df[likelihood_df['model_label']==x]['Likelihood'].median(),
    reverse=True # 尤度は高い方が良いので降順
)

plt.figure(figsize=(12, 7))

# KDEプロットで分布を表示
sns.kdeplot(
    data=likelihood_df,
    x='Likelihood',
    hue='model_label',
    hue_order=order,
    palette=palette,
    fill=True,
    common_norm=False, # 各分布が個別に正規化されるようにする
    alpha=0.5
)

plt.title('Overall Log-Likelihood Comparison Across Models', fontsize=16)
plt.xlabel('Final Log-Likelihood', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()

save_path = os.path.join(SUMMARY_DIR, 'plot2_log_likelihood.png')
plt.savefig(save_path)
plt.close()
print(f"プロットを保存しました: {save_path}")

# --- プロット③: 計算時間の比較 (合計版) ---
print("\n--- 4. プロット③ (計算時間) を作成中 [合計版]... ---")

# 計算に必要なモデルのデータのみを抽出
duration_df = df.dropna(subset=['Duration'])

# データを横長形式に変換して、realizationごとに行をまとめる
pivot_df = duration_df.pivot_table(
    index='realization', 
    columns='model_label', 
    values='Duration'
)

# VAR(3), VAR(1), Bayesian Stackingの時間を合計して新しい列を作成
pivot_df['Bayesian Stacking'] = pivot_df['VAR(3)'] + pivot_df['VAR(1)'] + pivot_df['Bayesian Stacking']

# 比較対象の2つの列（新しいBSとMM）だけを選択
final_duration_df = pivot_df[['Bayesian Stacking', 'Mixture Modeling']]

# プロットしやすいようにデータを縦長形式に戻す
melted_final_df = final_duration_df.melt(var_name='model_label', value_name='Duration')

# KDEプロットを作成
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=melted_final_df, 
    x='Duration', 
    hue='model_label',
    fill=True,
    alpha=0.6,
    palette={'Bayesian Stacking': 'steelblue', 'Mixture Modeling': 'lightcoral'}
)

# タイトル、ラベル、x軸の範囲を設定
plt.title('Computation Time Comparison', fontsize=16)
plt.xlabel('Duration (seconds)')
plt.ylabel('Density')
plt.xlim(0, 1000) # x軸の範囲を調整
plt.tight_layout()

# 新しいファイル名で保存
save_path = os.path.join(SUMMARY_DIR, 'plot3_computation_time.png')
plt.savefig(save_path)
plt.close()
print(f"合計時間を反映したプロットを保存しました: {save_path}")