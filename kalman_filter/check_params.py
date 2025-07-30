import torch
import pprint

filename = 'bps_params.pt'
print(f"--- Loading '{filename}' ---")

try:
    loaded_data = torch.load(filename, weights_only=False)

    print("\n[1] ファイルの全体構造 (トップレベルのキー):")
    print(list(loaded_data.keys()))
    
    params = loaded_data['params']
    constraints = loaded_data['constraints']

    print("\n[2] 特定のパラメータの値を確認 (例: theta_3fac_loc):")
    theta_3fac_loc_tensor = params['theta_3fac_loc']
    print(f"  - パラメータ名: theta_3fac_loc")
    print(f"  - 形状 (Shape): {theta_3fac_loc_tensor.shape}")
    
    # === ▼▼▼ ここを修正 ▼▼▼ ===
    # .numpy() の前に .detach() を追加
    print(f"  - 値 (最初の5人分): \n{theta_3fac_loc_tensor[:5].detach().numpy().round(4)}")
    # === ▲▲▲ ここまで修正 ▲▲▲ ===

    print("\n[3] パラメータの制約情報を確認 (一部):")
    pprint.pprint({k: v for i, (k, v) in enumerate(constraints.items()) if i < 5})

except FileNotFoundError:
    print(f"エラー: ファイル '{filename}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")