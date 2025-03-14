import pandas as pd
import numpy as np
import joblib  # 用於載入 MinMaxScaler
from tensorflow.keras.models import load_model

# =============== 1. 讀取資料，與訓練時一致 ===============
file_path = 'IMPACT.sensors.csv'
data = pd.read_csv(file_path)

timestamp_column = 'createdAt'
data[timestamp_column] = pd.to_datetime(data[timestamp_column])

# **強制移除時區，使其變為 tz-naive**
if data[timestamp_column].dt.tz is not None:
    data[timestamp_column] = data[timestamp_column].dt.tz_localize(None)

# 依時間排序
data = data.sort_values(by=timestamp_column).reset_index(drop=True)

# 只保留 conductivity 和 ppm 欄位
features = ['conductivity', 'ppm']
data_features = data[features]

# =============== 2. 載入 Scaler（確保與訓練時相同） ===============
scaler_filename = "tds_dissolved_solid_conductivity_scaler.pkl"
scaler = joblib.load(scaler_filename)
print("Scaler loaded successfully!")

# 進行標準化 (使用訓練時的 Scaler)
data_normalized = scaler.transform(data_features)

# =============== 3. 載入已訓練模型 ===============
model = load_model('tds_dissolved_solid_conductivity_gru_model.keras')
print("Model loaded successfully!")

# =============== 4. 指定目標時間點 ===============
target_time_str = '2025-03-03 00:00:00'
target_time = pd.to_datetime(target_time_str).tz_localize(None)

# 找到「在 target_time 之前」的資料
mask = (data[timestamp_column] <= target_time)
data_before = data[mask].reset_index(drop=True)

# 檢查是否至少有 12 筆（sequence_length = 12）
sequence_length = 12
if len(data_before) < sequence_length:
    raise ValueError(f"資料在 {target_time_str} 之前不足 {sequence_length} 筆，無法建立序列")

# 取最後 12 筆 (對應 sequence_length=12)
last_seq = data_before.iloc[-sequence_length:][features].values

# =============== 5. Normalize 數據 (與訓練相同) ===============
last_seq_scaled = scaler.transform(last_seq)

# Reshape 成 GRU 輸入格式
X_input = np.expand_dims(last_seq_scaled, axis=0)

# =============== 6. 預測未來 1~8 小時的 conductivity 和 ppm ===============
y_pred_scaled = model.predict(X_input)

# 反標準化
y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 2))

# =============== 7. 印出預測結果 ===============
prediction_steps = [6, 12, 18, 24, 30, 36, 42, 48]
print(f"\n=== 預測: 在 {target_time_str} 之後的未來 conductivity & ppm 平均 ===")
for i, step in enumerate(prediction_steps):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60
    print(f"未來 {hours}h{minutes:02d}min: conductivity ≈ {y_pred_real[i, 0]:.2f}, ppm ≈ {y_pred_real[i, 1]:.2f}")

# =============== 8. 與「實際觀測」比較 (若有未來真實數據) ===============
for i, step in enumerate(prediction_steps):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60

    # 觀測區段 (step ~ step+6)
    start_idx = len(data_before) + step
    end_idx   = start_idx + 6

    if end_idx <= len(data):
        real_conductivity = data.iloc[start_idx:end_idx]['conductivity'].mean()
        real_ppm = data.iloc[start_idx:end_idx]['ppm'].mean()
        print(f"真實 {hours}h{minutes:02d}min 平均 conductivity = {real_conductivity:.2f}, ppm = {real_ppm:.2f} (預測: {y_pred_real[i, 0]:.2f}, {y_pred_real[i, 1]:.2f})")
    else:
        print(f"無足夠資料可比較: {hours}h{minutes:02d}min 之後的觀測缺失.")
