from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =============== 1. 讀取資料，與訓練時一致 ===============
file_path = 'IMPACT.sensors.csv'
data = pd.read_csv(file_path)

timestamp_column = 'createdAt'
data[timestamp_column] = pd.to_datetime(data[timestamp_column]).dt.tz_localize(None)
data = data.sort_values(by=timestamp_column).reset_index(drop=True)

features = ['temperature', 'hour', 'day_of_week', 'is_weekend']

# 加入時間特徵
data['hour'] = data[timestamp_column].dt.hour
data['day_of_week'] = data[timestamp_column].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# 載入scaler
scaler = joblib.load('temperature_scaler.pkl')

# 載入模型
model = load_model('temperature_prediction_gru_model.keras')

# 指定目標時間點
target_time_str = '2025-03-03 16:00:00'
target_time = pd.to_datetime(target_time_str)

# 建立序列
sequence_length = 12
data_before = data[data[timestamp_column] <= target_time].reset_index(drop=True)

if len(data_before) < sequence_length:
    raise ValueError(f"資料在 {target_time_str} 之前不足 {sequence_length} 筆，無法建立序列")

last_seq = data_before.iloc[-sequence_length:][features].values

# Normalize
last_seq_scaled = scaler.transform(last_seq)
X_input = np.expand_dims(last_seq_scaled, axis=0)

# 預測
prediction_steps = [6, 12, 18, 24, 30, 36, 42, 48]
y_pred_scaled = model.predict(X_input)
y_pred_real = scaler.inverse_transform(
    np.hstack([y_pred_scaled.reshape(-1, 1), np.zeros((y_pred_scaled.shape[1], len(features)-1))])
)[:, 0]

print(f"\n=== 預測: 在 {target_time_str} 之後的未來 Temperature 平均 ===")
for i, step in enumerate(prediction_steps):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60
    print(f"未來 {hours}h{minutes:02d}min: Temperature ≈ {y_pred_real[i]:.4f}")

# 實際觀測比較
for i, step in enumerate(prediction_steps):
    row_target = data[data[timestamp_column] <= target_time].index[-1]
    start_idx = row_target + step
    end_idx = start_idx + 6

    if end_idx < len(data):
        real_avg = data.iloc[start_idx:end_idx]['temperature'].mean()
        hours = (step * 10) // 60
        minutes = (step * 10) % 60
        print(f"真實 {hours}h{minutes:02d}min 平均 Temperature = {real_avg:.2f} (預測 {y_pred_real[i]:.2f})")
    else:
        print(f"無足夠資料可比較: {hours}h{minutes:02d}min 之後的 6 筆觀測缺失.")