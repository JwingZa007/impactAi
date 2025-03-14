import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =============== 1. 讀取資料並預處理 ===============
file_path = 'IMPACT.sensors.csv'
data = pd.read_csv(file_path)

timestamp_column = 'createdAt'
data[timestamp_column] = pd.to_datetime(data[timestamp_column])
if data[timestamp_column].dt.tz is not None:
    data[timestamp_column] = data[timestamp_column].dt.tz_localize(None)

data = data.sort_values(by=timestamp_column).reset_index(drop=True)

# 定義目標時間
TARGET_TIME_STR = '2025-03-03 00:00:00'
target_time = pd.to_datetime(TARGET_TIME_STR).tz_localize(None)

# 定義時間步長
SEQUENCE_LENGTH = 12
PREDICTION_STEPS = [6, 12, 18, 24, 30, 36, 42, 48]

# =============== 2. 定義模型與特徵 ===============
models = {
    'temperature': {
        'features': ['temperature', 'hour', 'day_of_week', 'is_weekend'],
        'scaler': joblib.load('temperature_scaler.pkl'),
        'model': load_model('temperature_prediction_gru_model.keras')
    },
    'pH': {
        'features': ['pH', 'temperature', 'hour', 'day_of_week', 'is_weekend'],
        'scaler': MinMaxScaler(),
        'model': load_model('ph_prediction_gru_model.keras')
    },
    'oxygen': {
        'features': ['oxygen'],
        'scaler': MinMaxScaler(),
        'model': load_model('oxygen_prediction_gru_model.keras')
    },
    'pm25': {
        'features': ['pm25'],
        'scaler': MinMaxScaler(),
        'model': load_model('pm25_prediction_gru_model.keras')
    }
}

# 產生時間特徵
data['hour'] = data[timestamp_column].dt.hour
data['day_of_week'] = data[timestamp_column].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# =============== 3. 進行預測 ===============
predictions = {}
actual_values = {}

for key, info in models.items():
    features = info['features']
    scaler = info['scaler']
    model = info['model']

    # 取得目標時間前的數據
    data_before = data[data[timestamp_column] <= target_time].reset_index(drop=True)
    if len(data_before) < SEQUENCE_LENGTH:
        raise ValueError(f"{key}: 資料不足 {SEQUENCE_LENGTH} 筆，無法建立序列")

    # 取最近 SEQUENCE_LENGTH 筆數據
    last_seq = data_before.iloc[-SEQUENCE_LENGTH:][features].values

    # 標準化
    if key != 'temperature':  # temperature 用固定的 scaler
        scaler.fit(data_before[features])  # 避免測試時數據範圍不同
    last_seq_scaled = scaler.transform(pd.DataFrame(last_seq, columns=features))

    # 進行預測
    X_input = np.expand_dims(last_seq_scaled, axis=0)
    y_pred_scaled = model.predict(X_input)

    print(f"DEBUG: {key} 模型輸出形狀: {y_pred_scaled.shape}")

    # 反標準化
    y_pred_real = scaler.inverse_transform(
        np.column_stack([y_pred_scaled.flatten()] + [np.zeros_like(y_pred_scaled.flatten())] * (len(features) - 1))
    )[:, 0].flatten()

    predictions[key] = y_pred_real

    # 取得實際觀測值
    actual_values[key] = []
    for step in PREDICTION_STEPS:
        start_idx = len(data_before) + step
        end_idx = start_idx + 6
        if end_idx <= len(data):
            real_avg = data.iloc[start_idx:end_idx][features[0]].mean()
            actual_values[key].append(real_avg)
        else:
            actual_values[key].append(None)

# =============== 4. 進行 TDS (Conductivity & PPM) 預測 ===============
features = ['conductivity', 'ppm']
scaler = MinMaxScaler()
scaler.fit(data[features])

model = load_model('tds_dissolved_solid_conductivity_gru_model.keras')

data_before = data[data[timestamp_column] <= target_time].reset_index(drop=True)

if len(data_before) < SEQUENCE_LENGTH:
    raise ValueError(f"TDS: 資料不足 {SEQUENCE_LENGTH} 筆，無法建立序列")

last_seq = data_before.iloc[-SEQUENCE_LENGTH:][features].values
last_seq_scaled = scaler.transform(last_seq)
X_input = np.expand_dims(last_seq_scaled, axis=0)

y_pred_scaled = model.predict(X_input)
y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 2))

predictions['conductivity_ppm'] = y_pred_real

# 計算 TDS 觀測值
actual_values['conductivity_ppm'] = []
for step in PREDICTION_STEPS:
    start_idx = len(data_before) + step
    end_idx = start_idx + 6
    if end_idx <= len(data):
        real_conductivity = data.iloc[start_idx:end_idx]['conductivity'].mean()
        real_ppm = data.iloc[start_idx:end_idx]['ppm'].mean()

        # 確保 NaN 轉換為 None，避免格式化錯誤
        real_conductivity = None if pd.isna(real_conductivity) else real_conductivity
        real_ppm = None if pd.isna(real_ppm) else real_ppm

        actual_values['conductivity_ppm'].append((real_conductivity, real_ppm))
    else:
        actual_values['conductivity_ppm'].append(None)

# =============== 5. 顯示結果 ===============
print(f"\n=== 預測與真實比較 (時間: {TARGET_TIME_STR}) ===")
for key in models.keys():
    print(f"\n=== {key.capitalize()} 預測結果 ===")
    for i, step in enumerate(PREDICTION_STEPS):
        hours = (step * 10) // 60
        minutes = (step * 10) % 60
        pred = predictions[key][i]
        real = actual_values[key][i]

        real_str = f"{real:.2f}" if real is not None else "N/A"
        print(f"真實 {hours}h{minutes:02d}min 平均 {key.capitalize()} = {real_str} (預測 {pred:.2f})")

print(f"\n=== Conductivity & PPM 預測結果 ===")
for i, step in enumerate(PREDICTION_STEPS):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60
    pred_conductivity, pred_ppm = predictions['conductivity_ppm'][i]
    real_conductivity, real_ppm = actual_values['conductivity_ppm'][i] if actual_values['conductivity_ppm'][i] is not None else (None, None)

    if end_idx <= len(data):
        real_conductivity = data.iloc[start_idx:end_idx]['conductivity'].mean()
        real_ppm = data.iloc[start_idx:end_idx]['ppm'].mean()

        # 確保 NaN 轉換為 None，避免格式化錯誤
        real_conductivity = None if pd.isna(real_conductivity) else real_conductivity
        real_ppm = None if pd.isna(real_ppm) else real_ppm

        real_conductivity_str = f"{real_conductivity:.2f}" if real_conductivity is not None else "N/A"
        real_ppm_str = f"{real_ppm:.2f}" if real_ppm is not None else "N/A"

        print(
            f"真實 {hours}h{minutes:02d}min 平均 Conductivity = {real_conductivity_str} (預測 {y_pred_real[i, 0]:.2f})")
        print(f"真實 {hours}h{minutes:02d}min 平均 PPM = {real_ppm_str} (預測 {y_pred_real[i, 1]:.2f})")
    else:
        print(f"無足夠資料可比較: {hours}h{minutes:02d}min 之後的觀測缺失.")
