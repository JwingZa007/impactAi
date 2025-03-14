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

for key, info in models.items():
    features = info['features']
    scaler = info['scaler']
    model = info['model']

    # 取最近 SEQUENCE_LENGTH 筆數據
    last_seq = data.iloc[-SEQUENCE_LENGTH:][features].values

    # 標準化
    if key != 'temperature':  # temperature 用固定的 scaler
        scaler.fit(data[features])  # 避免測試時數據範圍不同
    last_seq_scaled = scaler.transform(pd.DataFrame(last_seq, columns=features))

    # 進行預測
    X_input = np.expand_dims(last_seq_scaled, axis=0)
    y_pred_scaled = model.predict(X_input)

    # 反標準化
    y_pred_real = scaler.inverse_transform(
        np.column_stack([y_pred_scaled.flatten()] + [np.zeros_like(y_pred_scaled.flatten())] * (len(features) - 1))
    )[:, 0].flatten()

    predictions[key] = y_pred_real

# =============== 4. 進行 TDS (Conductivity & PPM) 預測 ===============
features = ['conductivity', 'ppm']
scaler = MinMaxScaler()
scaler.fit(data[features])

model = load_model('tds_dissolved_solid_conductivity_gru_model.keras')

# 取最近 SEQUENCE_LENGTH 筆數據
last_seq = data.iloc[-SEQUENCE_LENGTH:][features].values
last_seq_scaled = scaler.transform(last_seq)
X_input = np.expand_dims(last_seq_scaled, axis=0)

y_pred_scaled = model.predict(X_input)
y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 2))

predictions['conductivity_ppm'] = y_pred_real

# =============== 5. 顯示預測結果 ===============
print(f"\n=== 預測結果 ===")
for key in models.keys():
    print(f"\n=== {key.capitalize()} 預測結果 ===")
    for i, step in enumerate(PREDICTION_STEPS):
        hours = (step * 10) // 60
        minutes = (step * 10) % 60
        pred = predictions[key][i]
        print(f"未來 {hours}h{minutes:02d}min: {key.capitalize()} ≈ {pred:.2f}")

print(f"\n=== Conductivity & PPM 預測結果 ===")
for i, step in enumerate(PREDICTION_STEPS):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60
    pred_conductivity, pred_ppm = predictions['conductivity_ppm'][i]
    print(f"未來 {hours}h{minutes:02d}min: Conductivity ≈ {pred_conductivity:.2f}, PPM ≈ {pred_ppm:.2f}")
