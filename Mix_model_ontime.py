import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

# =============== 1. 從 MongoDB 讀取最新 12 筆資料 ===============
mongo_uri = "mongodb://dmcl:unigrid@10.141.52.16:27017/?authMechanism=DEFAULT"
db_name = "IMPACT"
collection_name = "sensors"

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# 讀取最新的 12 筆數據
data_cursor = collection.find().sort("createdAt", -1).limit(12)
data_list = list(data_cursor)

# 轉換為 DataFrame
data = pd.DataFrame(data_list)

# 選擇需要的欄位（去掉 _id 和 __v）
columns_to_display = ["createdAt", "temperature", "pH", "conductivity", "oxygen", "ppm", "pm25"]
data_selected = data[columns_to_display]

# 顯示從 MongoDB 讀取的 12 筆數據
print("\n=== 從 MongoDB 讀取的 12 筆數據 ===")
print(data_selected)

# 產生時間特徵
data['hour'] = pd.to_datetime(data['createdAt']).dt.hour
data['day_of_week'] = pd.to_datetime(data['createdAt']).dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

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

# =============== 3. 進行預測 ===============
predictions = {}

for key, info in models.items():
    features = info['features']
    scaler = info['scaler']
    model = info['model']

    # 取最近 12 筆數據
    last_seq = data.iloc[-12:][features].values

    # 標準化
    if key != 'temperature':  # temperature 用固定的 scaler
        scaler.fit(data[features])
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
scaler_filename = "tds_dissolved_solid_conductivity_scaler.pkl"
scaler = joblib.load(scaler_filename)
model = load_model('tds_dissolved_solid_conductivity_gru_model.keras')

last_seq = data.iloc[-12:][features].values
last_seq_scaled = scaler.transform(last_seq)
X_input = np.expand_dims(last_seq_scaled, axis=0)

# 預測未來 1~8 小時的 conductivity 和 ppm
y_pred_scaled = model.predict(X_input)
y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 2))

predictions['conductivity_ppm'] = y_pred_real

# =============== 5. 顯示預測結果 ===============
print("\n=== 預測結果 ===")
for key in models.keys():
    print(f"\n=== {key.capitalize()} 預測結果 ===")
    for i, pred in enumerate(predictions[key]):
        print(f"預測 {(i+1) * 1} 小時後的 {key.capitalize()} = {pred:.2f}")

# 顯示 Conductivity 和 PPM 預測結果
prediction_steps = [6, 12, 18, 24, 30, 36, 42, 48]
print("\n=== Conductivity & PPM 預測結果 ===")
for i, (pred_conductivity, pred_ppm) in enumerate(predictions['conductivity_ppm']):
    hours = (prediction_steps[i] * 10) // 60
    minutes = (prediction_steps[i] * 10) % 60
    print(f"未來 {hours}h{minutes:02d}min: Conductivity ≈ {pred_conductivity:.2f}, PPM ≈ {pred_ppm:.2f}")