import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =============== 1. Load Data and Preprocess ===============
file_path = 'IMPACT.sensors.csv'
data = pd.read_csv(file_path)

timestamp_column = 'createdAt'
data[timestamp_column] = pd.to_datetime(data[timestamp_column])

# Remove timezone if any (ensure tz-naive for consistency)
if data[timestamp_column].dt.tz is not None:
    data[timestamp_column] = data[timestamp_column].dt.tz_localize(None)

# Sort by timestamp
data = data.sort_values(by=timestamp_column).reset_index(drop=True)

# Select relevant features (ensure consistency with training)
features = ['pH', 'temperature', 'hour', 'day_of_week', 'is_weekend']
data['hour'] = data[timestamp_column].dt.hour
data['day_of_week'] = data[timestamp_column].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

data_features = data[features]

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_features)

# =============== 2. Load Trained Model ===============
model = load_model('ph_prediction_gru_model.keras')

# =============== 3. Select Target Timestamp ===============
target_time_str = '2025-02-03 08:00:00'
target_time = pd.to_datetime(target_time_str).tz_localize(None)

# Get data up to target_time
mask = (data[timestamp_column] <= target_time)
data_before = data[mask].reset_index(drop=True)

# Ensure enough data points
sequence_length = 12
if len(data_before) < sequence_length:
    raise ValueError(f"Not enough data before {target_time_str} (required {sequence_length} samples)")

# Get last 12 time steps
last_seq = data_before.iloc[-sequence_length:][features].values
last_seq_scaled = scaler.transform(last_seq)  # Normalize

# Reshape for GRU input
X_input = np.expand_dims(last_seq_scaled, axis=0)

# =============== 4. Predict Future pH Values ===============
prediction_steps = [6, 12, 18, 24, 30, 36, 42, 48]  # Future time points
y_pred_scaled = model.predict(X_input)

# Convert back to original scale (only pH)
y_pred_real = scaler.inverse_transform(
    np.column_stack([y_pred_scaled.flatten()] + [np.zeros_like(y_pred_scaled.flatten())] * (len(features) - 1))
)[:, 0]

# =============== 5. Print Predictions ===============
print(f"\n=== Predicted pH Values After {target_time_str} ===")
for i, step in enumerate(prediction_steps):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60
    print(f"Future {hours}h{minutes:02d}min: pH ≈ {y_pred_real[i]:.2f}")

# =============== 6. Compare with Actual Observations ===============
data_after = data[data[timestamp_column] > target_time].reset_index(drop=True)

for i, step in enumerate(prediction_steps):
    hours = (step * 10) // 60
    minutes = (step * 10) % 60

    row_target = data.index[data[timestamp_column] <= target_time][-1]
    start_idx = row_target + step
    end_idx = start_idx + 6  # Take 6 future observations

    if end_idx < len(data):
        real_avg = data.iloc[start_idx:end_idx][['pH']].values.mean()
        print(f"Actual {hours}h{minutes:02d}min pH = {real_avg:.2f} (Predicted: {y_pred_real[i]:.2f})")
    else:
        print(f"Insufficient data for {hours}h{minutes:02d}min observation.")