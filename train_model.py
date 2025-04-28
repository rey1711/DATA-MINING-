# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load Data
# Ganti ini dengan path datasetmu
data = pd.read_csv('D:\FASTAPI\final data.csv')  # <--- GANTI nama file dataset kamu di sini

# 2. Preprocessing
# Misal memilih beberapa fitur
X = data[['Prevalence Rate (%)', 'Incidence Rate (%)', 'Mortality Rate (%)']]
y = data['Target Prevalence']  # <--- GANTI kolom target prediksi kamu

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Train Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6. Save Model dan Scaler
joblib.dump(rf_model, 'model.pkl', compress=3)
joblib.dump(scaler, 'scaler.pkl', compress=3)

print("✅ Model dan Scaler berhasil disimpan sebagai model.pkl dan scaler.pkl!")
