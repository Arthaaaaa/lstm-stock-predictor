# train_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Muat dataset
df = pd.read_csv("apex.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df[['close']]

# 2️⃣ Normalisasi ulang (biar aman, nanti hasilnya sama dengan scaler.save)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# 3️⃣ Buat data sekuens (sama seperti di prepare_data.py)
timesteps = 60
X, y = [], []

for i in range(timesteps, len(scaled_data)):
    X.append(scaled_data[i - timesteps:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4️⃣ Pisahkan data: 80% untuk training, 20% untuk testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Data train:", X_train.shape, " | Data test:", X_test.shape)

# 5️⃣ Bangun arsitektur model LSTM
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# 6️⃣ Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')

# 7️⃣ Latih model (ini bagian yang butuh waktu beberapa menit)
history = model.fit(
    X_train, y_train,
    epochs=10,             # bisa kamu ubah jadi 10 kalau mau cepat
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 8️⃣ Simpan model dan scaler
model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.save")

print("\n✅ Model LSTM sudah selesai dilatih dan disimpan!")

# 9️⃣ Prediksi hasil
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# 10️⃣ Kembalikan nilai aktual ke skala aslinya juga
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 11️⃣ Hitung RMSE (seberapa jauh prediksi dari nilai asli)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f"📉 Nilai RMSE: {rmse:.2f}")

# 12️⃣ Visualisasi hasil
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, color='blue', label='Harga Aktual')
plt.plot(predictions, color='orange', label='Harga Prediksi')
plt.title('Prediksi Harga Saham dengan LSTM')
plt.xlabel('Hari')
plt.ylabel('Harga Penutupan')
plt.legend()
plt.grid(True)
plt.show()
