# prepare_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# 1️⃣ Baca file CSV
df = pd.read_csv("apex.csv")  # ganti dengan nama file kamu kalau beda

# 2️⃣ Ubah kolom waktu jadi tipe datetime dan jadikan index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 3️⃣ Gunakan hanya kolom 'close'
df = df[['close']]

# 4️⃣ Visualisasi awal — biar kita bisa lihat grafik pergerakan harga
plt.figure(figsize=(12, 5))
plt.plot(df['close'])
plt.title('Pergerakan Harga Saham Harian')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (close)')
plt.grid(True)
plt.show()

# 5️⃣ Normalisasi data (biar skalanya 0–1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Simpan scaler supaya nanti bisa dipakai saat prediksi
joblib.dump(scaler, "scaler.save")

# 6️⃣ Buat data sekuens (windowing)
timesteps = 60  # artinya kita pakai 60 hari terakhir untuk prediksi hari berikutnya
X, y = [], []

for i in range(timesteps, len(scaled_data)):
    X.append(scaled_data[i - timesteps:i, 0])  # 60 data sebelumnya
    y.append(scaled_data[i, 0])                # data ke-61 (target)

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # ubah ke format (samples, timesteps, features)

print("✅ Data sudah siap untuk dilatih!")
print("Bentuk X:", X.shape)
print("Bentuk y:", y.shape)
