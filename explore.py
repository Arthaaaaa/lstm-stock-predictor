# explore.py
import pandas as pd

# Ganti nama file kalau beda
df = pd.read_csv("apex.csv")

# Lihat 5 baris pertama
print("5 baris pertama:")
print(df.head())

# Lihat nama-nama kolom
print("\nKolom-kolom dataset:")
print(df.columns)

# Informasi umum
print("\nInfo singkat:")
print(df.info())

# Cek ada nilai kosong atau tidak
print("\nJumlah nilai kosong per kolom:")
print(df.isnull().sum())
