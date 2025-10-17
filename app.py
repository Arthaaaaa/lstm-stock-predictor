from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load model dan data
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")
df = pd.read_csv("apex.csv")

# Fungsi untuk menghasilkan grafik
def plot_to_base64(predicted_price=None):
    plt.figure(figsize=(10, 4))
    plt.plot(df['close'], label='Harga Aktual', color='blue')

    if predicted_price is not None:
        plt.axvline(df.index[-1], color='gray', linestyle='--')
        plt.scatter(len(df)-1, df['close'].values[-1], color='green', label='Harga Terakhir')
        plt.scatter(len(df), predicted_price, color='orange', label='Prediksi Besok')

    plt.title('Prediksi Harga Saham (APEX)')
    plt.xlabel('Hari')
    plt.ylabel('Harga Penutupan')
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return plot_data


# Halaman utama
@app.route("/")
def index():
    plot_data = plot_to_base64()
    return render_template("index.html", plot_url=plot_data)


# Endpoint tombol prediksi
@app.route("/predict", methods=["POST"])
def predict():
    last_60 = df['close'].values[-60:].reshape(-1, 1)
    scaled_last_60 = scaler.transform(last_60)
    X_input = np.reshape(scaled_last_60, (1, scaled_last_60.shape[0], 1))
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    plot_data = plot_to_base64(predicted_price)
    return jsonify({
        "predicted_price": round(float(predicted_price), 2),
        "plot_url": plot_data
    })


if __name__ == "__main__":
    app.run(debug=True)
