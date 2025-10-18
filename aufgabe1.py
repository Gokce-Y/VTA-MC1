import numpy as np
import pandas as pd

# --- Veri ---
df = pd.read_csv("data/data_a1_mc1_vta_hs25.csv", sep=";")
x = df.iloc[:, 0].to_numpy(dtype=float)  # (N,)
y = df.iloc[:, 1].to_numpy(dtype=float)  # (N,)
N = len(x)

# --- x'i standardize et (GD'nin stabil yakınsaması için önerilir) ---
xm, xs = x.mean(), x.std()
xs = xs if xs != 0 else 1.0
x_s = (x - xm) / xs

# --- GD ayarları ---
w0, w1 = 0.0, 0.0           # b ve m'nin standardize edilmiş x üzerindeki halleri
mu = 0.1                     # öğrenme oranı (0.05–0.1 genelde iyi)
epochs = 50

for ep in range(1, epochs+1):
    y_hat = w1 * x_s + w0
    err = y - y_hat
    mse = np.mean(err**2)

    # gradyanlar (MSE'ye göre)
    dw0 = -(2.0/N) * np.sum(err)
    dw1 = -(2.0/N) * np.sum(x_s * err)

    # güncelle
    w0 -= mu * dw0
    w1 -= mu * dw1

    if ep in (1, 10, 20, 30, 40, 50):
        print(f"Epoch {ep:>2d} | MSE={mse:.6f} | w0={w0:.6f} | w1={w1:.6f}")

# --- Orijinal ölçeğe dönüştür ---
m_gd = w1 / xs
b_gd = w0 - m_gd * xm
print("\nGD (orijinal ölçekte): m =", m_gd, " b =", b_gd)

# --- Analitik (kapalı form) çözüm ---
x_mean, y_mean = x.mean(), y.mean()
m_star = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b_star = y_mean - m_star * x_mean
print("Analitik çözüm          : m* =", m_star, " b* =", b_star)

# --- MSE karşılaştırması ---
mse_gd = np.mean((y - (m_gd * x + b_gd))**2)
mse_star = np.mean((y - (m_star * x + b_star))**2)
print(f"\nMSE (GD)      : {mse_gd:.6f}")
print(f"MSE (Analitik): {mse_star:.6f}")
