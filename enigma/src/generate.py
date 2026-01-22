import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure your secret signal
# model_type = "linear"      # y = a*x + b
# model_type = "quadratic"   # y = a*x^2 + b*x + c
# model_type = "exponential" # y = a * e^(b*x)

model_type = "exponential"

a = 10.0
b = 10.0
c = 0.0

# Generate data
x_data = np.linspace(0, 10, 20)

if model_type == "linear":
    y_perfect = a * x_data + b
elif model_type == "quadratic":
    y_perfect = a * x_data**2 + b * x_data + c
elif model_type == "exponential":
    y_perfect = a * np.exp(b * x_data)

# Add noise
noise_intensity = 5.0
noise = np.random.normal(0, noise_intensity, len(x_data))
y_noisy = y_perfect + noise

# Visualize and export
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_noisy, color ="red", label="Corrupted Signal")
plt.plot ( x_data , y_perfect , color ="blue", linestyle ="--", label ="True Signal (Secret )")
plt.legend()
plt.title(f"Generated Signal: {model_type}")
plt.grid(True)
plt.show()

df = pd.DataFrame({
    "x": x_data,
    "y": y_noisy
})
df.to_csv("faction_signal.csv", index=False)

print("SUCCESS: faction_signal.csv created")