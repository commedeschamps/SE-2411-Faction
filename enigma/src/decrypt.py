import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Load the data
df = pd.read_csv('faction_signal.csv')
x = df['x'].values
y = df['y'].values

# 2. Define the Quadratic Model
# Equation: y = ax^2 + bx + c
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# 3. Fit the Model (Finding the Secret Key)
popt, _ = curve_fit(quadratic_model, x, y)
a, b, c = popt

# 4. Generate points for a smooth curve visualization
x_smooth = np.linspace(min(x), max(x), 100)
y_fit = quadratic_model(x_smooth, *popt)

# 5. Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Intercepted Signal (Noisy)', alpha=0.7)
plt.plot(x_smooth, y_fit, color='blue', linewidth=2.5, 
         label=f'Quadratic Fit: y = {a:.2f}x² + {b:.2f}x + {c:.2f}')

plt.title('Final Decryption: Quadratic Model Reconstruction', fontsize=14)
plt.xlabel('x (Independent Variable)', fontsize=12)
plt.ylabel('y (Signal Value)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Save the plot for submission
plt.savefig('final_quadratic_decryption.png')
plt.show()

print(f"Decrypted Constants: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")