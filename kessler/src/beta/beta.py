import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Define the file path
file_path = '/home/nurassyl/Desktop/AITU/Com math/Beta_Drift.csv'
save_path = '/home/nurassyl/Desktop/AITU/Com math/task two/beta_drift_fit.png'

def model_func(x, A, B, C, D):
    # A*x^3 + B*x^2 + C*x + D
    return A * x**3 + B * x**2 + C * x + D

def main():
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path)
    
    # Check for correct columns
    if 'Time' not in df.columns or 'Sensor_Reading' not in df.columns:
        print("Error: CSV must contain 'Time' and 'Sensor_Reading' columns")
        return

    x_data = df['Time'].values
    y_data = df['Sensor_Reading'].values

    # Initial guess
    # A*x^3 + B*x^2 + C*x + D
    # Just uniform ones
    p0 = [1.0, 1.0, 1.0, 1.0]

    try:
        # Fit the model
        popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=10000)
        
        A_opt, B_opt, C_opt, D_opt = popt
        
        print(f"Optimized Parameters:")
        print(f"A = {A_opt:.3f}")
        print(f"B = {B_opt:.3f}")
        print(f"C = {C_opt:.3f}")
        print(f"D = {D_opt:.3f}")
        
        print(f"\nEquation: y = {A_opt:.3f}*x^3 + {B_opt:.3f}*x^2 + {C_opt:.3f}*x + {D_opt:.3f}")

        # Calculate value at x=100
        val_100 = model_func(100, *popt)
        print(f"\nValue at x=100: {val_100:.3f}")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.scatter(x_data, y_data, label='Data', color='blue', alpha=0.5, s=10)
        
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = model_func(x_fit, *popt)
        
        plt.plot(x_fit, y_fit, label='Fitted Model', color='red', linewidth=2)
        
        plt.title('Beta Drift Sensor Reading Curve Fit')
        plt.xlabel('Time')
        plt.ylabel('Sensor Reading')
        plt.legend()
        plt.grid(True)
        
        # Show plot
        plt.show()

    except Exception as e:
        print(f"An error occurred during fitting: {e}")

if name == "main":
    main()
