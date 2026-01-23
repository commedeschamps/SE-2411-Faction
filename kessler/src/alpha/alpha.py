import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Define the file path relative to the repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
file_path = REPO_ROOT / "kessler" / "data" / "raw" / "Alpha_Altitude.csv"

def model_func(x, A, B, C, D, E):
    return A * np.exp(-B * x) + C * np.sin(D * x) + E

def main():
    if not file_path.exists():
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

    # Initial guess provided by user
    # A=8000, B=0.035, C=400, D=0.5, E=2500
    p0 = [8000, 0.035, 400, 0.5, 2500]

    try:
        # Fit the model
        popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=10000)
        
        A_opt, B_opt, C_opt, D_opt, E_opt = popt
        
        print(f"Optimized Parameters:")
        print(f"A = {A_opt:.3f}")
        print(f"B = {B_opt:.3f}")
        print(f"C = {C_opt:.3f}")
        print(f"D = {D_opt:.3f}")
        print(f"E = {E_opt:.3f}")
        
        print(f"\nEquation: y = {A_opt:.3f} * e^(-{B_opt:.3f}*x) + {C_opt:.3f} * sin({D_opt:.3f}*x) + {E_opt:.3f}")

        # Calculate value at x=100
        val_100 = model_func(100, *popt)
        print(f"\nValue at x=100: {val_100:.3f}")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.scatter(x_data, y_data, label='Data', color='blue', alpha=0.5, s=10)
        
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = model_func(x_fit, *popt)
        
        plt.plot(x_fit, y_fit, label='Fitted Model', color='red', linewidth=2)
        
        plt.title('Alpha Altitude Sensor Reading Curve Fit')
        plt.xlabel('Time')
        plt.ylabel('Sensor Reading')
        plt.legend()
        plt.grid(True)
        
        # Show plot instead of saving
        plt.show()

    except Exception as e:
        print(f"An error occurred during fitting: {e}")

if __name__ == "__main__":
    main()
