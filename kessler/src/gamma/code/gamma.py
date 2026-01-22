import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Define the file path
file_path = '/home/nurassyl/Desktop/AITU/Com math/Gamma_Thermal.csv'
save_path = '/home/nurassyl/Desktop/AITU/Com math/task two/gamma_thermal_fit.png'

def model_func(x, A, B, C, E):
    # A*ln(B+C*x)+E (Removed D)
    # Protect against invalid log arguments
    val = B + C * x
    # We will enforce bounds B, C > 0 in curve_fit
    return A * np.log(val) + E

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
    # Formula: A*ln(B+C*x)+E
    p0 = [100.0, 1.0, 1.0, 100.0]
    
    # Bounds to ensure log(B+Cx) is valid for x>=0.
    # A, E can be anything
    lower_bounds = [-np.inf, 1e-6, 1e-6, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    try:
        # Fit the model
        popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
        
        A_opt, B_opt, C_opt, E_opt = popt
        
        print(f"Optimized Parameters:")
        print(f"A = {A_opt:.3f}")
        print(f"B = {B_opt:.3f}")
        print(f"C = {C_opt:.3f}")
        print(f"E = {E_opt:.3f}")
        
        print(f"\nEquation: y = {A_opt:.3f} * ln({B_opt:.3f} + {C_opt:.3f}*x) + {E_opt:.3f}")

        # Calculate value at x=100
        val_100 = model_func(100, *popt)
        print(f"\nValue at x=100: {val_100:.3f}")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.scatter(x_data, y_data, label='Data', color='blue', alpha=0.5, s=10)
        
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = model_func(x_fit, *popt)
        
        plt.plot(x_fit, y_fit, label='Fitted Model', color='red', linewidth=2)
        
        plt.title('Gamma Thermal Sensor Reading Curve Fit')
        plt.xlabel('Time')
        plt.ylabel('Sensor Reading')
        plt.legend()
        plt.grid(True)
        
        # Show plot
        plt.show() # As requested in fit_alpha task

    except Exception as e:
        print(f"An error occurred during fitting: {e}")

if __name__ == "__main__:":
    main()