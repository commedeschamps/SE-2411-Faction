import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from sklearn.metrics import mean_squared_error
import os

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# File path
file_path = '/home/nurassyl/Desktop/AITU/Commath/SE-2411.csv'

# Define models
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, b):
    # Depending on how the data was generated, it might be a * exp(b*x)
    return a * np.exp(b * x)

def main():
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)
    # The columns seem to be "X values" and "Noisy Y" based on previous file view
    x_data = df['X values'].values
    y_data = df['Noisy Y'].values

    best_model = None
    best_params = None
    min_mse = float('inf')
    results = []

    print("Fitting models...")
    # 3. Exponential
    try:
        # standard curve_fit fails with overflow for very large numbers
        # Use log-transformation: ln(y) = ln(a) + b*x
        # Linear fit to log(y)
        
        # Filter positive values for log
        valid_indices = y_data > 0
        x_valid = x_data[valid_indices]
        y_valid = y_data[valid_indices]
        
        poly_coeffs = np.polyfit(x_valid, np.log(y_valid), 1) 
        # poly_coeffs is [slope, intercept] -> [b, ln(a)]
        b_est = poly_coeffs[0]
        a_est = np.exp(poly_coeffs[1])
        
        popt_exp = [a_est, b_est]
        y_pred_exp = exponential_model(x_data, *popt_exp)
        mse_exp = calculate_mse(y_data, y_pred_exp)
        results.append(('Exponential', mse_exp, popt_exp))
        print(f"Exponential MSE: {mse_exp:.4e}")
    except Exception as e:
        print(f"Exponential fit failed: {e}")

    # Determine best model
    for name, mse, params in results:
        if mse < min_mse:
            min_mse = mse
            best_model = name
            best_params = params

    print("-" * 30)
    print(f"Best Model: {best_model}")
    print(f"Parameters: {best_params}")

    if best_model == 'Exponential':
        a, b = best_params
        print(f"\nFinal Answer for Exponential:")
        print(f"model_type = 'exponential'")
        print(f"a = {a:.4f}")
        print(f"b = {b:.4f}")
    # Plotting to match user request
    plt.figure(figsize=(10, 6))
    
    # "without changing initial points" -> Plot original data
    plt.scatter(x_data, y_data, color='red', label='Corrupted Signal')
    
    # Plot the fitted exponential model
    # Use the best params found (or the exponential ones specifically if we assume it's obvious)
    if 'Exponential' in [r[0] for r in results]:
        # Find exponential params
        for name, mse, params in results:
            if name == 'Exponential':
                # Generate smooth line or just connect points? 
                # The user image shows a dashed line connecting points or smooth curve. 
                # "True Signal (Secret)" implies the underlying model.
                # Let's use smooth line for the model.
                x_smooth = np.linspace(min(x_data), max(x_data), 500)
                y_smooth = exponential_model(x_smooth, *params)
                plt.plot(x_smooth, y_smooth, color='blue', linestyle='--', label='True Signal (Secret )')

    plt.title('Generated Signal: exponential')
    # User image has linear scale (1e44 visible)
    # plt.yscale('log') # Removed log scale
    plt.grid(True)
    plt.legend()
    
    output_plot = '/home/nurassyl/Desktop/AITU/Com math/task two/key_solution.png'
    plt.savefig(output_plot)
    print(f"Comparison plot saved to {output_plot}")
    plt.show()

if name == "main":
    main()