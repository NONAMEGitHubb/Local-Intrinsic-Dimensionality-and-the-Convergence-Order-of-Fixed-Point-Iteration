import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import integrate
from LID_Calculator import LIDModelCalculator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
f = lambda x: (x**2 - 6 * x + 9)

x = np.zeros(64)
y = np.zeros(64)

# Set initial values
x[0] = float(input("Enter the first starting point: "))
x[1] = float(input("Enter the second starting point: "))

for i in range(0, 62):
    denom = f(x[i+1]) - f(x[i])
    if denom == 0:
        denom += 1e-7  # Avoid division by zero
    x[i+2] = x[i] - (f(x[i]) *((x[i+1] - x[i])/(denom)))
    y[i] = x[i+2]


k_values = [16,32,64]
results = []

for k in k_values:
    x_points = x[:k]
    y_noiseless = y[:k]
    x_points = x_points[::-1]
    y_noiseless = y_noiseless[::-1]
    noise = np.random.normal(0, 0.1, y_noiseless.shape)
    y_noisy = y_noiseless + noise

    results += calculator.calculate([k], x_points, y_noiseless, noisy=False, method='NR')
    results += calculator.calculate([k], x_points, y_noisy, noisy=True, method='NR')

df_results = pd.DataFrame(results)

methods = ['NR']

for method in methods:
    df_method = df_results[df_results['Method'] == method]
    df_display = df_method[['k', 'Type', 'OUR', 'Log-Log', 'HS-COE', 'IR-COE']]
    print(f"Results for {method}:")
    print(df_display)
    print()

    # Convert each method's results to LaTeX format
    latex_method = df_display.to_latex(index=False)
    print(f"LaTeX format of Results for {method}:")
    print(latex_method)
    print()

    # Calculate average times for MLE, Log_Log, and Order
avg_times = df_results.groupby('Method').agg({
    'Time OUR (s)': 'mean',
    'Time Log-Log (s)': 'mean',
    'Time HS-COE (s)': 'mean',
    'Time IR-COE (s)': 'mean'
}).reset_index()

# Display average times
print("Average Times:")
print(avg_times)

#print(y_noiseless)
#print(y_noisy)