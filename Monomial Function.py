import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import integrate
from LID_Calculator import LIDModelCalculator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
f = lambda x: 4 * x**(3)  # Define the function
a, b = 1e-5, 2.5  # Define the interval
total_area, _ = integrate.quad(f, a, b)  # Compute the area under the curve

def inverse_cdf(y, total_area):
        """ Calculate inverse CDF for transformation sampling. """
        return (total_area * y) ** (1 / 4)

k_values = [32, 64, 128, 256, 512]
results = []
for k in k_values:
    #Uniform Area Under Curve
    uniform_samples = np.linspace(1e-5, 1, k, endpoint=False)
    Uni_AUC_points = inverse_cdf(uniform_samples, total_area)
    Uni_AUC_points = np.sort(Uni_AUC_points)
    y_UniAUC_noiseless = f(Uni_AUC_points)
    noise = np.random.normal(0, 0.2, y_UniAUC_noiseless.shape)
    y_UniAUC_noisy = y_UniAUC_noiseless + noise

    results += calculator.calculate([k], Uni_AUC_points, y_UniAUC_noiseless, noisy=False, method='Uniform AUC')
    results += calculator.calculate([k], Uni_AUC_points, y_UniAUC_noisy, noisy=True, method='Uniform AUC')

    # Uniform Intervals
    Uni_Int_points = np.linspace(a, b, k)
    y_UniInt_noiseless = f(Uni_Int_points)
    noise = np.random.normal(0, 0.2, y_UniInt_noiseless.shape)
    y_UniInt_noisy = y_UniInt_noiseless + noise

    results += calculator.calculate([k], Uni_Int_points, y_UniInt_noiseless, noisy=False, method='Uniform Interval')
    results += calculator.calculate([k], Uni_Int_points, y_UniInt_noisy, noisy=True, method='Uniform Interval')

df_results = pd.DataFrame(results)

methods = ['Uniform AUC', 'Uniform Interval']

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

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

# First subplot
ax1.scatter(Uni_Int_points, y_UniInt_noiseless, alpha=0.6, color='blue', s=15)
ax1.set_xlabel('r')
ax1.set_ylabel('Phi(r)')
ax1.set_xlim(0, 2.5)
ax1.set_ylim(0, 65)
ax1.grid(True)

# Second subplot
ax2.scatter(Uni_AUC_points, y_UniAUC_noiseless, alpha=0.6, color='blue', s=15)
ax2.set_xlabel('r')
ax2.set_ylabel('Phi(r)')
ax2.set_xlim(0, 2.5)
ax2.set_ylim(0, 65)
ax2.grid(True)

# Save the plot as a JPEG file
plt.tight_layout()
plt.savefig("monomial.jpeg", format='jpg', dpi=300)

# Show the combined plot
plt.show()
