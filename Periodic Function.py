import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import integrate
from LID_Calculator import LIDModelCalculator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
f = lambda x: np.cos(x) + 1  # Define the function
a, b = 1e-5, 1.5  # Define the interval
#total_area, _ = integrate.quad(f, a, b)  # Compute the area under the curve
import numpy as np

class RootFinder:
    def __init__(self, a, b, k):
        self.a = a
        self.b = b
        self.k = k
        self.normalization_value = self.f(b)  # Using the primary function to get the normalization value.

    def f(self, x):
        """ Define the primary function of the equation to solve. """
        return 1*x + np.sin(x)
    def normalized_f(self, x, val):
        """ Define the normalized function to find roots for, incorporating the value val. """
        return self.f(x) / self.normalization_value - val

    def bisection(self, a, b, val, tol=1e-5, max_iter=1000):
        """ Bisection method to find the root of normalized_f(x) = val between a and b. """
        if self.normalized_f(a, val) * self.normalized_f(b, val) > 0:
            print("No sign change over the interval [{}, {}] for value {}".format(a, b, val))
            return None

        iter_count = 0
        while (b - a) / 2.0 > tol:
            midpoint = (a + b) / 2.0
            if self.normalized_f(midpoint, val) == 0:
                return midpoint  # The midpoint is a root
            elif self.normalized_f(a, val) * self.normalized_f(midpoint, val) < 0:
                b = midpoint
            else:
                a = midpoint
            iter_count += 1
            if iter_count >= max_iter:
                print("Maximum iterations reached")
                return midpoint
        return (a + b) / 2.0

    def find_roots(self):
        """ Find roots for the equation given k divisions between a and b. """
        roots = []
        for i in range(self.k + 1):
            val = i / self.k
            root = self.bisection(self.a, self.b, val)
            if root is not None:
                roots.append(root)
        return np.array(roots)


k_values = [32, 64, 128, 256, 512]
results = []
for k in k_values:
    #Uniform Area Under Curve
    root_finder = RootFinder(a, b, k)
    Uni_AUC_points = root_finder.find_roots()
    Uni_AUC_points = np.sort(Uni_AUC_points)
    y_UniAUC_noiseless = f(Uni_AUC_points)
    noise = np.random.normal(0, 0.1, y_UniAUC_noiseless.shape)
    y_UniAUC_noisy = y_UniAUC_noiseless + noise

    results += calculator.calculate([k], Uni_AUC_points, y_UniAUC_noiseless, noisy=False, method='Uniform AUC')
    results += calculator.calculate([k], Uni_AUC_points, y_UniAUC_noisy, noisy=True, method='Uniform AUC')

    # Uniform Intervals
    Uni_Int_points = np.linspace(a, b, k)
    y_UniInt_noiseless = f(Uni_Int_points)
    noise = np.random.normal(0, 0.005, y_UniInt_noiseless.shape)
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
ax1.set_xlim(0, 1.5)
ax1.set_ylim(1, 2.02)
ax1.grid(True)

# Second subplot
ax2.scatter(Uni_AUC_points, y_UniAUC_noiseless, alpha=0.6, color='blue', s=15)
ax2.set_xlabel('r')
ax2.set_ylabel('Phi(r)')
ax2.set_xlim(0, 1.5)
ax2.set_ylim(1, 2.02)
ax2.grid(True)

# Save the plot as a JPEG file
plt.tight_layout()
plt.savefig("cos.jpeg", format='jpg', dpi=300)

# Show the combined plot
plt.show()