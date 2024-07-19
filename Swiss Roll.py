!pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import torch
import random
from scipy import integrate
from LID_Calculator import LIDModelCalculator  

def generate_swiss_roll(n_samples=5000):
    """
    Generate a Swiss Roll dataset.
    """
    t = 4.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    z = 5 * np.random.rand(n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    return np.vstack((x, y, z)).T

# Plotting function for Swiss Roll
def plot_swiss_roll(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], cmap=plt.cm.Spectral)
    ax.set_title('Swiss Roll Manifold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# SwissRollAnalyzer class definition
class SwissRollAnalyzer:
    def __init__(self, n_samples=5000, k=32):
        self.n_samples = n_samples
        self.k = k
        self.data = self.generate_swiss_roll()

    def generate_swiss_roll(self):
        t = 4.5 * np.pi * (1 + 2 * np.random.rand(self.n_samples))
        z = 4 * np.random.rand(self.n_samples)
        x = t * np.cos(t)
        y = t * np.sin(t)
        return np.vstack((x, y, z)).T

    def calculate_distances(self, p_0):
        return np.sqrt(np.sum((self.data - p_0) ** 2, axis=1))

    def count_proportions(self, distances, radii):
        proportions = []
        total_points = len(distances)
        for r in radii:
            count = np.sum(distances <= r)
            proportions.append(count / total_points)
        return proportions

    def analyze_point(self, index=0, offset=1, max_radius=2.5):
        p_0 = self.data[index]
        distances = self.calculate_distances(p_0)
        radii = np.linspace(1e-5, max_radius, self.k)
        proportions = self.count_proportions(distances, radii)
        y = [prop + offset for prop in proportions]
        return radii, y

# Define parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)

# Generate and plot the Swiss Roll data
swiss_roll = generate_swiss_roll(5000)
plot_swiss_roll(swiss_roll)

# Analyze the Swiss Roll data
index = random.randint(0, 9999)
k_values = [32, 64, 128, 256, 512]
results = []
for k in k_values:
    analyzer = SwissRollAnalyzer(n_samples=10000, k=k)
    x, y = analyzer.analyze_point(index=index, offset=1, max_radius=1.5)
    noise = np.random.normal(0, 0.001, k)
    y_noisy = y + noise

    results += calculator.calculate([k], x, y, noisy=False, method='Uniform Sampling')
    results += calculator.calculate([k], x, y_noisy, noisy=True, method='Uniform Sampling')

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Display and convert results to LaTeX format
methods = ['Uniform Sampling']
for method in methods:
    df_method = df_results[df_results['Method'] == method]
    df_display = df_method[['k', 'Type', 'OUR', 'Log-Log', 'HS-COE', 'IR-COE']]
    print(f"Results for {method}:")
    print(df_display)
    print()

    latex_method = df_display.to_latex(index=False)
    print(f"LaTeX format of Results for {method}:")
    print(latex_method)
    print()

# Calculate average times for each method
avg_times = df_results.groupby('Method').agg({
    'Time OUR (s)': 'mean',
    'Time Log-Log (s)': 'mean',
    'Time HS-COE (s)': 'mean',
    'Time IR-COE (s)': 'mean'
}).reset_index()

# Display average times
print("Average Times:")
print(avg_times)

# Plotting the analysis results
plt.figure(figsize=(5, 3))
plt.scatter(x, y, alpha=0.6, color='blue', s=20)
plt.xlabel('Radius')
plt.ylabel('Proportion of Points + Offset')
plt.title('Proportion of Points within Various Radii')
plt.grid(True)
plt.show()
