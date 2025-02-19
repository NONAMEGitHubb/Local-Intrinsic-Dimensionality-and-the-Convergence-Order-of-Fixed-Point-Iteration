# -*- coding: utf-8 -*-
"""LID Esimators v2.0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QEvM5JfRq7o6lKY2I2aFaEugr0MoN3UZ
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import norm

"""**LID MLE**"""

class LIDCalculator:
    def __init__(self, distances):
        self.distances = distances

    def calculate_MLE_LID(self):
        k = self.distances.shape[0]
        D = self.distances[:k-1]
        last_value = self.distances[k-1]

        D_non_zero = D[D > 0]
        if D_non_zero.size == 0:
            return np.nan

        LID = - (k / np.sum(np.log(D_non_zero / last_value)))
        return LID

"""**Log-Log Estimator**"""

class Log_Log(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.m = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.m * x + self.a

"""**HS_Convergence Order Estimator**"""

class HS_ConvergenceOrderCalculator:
    def __init__(self, data):
        self.data = data

    def compute_order_of_accuracy(self):
        p_estimates = []
        for i in range(len(self.data) - 2):
            u_diff1 = np.abs(self.data[i][1] - self.data[i + 1][1])
            u_diff2 = np.abs(self.data[i + 1][1] - self.data[i + 2][1])

            # Prevent division by zero
            if u_diff2 == 0:
                continue

            if u_diff1 == 0:
                continue

            ratio = u_diff1 / u_diff2

            # Estimate p using log2(ratio)
            p_estimate = np.log2(ratio)
            p_estimates.append(p_estimate)

        return p_estimates[-1] if p_estimates else None

"""**IR_Convergence Order Estimator**"""

class IR_ConvergenceOrderCalculator:
    def __init__(self, data):
        self.data = np.array(data)

    def compute_order_of_accuracy(self):
        n = self.data.size
        p_estimates = []
        for i in range(0,n-3):
            N_diff1 = (self.data[i+3] - self.data[i+2])
            N_diff2 = (self.data[i+2] - self.data[i+1])
            D_diff1 = (self.data[i+2] - self.data[i+1])
            D_diff2 = (self.data[i+1] - self.data[i])

            # Prevent division by zero
            if N_diff1 == 0:
                continue
            if N_diff2 == 0:
                continue
            if D_diff1 == 0:
                continue
            if D_diff2 == 0:
                continue

            N = np.log(np.abs(N_diff1 / N_diff2))
            D = np.log(np.abs(D_diff1 / D_diff2))
            ratio = N / D

            # Estimate p using log2(ratio)
            p_estimate = ratio
            p_estimates.append(p_estimate)
        return p_estimates[-1] if p_estimates else None

"""**LID Calculator**"""

class LIDModelCalculator:
    def __init__(self, device='cpu'):
        self.device = device

    def train_model(self, X, G, lr=0.01, n_epochs=10):
    #def train_model(self, X, G, lr=0.01, n_epochs=2000):
        """ Train a simple linear regression model using log-log relationship. """
        model = Log_Log().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction='mean')
        x_train_tensor = torch.from_numpy(X).float().to(self.device).view(-1, 1)
        y_train_tensor = torch.from_numpy(G).float().to(self.device).view(-1, 1)

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            yhat = model(x_train_tensor)
            loss = loss_fn(y_train_tensor, yhat)
            loss.backward()
            optimizer.step()

            #if epoch % 10 == 0:
            #    print(f'Epoch {epoch}, Loss: {loss.item()}')

            # Check the parameters
            #print(f'a: {model.a.item()}, m: {model.m.item()}')

        return model.m.item()

    def calculate(self, k, sampled_points, y, noisy=False, method=''):
        """ Calculate LID and train the model for provided sample points and y values. """
        results = []
        for ki in k:
            #b_index = np.argmin(y)
            b_index = 0
            b = y[b_index]
            x_adjusted = np.delete(sampled_points, b_index)
            y_adjusted = np.delete(y, b_index)
            w = y_adjusted.shape[0]
            y_k = y_adjusted[w - 1]
            F = (y_adjusted - b) / (y_k - b)

            start_time_lid = time.time()
            lid_calculator_F = LIDCalculator(F)
            F_lid_value = lid_calculator_F.calculate_MLE_LID()
            time_lid = time.time() - start_time_lid

            lid_calculator_X = LIDCalculator(x_adjusted)
            X_lid_value = lid_calculator_X.calculate_MLE_LID()

            #print(X_lid_value,"&",F_lid_value)
            lid_value = X_lid_value / F_lid_value

            start_time_m = time.time()
            G = np.log(np.abs(y_adjusted-b)+1e-7)
            #print("G",G)
            X = np.log(np.abs(x_adjusted))
            #print("X",X)
            m_value = self.train_model(X, G)
            #print (m_value)
            time_m = time.time() - start_time_m

            start_time_p = time.time()
            data = list(zip(sampled_points[::-1], y[::-1]))
            order_calculator = HS_ConvergenceOrderCalculator(data)
            p_value = order_calculator.compute_order_of_accuracy()
            time_p = time.time() - start_time_p

            start_time_q = time.time()
            data = y[::-1]
            order_calculator = IR_ConvergenceOrderCalculator(data)
            q_value = order_calculator.compute_order_of_accuracy()
            time_q = time.time() - start_time_q

            results.append({
                'k': ki,
                'Type': 'Noisy' if noisy else 'Noiseless',
                'OUR': lid_value,
                'Log-Log': m_value,
                'HS-COE': p_value,
                'IR-COE': q_value,
                'Time OUR (s)': time_lid,
                'Time Log-Log (s)': time_m,
                'Time HS-COE (s)': time_p,
                'Time IR-COE (s)': time_q,
                'Method': method
            })
        return results

"""**Estimating LID for Monomial Function**"""

# Define parameters
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

import matplotlib.pyplot as plt
from google.colab import files

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

# Download the file
files.download('monomial.jpeg')

"""**Estimating LID for Polynomial Function**"""

# Define parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
f = lambda x: 4 * x**7 - 1* x**3 + 1   # Define the function
a, b = 0.01, 2.5  # Define the interval
total_area, _ = integrate.quad(f, a, b)  # Compute the area under the curve
import numpy as np

class RootFinder:
    def __init__(self, a, b, k):
        self.a = a
        self.b = b
        self.k = k
        self.normalization_value = self.f(b)  # Using the primary function to get the normalization value.

    def f(self, x):
        """ Define the primary function of the equation to solve. """
        return 0.5* x**8 - 0.25 * x**4 + 1 * x

    def normalized_f(self, x, val):
        """ Define the normalized function to find roots for, incorporating the value val. """
        return self.f(x) / self.normalization_value - val

    def bisection(self, a, b, val, tol=1e-5, max_iter=1000):
        """ Bisection method to find the root of normalized_f(x) = val between a and b. """
        if self.normalized_f(a, val) * self.normalized_f(b, val) > 0:
            #print("No sign change over the interval [{}, {}] for value {}".format(a, b, val))
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
    #noise = np.random.normal(0, 0.1, y_UniInt_noiseless.shape)
    mean = 0
    std_dev = 1e-5
    noise = norm.rvs(mean, std_dev, size=y_UniInt_noiseless.shape)
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

import matplotlib.pyplot as plt
from google.colab import files

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

# First subplot
ax1.scatter(Uni_Int_points, y_UniInt_noiseless, alpha=0.6, color='blue', s=15)
ax1.set_xlabel('r')
ax1.set_ylabel('Phi(r)')
ax1.set_xlim(0, 2.5)
ax1.set_ylim(0, 2500)
ax1.grid(True)

# Second subplot
ax2.scatter(Uni_AUC_points, y_UniAUC_noiseless, alpha=0.6, color='blue', s=15)
ax2.set_xlabel('r')
ax2.set_ylabel('Phi(r)')
ax2.set_xlim(0, 2.5)
ax2.set_ylim(0, 2500)
ax2.grid(True)

# Save the plot as a JPEG file
plt.tight_layout()
plt.savefig("polynomial.jpeg", format='jpg', dpi=300)

# Show the combined plot
plt.show()

# Download the file
files.download('polynomial.jpeg')

"""**Estimating LID for Decreasing Function**"""

# Define parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
f = lambda x: 1/(x**5 + 5)  # Define the function
a, b =1e-5, 2.5  # Define the interval
#total_area, _ = integrate.quad(f, a, b)  # Compute the area under the curve

class RootFinder:
    def __init__(self, a, b, k):
        self.a = a
        self.b = b
        self.k = k
        self.normalization_value = self.f(b)  # Using the primary function to get the normalization value.

    def f(self, x):
        """ Define the primary function of the equation to solve. """
        return (1/6)*x**(6) + 5*x

    def normalized_f(self, x, val):
        """ Define the normalized function to find roots for, incorporating the value val. """
        return self.f(x) / self.normalization_value - val

    def bisection(self, a, b, val, tol=1e-5, max_iter=1000):
        """ Bisection method to find the root of normalized_f(x) = val between a and b. """
        if self.normalized_f(a, val) * self.normalized_f(b, val) > 0:
            #print("No sign change over the interval [{}, {}] for value {}".format(a, b, val))
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
    Uni_AUC_points = (b - Uni_AUC_points)
    Uni_AUC_points = np.sort(Uni_AUC_points)
    y_UniAUC_noiseless = f(Uni_AUC_points)
    noise = np.random.normal(0, 0.001, y_UniAUC_noiseless.shape)
    y_UniAUC_noisy = y_UniAUC_noiseless + noise

    results += calculator.calculate([k], Uni_AUC_points, y_UniAUC_noiseless, noisy=False, method='Uniform AUC')
    results += calculator.calculate([k], Uni_AUC_points, y_UniAUC_noisy, noisy=True, method='Uniform AUC')

    # Uniform Intervals
    Uni_Int_points = np.linspace(a, b, k)
    y_UniInt_noiseless = f(Uni_Int_points)
    noise = np.random.normal(0, 0.001, y_UniInt_noiseless.shape)
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

import matplotlib.pyplot as plt
from google.colab import files

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

# First subplot
ax1.scatter(Uni_Int_points, y_UniInt_noiseless, alpha=0.6, color='blue', s=15)
ax1.set_xlabel('r')
ax1.set_ylabel('Phi(r)')
ax1.set_xlim(0, 2.5)
#ax1.set_ylim(0, 65)
ax1.grid(True)

# Second subplot
ax2.scatter(Uni_AUC_points, y_UniAUC_noiseless, alpha=0.6, color='blue', s=15)
ax2.set_xlabel('r')
ax2.set_ylabel('Phi(r)')
ax2.set_xlim(0, 2.5)
#ax2.set_ylim(0, 65)
ax2.grid(True)

# Save the plot as a JPEG file
plt.tight_layout()
plt.savefig("decreasing.jpeg", format='jpg', dpi=300)

# Show the combined plot
plt.show()

# Download the file
files.download('decreasing.jpeg')

"""**Estimating LID for Periodic Function**"""

# Define parameters
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

import matplotlib.pyplot as plt
from google.colab import files

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

# Download the file
files.download('cos.jpeg')

"""**Estimating LID for Distance Function in Swiss Roll Manifold**"""

pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_swiss_roll(n_samples=5000):
    """
    Generate a Swiss Roll dataset.

    Args:
    n_samples (int): Number of samples to generate.

    Returns:
    numpy.ndarray: Coordinates of points (n_samples, 3).
    """
    # Generate random values for t and height (z axis)
    t = 4.5 * np.pi * (1 + 2 * np.random.rand(n_samples))  # t is the angle
    z = 5 * np.random.rand(n_samples)  # z is the height

    # Convert polar to cartesian coordinates
    x = t * np.cos(t)
    y = t * np.sin(t)

    # Scale and return the points in 3D space
    data = np.vstack((x, y, z)).T
    return data

# Generate Swiss Roll data
swiss_roll = generate_swiss_roll(5000)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(swiss_roll[:, 0], swiss_roll[:, 1], swiss_roll[:, 2], c=swiss_roll[:, 2], cmap=plt.cm.Spectral)
ax.set_title('Swiss Roll Manifold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

class SwissRollAnalyzer:
    def __init__(self, n_samples=5000, k=32):
        self.n_samples = n_samples
        self.k = k
        self.data = self.generate_swiss_roll()

    def generate_swiss_roll(self):
        """Generate a Swiss Roll dataset."""
        t = 0.35 * np.pi * (1 + 2 * np.random.rand(self.n_samples))
        z = 4 * np.random.rand(self.n_samples)
        x = t * np.cos(t)
        y = t * np.sin(t)
        return np.vstack((x, y, z)).T

    def calculate_distances(self, p_0):
        """Calculate the Euclidean distance from all points in 'data' to the point 'p_0'."""
        return np.sqrt(np.sum((self.data - p_0) ** 2, axis=1))

    def count_proportions(self, distances, radii):
        """Count the proportion of points within various radii from p_0."""
        proportions = []
        total_points = len(distances)
        for r in radii:
            count = np.sum(distances <= r)
            proportions.append(count / total_points)
        return proportions

    def analyze_point(self, index=0, offset=1, max_radius=2.5):
        """Analyze and plot proportions of points within various radii from a given point in the dataset."""
        p_0 = self.data[index]
        distances = self.calculate_distances(p_0)
        radii = np.linspace(1e-5, max_radius, self.k)
        proportions = self.count_proportions(distances, radii)
        y = [prop + offset for prop in proportions]
        return radii,y

        # Plotting the proportions
        plt.figure(figsize=(10, 6))
        plt.plot(radii, y, marker='o')
        plt.xlabel('Radius')
        plt.ylabel('Proportion of Points + Offset')
        plt.title(f'Proportion of Points within Various Radii from $p_0$ (Index {index})')
        plt.grid(True)
        plt.show()

# Define parameters
import random


# Generate a random integer between 0 and 9999
index = random.randint(0, 9999)
print(index)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)

k_values = [32, 64, 128, 256, 512]
results = []
for k in k_values:
    analyzer = SwissRollAnalyzer(n_samples=10000, k=k)
    x,y = analyzer.analyze_point(index=index, offset=1, max_radius=1.5)
    noise = np.random.normal(0, 0.001, k)
    y_noisy = y + noise

    results += calculator.calculate([k], x, y, noisy=False, method='Uniform Sampling')
    results += calculator.calculate([k], x, y_noisy, noisy=True, method='Uniform Sampling')

df_results = pd.DataFrame(results)
methods = ['Uniform Sampling']

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

plt.figure(figsize=(5, 3))
plt.scatter(x, y, alpha=0.6, color='blue', s=20)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

"""**Quadratic Convergence Function**"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
U = lambda x: (-2*x**3 + 4*x**2 -1)/(-3*x**2 + 8*x - 2)
x = np.zeros(64)
y = np.zeros(64)

# Set initial value
x[0] = float(input("Enter the starting point: "))

for i in range(0, 63):
    x[i+1] = U(x[i])

y = U(x)

k_values = [16, 32, 64]
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
#print(x_points)

plt.figure(figsize=(5, 3))
plt.scatter(x_points, y_noiseless, alpha=0.6, color='blue', s=20)
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(3, 5)
plt.ylim(1, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

"""**Linear Convergence Function**"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
calculator = LIDModelCalculator(device=device)
U = lambda x: (x+3)/(2)

x = np.zeros(64)
y = np.zeros(64)

# Set initial value
x[0] = float(input("Enter the starting point: "))

for i in range(0, 63):
    x[i+1] = U(x[i])

y = U(x)

k_values = [16, 32, 64]
results = []

for k in k_values:
    x_points = x[:k]
    y_noiseless = y[:k]
    x_points = x_points[::-1]
    y_noiseless = y_noiseless[::-1]
    noise = np.random.normal(0.1, 0.01, y_noiseless.shape)
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

"""**Super-Linear Convergence Function**"""

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