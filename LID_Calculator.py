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