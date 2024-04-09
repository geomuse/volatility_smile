import numpy as np
import matplotlib.pyplot as pt
from scipy.stats import norm
from scipy.optimize import minimize

# Heston model parameters
r = 0.05  # risk-free rate

v0, kappa, theta, sigma, rho = [3.994e-02, 2.070e+00, 3.998e-02, 1.004e-01, -7.003e-01]

# Time to maturity and strike price ranges for the volatility surface
T = np.linspace(0.1, 2, 20)  # Time to maturity from 0.1 to 2 years
K = np.linspace(80, 120, 20)  # Strike prices from 80 to 120

# Define a placeholder for the implied volatilities
IV = np.zeros((len(T), len(K)))

# Simplified method to estimate option prices under Heston model parameters
def simplified_option_price(S, K, T, r, kappa, theta, sigma, rho, v0):
    """A placeholder function to simulate option prices. Not an accurate representation of the Heston model."""
    return S * np.exp(-r * T) * norm.cdf((np.log(S / K) + (r + v0 / 2) * T) / np.sqrt(v0 * T))

# Estimating implied volatility using a simplified method (placeholder)
def estimate_iv(S, K, T, r, market_price):
    """Estimate implied volatility through numerical optimization."""
    objective = lambda iv: (simplified_option_price(S, K, T, r, kappa, theta, sigma, rho, iv) - market_price)**2
    result = minimize(objective, 0.2, bounds=[(0.01, 1)])
    return result.x[0]

# Spot price of the underlying asset
S = 100

# Compute the market price of options across different strikes and maturities
for i, ti in enumerate(T):
    for j, kj in enumerate(K):
        market_price = simplified_option_price(S, kj, ti, r, kappa, theta, sigma, rho, v0)
        IV[i, j] = estimate_iv(S, kj, ti, r, market_price)

# Plotting the implied volatility surface
T_mesh, K_mesh = np.meshgrid(T, K, indexing='ij')
fig = pt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T_mesh, K_mesh, IV, cmap='viridis')

ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Strike Price')
ax.set_zlabel('Implied Volatility')
ax.set_title('Heston Model Implied Volatility Surface')

pt.colorbar(surf)
pt.show()
