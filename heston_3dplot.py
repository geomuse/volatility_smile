import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Heston模型参数
v0, kappa, theta, sigma, rho = [3.994e-02, 2.070e+00, 3.998e-02, 1.004e-01, -7.003e-01]

# 时间和波动率的网格
T = np.linspace(0.01, 2, 50) # 时间从0.01到2年
V = np.linspace(0.01, 0.1, 50) # 波动率从0.01到0.1

T, V = np.meshgrid(T, V)

# 使用Heston模型中的波动率动态方程计算波动率的变化
# 这里我们使用一个简化的形式来模拟波动率的变化，而不是完整的Heston模型
Z = theta + (V - theta) * np.exp(-kappa * T)

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(T, V, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Volatility')
ax.set_zlabel('Price')
ax.set_title('Simplified Volatility Surface under Heston Model')

plt.colorbar(surf)
plt.show()

"""
后续可以计算隐含波动率通过价格计算.
"""

