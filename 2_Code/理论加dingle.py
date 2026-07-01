import numpy as np
import matplotlib.pyplot as plt

# x = 1/B
x = np.linspace(0.06, 0.25, 2000)
B = 1 / x

# 三个频率 & 有效质量
F = [46, 79, 139]
m = [0.22, 0.15, 0.27]

# 温度范围
T_list = [2, 3, 4, 5, 6, 7]

# 颜色
colors = ['black', 'red', 'green', 'blue', 'orange', 'purple']

# Dingle 温度（可调）
T_D = 2

# 温度阻尼项
def RT(T, m, B):
    X = 14.69 * m * T / B
    return X / np.sinh(X)

# Dingle 因子
def RD(m, B):
    return np.exp(-14.69 * m * T_D / B)

plt.figure(figsize=(8,6))

for T, c in zip(T_list, colors):
    total = 0
    
    for Fi, mi in zip(F, m):
        total += RT(T, mi, B) * RD(mi, B) * \
                 np.sin(2*np.pi*(Fi/B - 0.5) + np.pi/4)
    
    # 归一化
    total = total / np.max(np.abs(total))
    
    plt.plot(x, total, color=c, label=f"T = {T} K", linewidth=1.5)

# 图像设置
plt.xlabel("1/B (T$^{-1}$)", fontsize=12)
plt.ylabel("Normalized ΔR", fontsize=12)
plt.xlim(0.06, 0.25)

plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(direction='in')
plt.minorticks_on()

plt.tight_layout()
plt.show()
