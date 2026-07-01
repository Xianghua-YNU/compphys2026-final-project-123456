import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ====== 常数 ======
kB = 1.380649e-23
e = 1.602176634e-19
hbar = 1.054571817e-34
m0 = 9.10938356e-31

# ====== 温度 ======
T = np.array([2,3,4,5,6,7])

# ====== 三组振幅（你填）=====
A1 = np.array([ 12.2908,9.98954,7.16527,5.91004,4.1841,3.19038 ])   # 50T
A2 = np.array([ 25.8368,23.1172,19.9791,17.2594,14.4874,12.0293 ])   # 70T（主峰）
A3 = np.array([ 5.17782,3.66109,2.45816,1.88285,1.20293,0.784519])   # 140T

# ====== 平均磁场 ======
B = 6.4

# ====== LK函数 ======
def LK(T, m_eff, A0):
    X = 2 * np.pi**2 * kB * T * m_eff / (e * hbar * B)
    return A0 * (X / np.sinh(X))

# ====== 拟合函数 ======
def fit_mass(A):
    popt, _ = curve_fit(LK, T, A, p0=[0.2*m0, max(A)])
    return popt

# ====== 分别拟合 ======
m1, A01 = fit_mass(A1)
m2, A02 = fit_mass(A2)
m3, A03 = fit_mass(A3)

# ====== 输出 ======
print("50T:  m*/m0 = {:.4f}".format(m1/m0))
print("70T:  m*/m0 = {:.4f}".format(m2/m0))
print("140T: m*/m0 = {:.4f}".format(m3/m0))

# ====== 画图 ======
T_fit = np.linspace(2,7,200)

plt.scatter(T, A1, color='black', label='50T data')
plt.scatter(T, A2, color='red', label='70T data')
plt.scatter(T, A3, color='blue', label='140T data')

plt.plot(T_fit, LK(T_fit, m1, A01), color='black')
plt.plot(T_fit, LK(T_fit, m2, A02), color='red')
plt.plot(T_fit, LK(T_fit, m3, A03), color='blue')

plt.xlabel('Temperature (K)')
plt.ylabel('FFT Amplitude')
plt.legend()
plt.show()
